"""Collect frozen-student rollouts and inspect group-relative rewards."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path


def advantages(rewards):
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards)  # population standard deviation
    return [(reward - mean) / (std + 1e-8) for reward in rewards]


def sampling_options(settings):
    return {
        "do_sample": True,
        **{key: settings[key] for key in
           ("temperature", "top_p", "top_k", "min_p", "max_new_tokens")},
    }


def make_generation(settings, defaults):
    from transformers import GenerationConfig
    return GenerationConfig(
        **sampling_options(settings), use_cache=True,
        eos_token_id=defaults.eos_token_id,
        pad_token_id=defaults.pad_token_id,
        bos_token_id=defaults.bos_token_id,
    )


def resolve_and_verify(model, requested, sampling):
    # Exercise the installed library's actual configuration resolution.
    effective, _ = model._prepare_generation_config(
        requested, use_model_defaults=False, **sampling,
    )
    actual = {key: getattr(effective, key) for key in sampling}
    if actual != sampling:
        raise RuntimeError(f"Sampling mismatch: expected {sampling}, got {actual}")
    return effective, actual


def check_sampling_without_weights(spec, settings):
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    class ConfigOnlyModel(GenerationMixin):
        pass

    model = ConfigOnlyModel()
    path = Path(spec["local_snapshot"]) / "generation_config.json"
    model.generation_config = GenerationConfig.from_dict(json.loads(path.read_bytes()))
    requested = make_generation(settings, model.generation_config)
    _, actual = resolve_and_verify(model, requested, sampling_options(settings))
    print("PASS: Effective sampling checked without loading model weights.")
    print("Verified sampler:", json.dumps(actual), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--name", default="rollout_fixed_check_01")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    experiment = args.project_root.resolve() / "experiments"
    config_bytes = (experiment / "configs/pilot.json").read_bytes()
    config = json.loads(config_bytes)
    settings = config["rollout"]

    # Verify the exact inputs before loading the GPU model.
    datasets = {}
    for split in ("train", "dev"):
        path = experiment / config["data"][f"{split}_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["data"][f"{split}_sha256"]
        with path.open(encoding="utf-8") as stream:
            datasets[split] = [json.loads(line) for line in stream if line.strip()]
    train_ids = {row["question_sha256"] for row in datasets["train"]}
    dev_ids = {row["question_sha256"] for row in datasets["dev"]}
    assert len(train_ids) == len(datasets["train"]) == 6961
    assert len(dev_ids) == len(datasets["dev"]) == 512
    assert not train_ids & dev_ids
    for filename, expected in config["reward"]["code_sha256"].items():
        content = (experiment / "reward" / filename).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected
    sys.path.insert(0, str(experiment / "reward"))
    from text_answer_reward import score_answer

    assert 1 <= args.questions <= settings["pilot_questions"]
    assert args.name and Path(args.name).name == args.name and args.name not in (".", "..")
    group_size = settings["answers_per_question"]
    assert group_size >= 2
    assert (settings["temperature"], settings["top_p"], settings["top_k"], settings["min_p"]) == (1.0, 1.0, 0, 0.0)
    print("PASS: Training data, development separation, and reward fingerprints.")
    print("Example advantages:", [round(x, 3) for x in advantages([1, 0, 1, 0])])
    spec = config["models"]["student"]
    check_sampling_without_weights(spec, settings)
    if args.check_only:
        return

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "CUDA is unavailable."
    run_dir = experiment / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=False)
    run_started = time.perf_counter()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        spec["local_snapshot"], local_files_only=True, trust_remote_code=False,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["local_snapshot"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    model.requires_grad_(False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - run_started
    torch.cuda.reset_peak_memory_stats()

    requested_generation = make_generation(settings, model.generation_config)
    requested_generation.pad_token_id = tokenizer.pad_token_id
    sampling = sampling_options(settings)
    generation, actual_sampling = resolve_and_verify(
        model, requested_generation, sampling,
    )
    print("Verified loaded-model sampler:", json.dumps(actual_sampling), flush=True)
    eos = generation.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    selected = datasets["train"][:args.questions]
    metadata = {
        "config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "questions": args.questions, "selection": "first questions in saved training order",
        "source_rows": [row["source_row"] for row in selected],
        "generation": generation.to_dict(), "group_size": group_size,
        "requested_generation": requested_generation.to_dict(),
        "model_generation_defaults": model.generation_config.to_dict(),
        "use_model_defaults": False, "verified_sampling": actual_sampling,
        "sampler_fix_version": 2,
        "advantage_std": "population", "advantage_epsilon": 1e-8,
        "seed_rule": "config.seed + zero-based question index",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "model_updates": 0,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    groups = []
    print(f"Student loaded; generating {args.questions} questions x {group_size} answers.", flush=True)

    with (run_dir / "rollouts.jsonl").open("w", encoding="utf-8") as stream, torch.inference_mode():
        for index, row in enumerate(selected):
            messages = [{"role": "user", "content": config["prompt_instruction"] + row["question"]}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=settings["enable_thinking"],
            )
            inputs = tokenizer([prompt] * group_size, add_special_tokens=False,
                               padding=True, return_tensors="pt").to(model.device)
            prompt_ids = inputs["input_ids"][0].tolist()
            seed = config["seed"] + index
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.generate(
                **inputs, generation_config=generation,
                use_model_defaults=False, **sampling,
            )
            torch.cuda.synchronize()
            generation_seconds = time.perf_counter() - started
            answers = []
            for sequence in output[:, len(prompt_ids):].cpu().tolist():
                end = next((i for i, token in enumerate(sequence) if token in eos_ids), None)
                token_ids = sequence if end is None else sequence[:end + 1]
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                grade = score_answer(text, row["gold"])
                answers.append({
                    "completion": text, "token_ids": token_ids, **grade,
                    "reward": int(grade["correct"]), "eos_terminated": end is not None,
                    "truncated": end is None and len(sequence) >= settings["max_new_tokens"],
                })
            rewards = [answer["reward"] for answer in answers]
            values = advantages(rewards)
            for answer, value in zip(answers, values):
                answer["advantage"] = value
            group = {**row, "prompt_token_ids": prompt_ids, "batch_seed": seed,
                     "generation_seconds": generation_seconds, "answers": answers}
            stream.write(json.dumps(group, ensure_ascii=False) + "\n")
            stream.flush()
            groups.append(group)
            print(f"Question {index + 1}/{args.questions}: rewards={rewards}; "
                  f"advantages={[round(value, 3) for value in values]}", flush=True)

    answers = [answer for group in groups for answer in group["answers"]]
    mixed = sum(0 < sum(a["reward"] for a in g["answers"]) < group_size for g in groups)
    summary = {
        "status": "complete", "questions": len(groups), "responses": len(answers),
        "correct": sum(a["reward"] for a in answers), "mixed_reward_groups": mixed,
        "invalid_answers": sum(not a["answer_valid"] for a in answers),
        "truncated_answers": sum(a["truncated"] for a in answers),
        "model_load_seconds": load_seconds,
        "generation_wall_seconds": sum(g["generation_seconds"] for g in groups),
        "load_and_rollout_wall_seconds": time.perf_counter() - run_started,
        "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "model_updates": 0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
