"""Inspect a fresh LoRA adapter, or take exactly one GRPO-style RL step.

This is an engineering check on one group, not a performance experiment.
No teacher is loaded. Use fresh rollouts for subsequent policy updates.
"""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_inputs(experiment):
    """Validate provenance and the first group, without loading model weights."""
    from transformers import AutoTokenizer

    config_path = experiment / "configs/pilot.json"
    config = json.loads(config_path.read_bytes())
    run = experiment / "runs/rollout_probe_32_fixed_01"
    metadata = json.loads((run / "metadata.json").read_bytes())
    summary = json.loads((run / "summary.json").read_bytes())
    assert summary["status"] == "complete" and summary["model_updates"] == 0
    assert metadata["sampler_fix_version"] == 2
    assert metadata["config_sha256"] == digest(config_path)
    assert metadata["script_sha256"] == digest(experiment / "collect_rollouts_fixed.py")
    expected = dict(do_sample=True, temperature=1.0, top_p=1.0,
                    top_k=0, min_p=0.0, max_new_tokens=1024)
    assert metadata["verified_sampling"] == expected
    assert metadata["use_model_defaults"] is False
    assert all(metadata["generation"][key] == value for key, value in expected.items())
    for filename, sha in config["reward"]["code_sha256"].items():
        assert digest(experiment / "reward" / filename) == sha
    sys.path.insert(0, str(experiment / "reward"))
    from text_answer_reward import score_answer

    data = {}
    for split in ("train", "dev"):
        path = experiment / config["data"][f"{split}_file"]
        assert digest(path) == config["data"][f"{split}_sha256"]
        data[split] = read_jsonl(path)
    assert len(data["train"]) == 6961 and len(data["dev"]) == 512
    assert not ({r["question_sha256"] for r in data["train"]}
                & {r["question_sha256"] for r in data["dev"]})
    groups = read_jsonl(run / "rollouts.jsonl")
    assert len(groups) == 32
    group = groups[0]  # Explicit engineering example; no automatic search for successes.
    for key in ("source_row", "question_sha256", "question", "gold"):
        assert group[key] == data["train"][0][key]
    answers = group["answers"]
    assert len(answers) == 4
    rewards = [answer["reward"] for answer in answers]
    assert 0 < sum(rewards) < len(rewards)
    mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"]["local_snapshot"],
        local_files_only=True, trust_remote_code=False,
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": config["prompt_instruction"] + group["question"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert tokenizer(text, add_special_tokens=False)["input_ids"] == group["prompt_token_ids"]
    eos = metadata["generation"]["eos_token_id"]
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    for answer in answers:
        assert answer["answer_valid"] and answer["eos_terminated"] and not answer["truncated"]
        tokens = answer["token_ids"]
        assert tokens and tokens[-1] in eos_ids
        assert not any(token in eos_ids for token in tokens[:-1])
        assert tokenizer.decode(tokens, skip_special_tokens=True) == answer["completion"]
        grade = score_answer(answer["completion"], group["gold"])
        assert all(answer[key] == value for key, value in grade.items())
        assert answer["reward"] == int(grade["correct"])
        assert math.isclose(answer["advantage"], (answer["reward"] - mean) / (std + 1e-8), abs_tol=1e-10)
    assert config["lora"]["lora_dropout"] == 0.0
    print("PASS: Corrected sampler, source data, tokens, rewards, and advantages.")
    print("Selected rewards:", rewards)
    print("Selected advantages:", [round(a["advantage"], 6) for a in answers])
    return config, group, run


def token_logprobs(model, prompt, completion):
    """Logit P-1+j predicts completion token j; include EOS, exclude prompt."""
    import torch
    import torch.nn.functional as F

    ids = torch.tensor([prompt + completion], device=next(model.parameters()).device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    logits = output.logits[0, len(prompt) - 1:-1, :].float()
    targets = ids[0, len(prompt):]
    assert logits.shape[0] == len(completion)
    values = -F.cross_entropy(logits, targets, reduction="none")
    assert torch.isfinite(values).all(), "Non-finite token log probabilities."
    return values


def surrogate_loss(current, old, advantage, group_size):
    """Original per-response averaging, clipping 0.2, reference KL beta=0."""
    import torch

    ratio = torch.exp(current - old.detach())
    unclipped = ratio * advantage
    clipped = ratio.clamp(0.8, 1.2) * advantage
    return -torch.minimum(unclipped, clipped).mean() / group_size


def base_fingerprint(model):
    """Hash every frozen parameter byte, moving only one tensor at a time to CPU."""
    import torch

    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            h.update(f"{name}:{parameter.dtype}:{tuple(parameter.shape)}".encode())
            raw = parameter.detach().contiguous().cpu().view(torch.uint8).numpy()
            h.update(memoryview(raw))
    return h.hexdigest()


def adapter_snapshot(model):
    from peft import get_peft_model_state_dict
    return {key: value.detach().cpu().clone() for key, value in
            get_peft_model_state_dict(model, save_embedding_layers=False).items()}


def apply_one_update(model, prompt, answers, old_logprobs, learning_rate):
    """Four backwards accumulate gradients; exactly one optimizer step follows."""
    import torch

    assert len(answers) == len(old_logprobs) and len(answers) >= 2
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert named and all("lora_A." in name or "lora_B." in name for name, _ in named)
    parameters = [p for _, p in named]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in parameters}
    optimizer.zero_grad(set_to_none=True)
    before_loss = 0.0
    max_logp_difference = 0.0
    for index, (answer, old) in enumerate(zip(answers, old_logprobs), start=1):
        current = token_logprobs(model, prompt, answer["token_ids"])
        difference = float((current.detach() - old).abs().max())
        max_logp_difference = max(max_logp_difference, difference)
        assert difference <= 1e-3, "Unexpected pre-update policy mismatch; stop before optimizer step."
        loss = surrogate_loss(current, old, answer["advantage"], len(answers))
        before_loss += float(loss.detach())
        loss.backward()
        print(f"Backward {index}/{len(answers)} complete; optimizer steps so far: 0", flush=True)
        del current, loss

    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    nonzero = {"A": 0, "B": 0}
    for name, p in named:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        if torch.count_nonzero(p.grad).item():
            nonzero["A" if "lora_A." in name else "B"] += 1
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    assert float(norm) > 0, "No learning signal reached the adapters."
    optimizer.step()  # THE SINGLE MODEL UPDATE
    assert all(torch.isfinite(p).all() for p in parameters)
    optimizer.zero_grad(set_to_none=True)
    return dict(optimizer_steps=1, loss_before=before_loss,
                gradient_norm_before_clipping=float(norm), gradient_clip_max_norm=1.0,
                nonzero_gradient_tensors=nonzero, max_preupdate_logp_difference=max_logp_difference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--inspect-adapter", action="store_true")
    mode.add_argument("--update", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    experiment = args.project_root.resolve() / "experiments"
    config, group, source_run = validate_inputs(experiment)
    if args.check_only:
        print("Model weights loaded: False. Optimizer steps: 0.")
        return

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM
    from importlib.metadata import version

    assert torch.cuda.is_available()
    name = args.name or ("rl_adapter_inspection_01" if args.inspect_adapter else "rl_one_update_01")
    assert Path(name).name == name and name not in (".", "..")
    run = experiment / "runs" / name
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    spec = config["models"]["student"]
    base = AutoModelForCausalLM.from_pretrained(
        spec["local_snapshot"], local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    base.config.use_cache = False
    base.requires_grad_(False)
    torch.manual_seed(config["seed"])
    lora = LoraConfig(**config["lora"], task_type=TaskType.CAUSAL_LM,
                      bias="none", init_lora_weights=True, use_rslora=False)
    model = get_peft_model(base, lora, autocast_adapter_dtype=True, revision=spec["revision"])
    model.eval()  # Disable stochastic layers; eval() does not disable gradients.
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_A." in n or "lora_B." in n for n, _ in trainable)
    assert all(p.dtype == torch.float32 for _, p in trainable)
    b_parameters = [p for n, p in trainable if "lora_B." in n]
    assert b_parameters and all(torch.count_nonzero(p).item() == 0 for p in b_parameters)
    model.print_trainable_parameters()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    prompt, answers = group["prompt_token_ids"], group["answers"]

    scoring_started = time.perf_counter()
    with torch.no_grad():
        with model.disable_adapter():
            base_scores = [token_logprobs(model, prompt, a["token_ids"]).detach() for a in answers]
        old = [token_logprobs(model, prompt, a["token_ids"]).detach() for a in answers]
    initial_difference = max(float((a - b).abs().max()) for a, b in zip(base_scores, old))
    assert initial_difference <= 1e-5, "Initial adapter changed the policy that generated these responses."
    torch.cuda.synchronize()
    scoring_seconds = time.perf_counter() - scoring_started
    print(f"PASS: Initial adapter matches base; maximum log-probability difference {initial_difference:.3g}", flush=True)
    before_state = adapter_snapshot(model)
    model.save_pretrained(run / "adapter_initial", safe_serialization=True, save_embedding_layers=False)
    metadata = {
        "config": config, "versions": {k: version(k) for k in ("torch", "transformers", "peft")},
        "script_sha256": digest(Path(__file__)), "config_sha256": digest(experiment / "configs/pilot.json"),
        "source_rollouts_sha256": digest(source_run / "rollouts.jsonl"),
        "source_metadata_sha256": digest(source_run / "metadata.json"),
        "source_run": str(source_run), "source_row": group["source_row"],
        "selection": "first corrected group, engineering check only", "seed": config["seed"],
        "objective": "GRPO-style clipped surrogate; per-response mean then group mean; KL beta=0",
        "clip_epsilon": 0.2, "reference_kl_beta": 0.0, "advantage_std": "population",
        "eos_in_loss": True, "prompt_in_loss": False, "microbatch_responses": 1,
        "optimizer": "AdamW", "weight_decay": 0.0, "learning_rate": config["learning_rate"],
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "teacher_loaded": False, "checkpoint_scope": "adapter only; optimizer state not saved",
    }
    save_json(run / "metadata.json", metadata)
    result = dict(status="complete", mode="inspect" if args.inspect_adapter else "update",
                  optimizer_steps=0, initial_adapter_max_logp_difference=initial_difference,
                  trainable_parameters=metadata["trainable_parameters"], model_load_seconds=load_seconds,
                  initial_scoring_seconds=scoring_seconds)
    if args.update:
        print("Hashing all frozen base parameters before the update...", flush=True)
        base_before = base_fingerprint(model)
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        result.update(apply_one_update(model, prompt, answers, old, config["learning_rate"]))
        torch.cuda.synchronize()
        result["update_wall_seconds"] = time.perf_counter() - update_started
        after_state = adapter_snapshot(model)
        changed = [key for key in before_state if not torch.equal(before_state[key], after_state[key])]
        assert changed, "No adapter weights changed."
        result["changed_adapter_tensors"] = len(changed)
        result["adapter_max_absolute_change"] = max(float((after_state[k] - before_state[k]).abs().max()) for k in changed)
        with torch.no_grad():
            after = [token_logprobs(model, prompt, a["token_ids"]).detach() for a in answers]
        result["loss_after_on_same_responses"] = sum(float(surrogate_loss(p, q, a["advantage"], len(answers)))
                                                   for p, q, a in zip(after, old, answers))
        result["mean_logp_changes_by_response"] = [float((p - q).mean()) for p, q in zip(after, old)]

        checkpoint = run / "adapter_after_one_update"
        model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
        loaded = load_peft_weights(str(checkpoint), device="cpu", local_files_only=True)
        assert loaded.keys() == after_state.keys()
        assert all(torch.equal(loaded[key], after_state[key]) for key in loaded)
        with torch.no_grad():
            for _, parameter in trainable:
                parameter.zero_()
        set_peft_model_state_dict(model, loaded)
        restored = adapter_snapshot(model)
        assert all(torch.equal(restored[key], after_state[key]) for key in restored)
        with torch.no_grad():
            reloaded = [token_logprobs(model, prompt, a["token_ids"]).detach() for a in answers]
        reload_error = max(float((p - q).abs().max()) for p, q in zip(reloaded, after))
        assert reload_error <= 1e-5
        base_after = base_fingerprint(model)
        assert base_before == base_after, "A frozen base weight changed."
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        result.update(base_weights_unchanged=True, base_sha256_before=base_before,
                      base_sha256_after=base_after, adapter_reload_max_logp_difference=reload_error)
        print("PASS: One optimizer step, changed adapter, unchanged base, and checkpoint reload.", flush=True)
    else:
        print("PASS: Adapter inspection only; optimizer steps: 0.", flush=True)
    torch.cuda.synchronize()
    result["load_update_and_verification_wall_seconds"] = time.perf_counter() - started
    result["peak_torch_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    result["note"] = "Engineering check; includes extra audits, excludes earlier rollout generation; not evidence of accuracy improvement."
    save_json(run / "summary.json", result)
    print(json.dumps(result, indent=2))
    print("Saved:", run)


if __name__ == "__main__":
    main()
