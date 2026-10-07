"""Repeated RL / sampled-token OPD / joint pilot with resumable boundaries.

Place beside one_rl_update.py, part2_methods.py and collect_rollouts_fixed.py.
This fixed-small-pilot script is not an equal-compute performance comparison.
OPD is a token-local policy surrogate, not an exact sequence-level KL gradient.
"""
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from collect_rollouts_fixed import (
    advantages, check_sampling_without_weights, make_generation,
    resolve_and_verify, sampling_options,
)
from one_rl_update import (
    adapter_snapshot, base_fingerprint, digest, read_jsonl, surrogate_loss,
    token_logprobs,
)
from part2_methods import state_fingerprint


def atomic_json(path, value):
    """Replace one JSON document atomically on its own filesystem."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        assert torch.cuda.is_available()
        assert len(state["torch_cuda"]) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def remove_owned_checkpoint(path, root):
    """Retention applies only to completed directories inside our checkpoint root."""
    path, root = Path(path).resolve(), Path(root).resolve()
    assert path.parent == root and path.name.startswith("boundary_")
    assert (path / "complete.json").is_file()
    shutil.rmtree(path)


def verified_boundaries(run, identity):
    """Incomplete writes are ignored; completed but corrupt writes fail loudly."""
    run = Path(run)
    found = []
    for marker in (run / "checkpoints").glob("boundary_*/complete.json"):
        document = json.loads(marker.read_bytes())
        assert document["identity"] == identity, "Checkpoint identity mismatch."
        payload_path = marker.parent / "state.pt"
        assert digest(payload_path) == document["state_sha256"], "Checkpoint bytes changed."
        for relative, expected in document["evidence_sha256"].items():
            path = (run / relative).resolve()
            assert path.is_relative_to(run.resolve())
            assert digest(path) == expected, f"Checkpoint evidence changed: {relative}"
        found.append((document["boundary"], marker.parent, document))
    found.sort(key=lambda entry: (entry[0], entry[1].name))
    assert len({entry[0] for entry in found}) == len(found), "Duplicate completed boundary."
    return found


def save_boundary(run, payload, identity, evidence_files):
    """Commit state + hashes atomically; retain the latest two complete boundaries.

    Latest-pointer recovery scans verified completion markers, so interruption
    between the directory commit and pointer update cannot replay an update.
    Only load these locally written, checksum-verified torch files.
    """
    import torch
    run = Path(run)
    checkpoint_root = run / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    boundary = payload["next_iteration"]
    unique = uuid.uuid4().hex
    pending = checkpoint_root / (".pending_" + unique)
    complete = checkpoint_root / f"boundary_{boundary:04d}_{unique}"
    pending.mkdir()
    started = time.perf_counter()
    evidence = {str(Path(path).resolve().relative_to(run.resolve())): digest(Path(path))
                for path in evidence_files}
    cpu_payload = cpu_tree(payload)
    cpu_payload["identity"] = identity
    with (pending / "state.pt").open("wb") as stream:
        torch.save(cpu_payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checksum = digest(pending / "state.pt")
    elapsed = time.perf_counter() - started
    marker = {"boundary": boundary, "identity": identity, "state_sha256": checksum,
              "evidence_sha256": evidence, "checkpoint_payload_write_seconds": elapsed,
              "timing_scope": "CPU copies, payload serialization, fsync and checksums; excludes completion-marker commit and retention pruning"}
    atomic_json(pending / "complete.json", marker)
    os.replace(pending, complete)
    atomic_json(checkpoint_root / "latest.json", {"directory": complete.name, **marker})
    completed = verified_boundaries(run, identity)
    for _, directory, _ in completed[:-2]:
        remove_owned_checkpoint(directory, checkpoint_root)
    return marker


def load_boundary(run, identity):
    import torch
    completed = verified_boundaries(run, identity)
    assert completed, "No complete, verified checkpoint boundary exists."
    boundary, directory, marker = completed[-1]
    # This is our own complete checksum-verified file, never an external checkpoint.
    payload = torch.load(directory / "state.pt", map_location="cpu", weights_only=False)
    assert payload["identity"] == identity and payload["next_iteration"] == boundary
    assert payload["next_question_offset"] == boundary * payload["prompts_per_iteration"]
    for metric in payload["history"]:
        for relative, expected in metric.get("evidence_sha256", {}).items():
            path = (Path(run) / relative).resolve()
            assert path.is_relative_to(Path(run).resolve())
            assert digest(path) == expected, f"Historical iteration evidence changed: {relative}"
    if payload["history"]:
        payload["history"][-1]["checkpoint_payload_write_seconds"] = marker["checkpoint_payload_write_seconds"]
    return payload, marker


def build_signals(groups, old_scores, teacher_scores, method, coefficient=1.0):
    import torch
    assert method in ("rl", "opd", "joint")
    assert math.isfinite(coefficient) and coefficient > 0
    assert len(groups) == len(old_scores)
    if method != "rl":
        assert teacher_scores is not None and len(groups) == len(teacher_scores)
    signals = []
    for group_index, (group, previous_group) in enumerate(zip(groups, old_scores)):
        assert len(group["answers"]) == len(previous_group)
        group_signals = []
        for answer_index, (answer, previous) in enumerate(zip(group["answers"], previous_group)):
            assert previous.shape == (len(answer["token_ids"]),)
            assert not previous.requires_grad and torch.isfinite(previous).all()
            if method == "rl":
                signal = torch.full_like(previous, answer["advantage"])
            else:
                target = teacher_scores[group_index][answer_index]
                assert target.shape == previous.shape and not target.requires_grad
                signal = coefficient * (target.detach() - previous.detach())
                if method == "joint":
                    signal = signal + answer["advantage"]
            assert not signal.requires_grad and torch.isfinite(signal).all()
            group_signals.append(signal)
        signals.append(group_signals)
    return signals


def apply_training_update(model, optimizer, groups, old_scores, signals):
    """One accumulated step; means over tokens, responses, then prompt groups."""
    import torch
    assert groups and len(groups) == len(old_scores) == len(signals)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert parameters
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in parameters}
    optimizer.zero_grad(set_to_none=True)
    any_signal = any(torch.count_nonzero(signal).item() for group in signals for signal in group)
    if not any_signal:
        # Adam momentum would otherwise move weights even with zero gradients.
        return {"optimizer_steps": 0, "skip_reason": "all_effective_signals_zero",
                "loss_before": 0.0, "gradient_norm_before_clipping": 0.0,
                "max_preupdate_logp_difference": 0.0, "backward_responses": 0}
    total_loss, maximum_difference, backwards = 0.0, 0.0, 0
    for group, previous_group, signal_group in zip(groups, old_scores, signals):
        assert len(group["answers"]) == len(previous_group) == len(signal_group)
        for answer, previous, signal in zip(group["answers"], previous_group, signal_group):
            current = token_logprobs(model, group["prompt_token_ids"], answer["token_ids"])
            difference = float((current.detach() - previous).abs().max())
            maximum_difference = max(maximum_difference, difference)
            assert difference <= 1e-3, "Pre-update policy mismatch; stop before optimizer step."
            loss = surrogate_loss(current, previous, signal, len(group["answers"])) / len(groups)
            assert torch.isfinite(loss)
            total_loss += float(loss.detach())
            loss.backward()
            backwards += 1
            del current, loss
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    assert float(norm) > 0, "Nonzero sampled signals produced no adapter gradient."
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in parameters)
    optimizer.zero_grad(set_to_none=True)
    return {"optimizer_steps": 1, "skip_reason": None, "loss_before": total_loss,
            "gradient_norm_before_clipping": float(norm),
            "max_preupdate_logp_difference": maximum_difference, "backward_responses": backwards}


def validate_pilot_inputs(experiment, iterations, prompts_per_iteration):
    config_path = experiment / "configs/pilot.json"
    config = json.loads(config_path.read_bytes())
    protocol_path = experiment / "configs/repeated_pilot_v1.json"
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol["source_config_sha256"] == digest(config_path)
    assert protocol["seed"] == config["seed"]
    assert protocol["iterations"] == iterations and protocol["prompts_per_iteration"] == prompts_per_iteration
    assert protocol["answers_per_prompt"] == config["rollout"]["answers_per_question"]
    assert protocol["joint_coefficient"] == 1.0 and protocol["formal_equal_budget_study"] is False
    assert config["seed"] == 42 and config["learning_rate"] == 1e-5
    assert config["lora"]["lora_dropout"] == 0.0
    assert config["rollout"]["answers_per_question"] == 4
    assert config["rollout"]["enable_thinking"] is False
    assert sampling_options(config["rollout"]) == {
        "do_sample": True, "temperature": 1.0, "top_p": 1.0,
        "top_k": 0, "min_p": 0.0, "max_new_tokens": 1024,
    }
    assert 1 <= iterations and 1 <= prompts_per_iteration
    assert iterations * prompts_per_iteration <= 32, "This script is a bounded 32-question pilot."
    datasets = {}
    for split, expected_length in (("train", 6961), ("dev", 512)):
        path = experiment / config["data"][f"{split}_file"]
        assert digest(path) == config["data"][f"{split}_sha256"]
        datasets[split] = read_jsonl(path)
        assert len(datasets[split]) == expected_length
        assert len({row["question_sha256"] for row in datasets[split]}) == expected_length
    assert not ({row["question_sha256"] for row in datasets["train"]}
                & {row["question_sha256"] for row in datasets["dev"]})
    for filename, expected in config["reward"]["code_sha256"].items():
        assert digest(experiment / "reward" / filename) == expected
    check_sampling_without_weights(config["models"]["student"], config["rollout"])
    return config, datasets["train"][:iterations * prompts_per_iteration]


def write_jsonl(path, values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def generate_groups(model, tokenizer, generation, sampling, settings, rows,
                    instruction, seed, offset, iteration, policy_step, policy_sha, score_answer):
    import torch
    groups = []
    eos = generation.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    with torch.inference_mode():
        for within, row in enumerate(rows):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction + row["question"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = tokenizer([prompt] * settings["answers_per_question"], add_special_tokens=False,
                               padding=True, return_tensors="pt").to(model.device)
            prompt_ids = inputs["input_ids"][0].tolist()
            batch_seed = seed + offset + within
            torch.manual_seed(batch_seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.generate(**inputs, generation_config=generation,
                                    use_model_defaults=False, **sampling)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            answers = []
            for sequence in output[:, len(prompt_ids):].cpu().tolist():
                end = next((index for index, token in enumerate(sequence) if token in eos_ids), None)
                tokens = sequence if end is None else sequence[:end + 1]
                assert tokens
                completion = tokenizer.decode(tokens, skip_special_tokens=True)
                grade = score_answer(completion, row["gold"])
                answers.append({"completion": completion, "token_ids": tokens, **grade,
                                "reward": int(grade["correct"]), "eos_terminated": end is not None,
                                "truncated": end is None and len(sequence) >= settings["max_new_tokens"]})
            for answer, advantage in zip(answers, advantages([answer["reward"] for answer in answers])):
                answer["advantage"] = advantage
            group = {**row, "prompt_token_ids": prompt_ids, "batch_seed": batch_seed,
                     "question_offset": offset + within, "iteration": iteration,
                     "policy_step": policy_step, "policy_adapter_sha256": policy_sha,
                     "generation_seconds": elapsed, "answers": answers}
            groups.append(group)
            print(f"Iteration {iteration + 1}, question {offset + within + 1}: "
                  f"rewards={[answer['reward'] for answer in answers]}", flush=True)
            del output, inputs
    return groups


def score_groups(model, groups):
    import torch
    with torch.no_grad():
        return [[token_logprobs(model, group["prompt_token_ids"], answer["token_ids"]).detach()
                 for answer in group["answers"]] for group in groups]


def verify_saved_adapter(model, path, expected):
    import torch
    from peft.utils.save_and_load import load_peft_weights
    loaded = load_peft_weights(str(path), device="cpu", local_files_only=True)
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[key], expected[key]) for key in expected)
    return state_fingerprint(loaded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("rl", "opd", "joint"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--prompts-per-iteration", type=int, default=4)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-iterations", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    assert Path(args.name).name == args.name and args.name not in (".", "..")
    stop_at = args.stop_after_iterations if args.stop_after_iterations is not None else args.iterations
    assert 1 <= stop_at <= args.iterations
    experiment = args.project_root.resolve() / "experiments"
    config, rows = validate_pilot_inputs(experiment, args.iterations, args.prompts_per_iteration)
    if args.check_only:
        print("PASS: Pilot config, source separation, reward hashes and sampler. No weights loaded.")
        return

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils.save_and_load import set_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    run = experiment / "runs" / args.name
    sources = {Path(__file__).name: digest(Path(__file__))}
    for filename in ("one_rl_update.py", "part2_methods.py", "collect_rollouts_fixed.py"):
        sources[filename] = digest(experiment / filename)
    spec = {"config": config, "config_sha256": digest(experiment / "configs/pilot.json"),
            "protocol_sha256": digest(experiment / "configs/repeated_pilot_v1.json"),
            "source_sha256": sources, "method": args.method, "iterations": args.iterations,
            "prompts_per_iteration": args.prompts_per_iteration, "group_size": 4,
            "source_rows": [row["source_row"] for row in rows],
            "question_sha256": [row["question_sha256"] for row in rows],
            "seed": config["seed"], "generation_seed_rule": "seed + zero-based question offset",
            "versions": {package: version(package) for package in ("torch", "transformers", "peft", "numpy")},
            "objective": "clipped ratio surrogate; mean over tokens, responses, then prompt groups",
            "opd_signal": "detached teacher_logp - old_student_logp per sampled token",
            "joint_signal": "sequence task advantage + 1.0 * opd_signal",
            "opd_scope": "token-local surrogate, not exact sequence-level KL gradient",
            "coefficient": 1.0, "clip_epsilon": 0.2, "gradient_clip_max_norm": 1.0,
            "optimizer": "persistent AdamW", "weight_decay": 0.0,
            "learning_rate": config["learning_rate"], "reference_kl_beta": 0.0,
            "prompt_in_loss": False, "eos_in_loss": True,
            "truncation_policy": "retain every generated token and frozen reward; flag truncation",
            "zero_signal_policy": "skip AdamW entirely if every effective signal is zero",
            "microbatch_responses": 1, "optimizer_epochs_per_batch": 1,
            "gpu": torch.cuda.get_device_name(), "dtype": "BF16 base and teacher; FP32 adapters and logits for cross entropy",
            "pilot_scope": "fixed small engineering pilot; not equal-compute method ranking"}
    identity = json_digest(spec)
    if args.resume:
        assert run.is_dir()
        metadata = json.loads((run / "metadata.json").read_bytes())
        assert metadata["identity"] == identity and metadata["spec"] == spec
        payload, marker = load_boundary(run, identity)
        start_iteration = payload["next_iteration"]
        assert start_iteration <= stop_at
        if start_iteration == args.iterations and (run / "summary.json").is_file():
            summary = json.loads((run / "summary.json").read_bytes())
            if summary.get("status") == "complete":
                print("Already complete:", run)
                print(json.dumps(summary, indent=2))
                return
    else:
        run.mkdir(parents=True, exist_ok=False)
        start_iteration = 0
        metadata = {"identity": identity, "spec": spec}
        atomic_json(run / "metadata.json", metadata)
        payload = None
    (run / "iterations").mkdir(exist_ok=True)
    sys.path.insert(0, str(experiment / "reward"))
    from text_answer_reward import score_answer

    session_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"]["local_snapshot"],
                                              local_files_only=True, trust_remote_code=False, padding_side="left")

    def load(role):
        value = AutoModelForCausalLM.from_pretrained(
            config["models"][role]["local_snapshot"], local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()
        value.config.use_cache = False
        value.requires_grad_(False)
        return value

    load_started = time.perf_counter()
    base = load("student")
    torch.manual_seed(config["seed"])
    model = get_peft_model(base, LoraConfig(
        **config["lora"], task_type=TaskType.CAUSAL_LM, bias="none", init_lora_weights=True, use_rslora=False,
    ), autocast_adapter_dtype=True, revision=config["models"]["student"]["revision"]).eval()
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert named and all("lora_A." in name or "lora_B." in name for name, _ in named)
    assert all(parameter.dtype == torch.float32 for _, parameter in named)
    assert all(torch.count_nonzero(parameter).item() == 0 for name, parameter in named if "lora_B." in name)
    torch.cuda.synchronize()
    student_load_seconds = time.perf_counter() - load_started
    teacher, teacher_load_seconds = None, 0.0
    if args.method != "rl":
        target_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"]["local_snapshot"],
                                                        local_files_only=True, trust_remote_code=False)
        assert tokenizer.get_vocab() == target_tokenizer.get_vocab()
        assert tokenizer.backend_tokenizer.to_str() == target_tokenizer.backend_tokenizer.to_str()
        assert tokenizer.special_tokens_map == target_tokenizer.special_tokens_map
        assert tokenizer.chat_template == target_tokenizer.chat_template
        load_started = time.perf_counter()
        teacher = load("teacher")
        assert model.config.vocab_size == teacher.config.vocab_size
        torch.cuda.synchronize()
        teacher_load_seconds = time.perf_counter() - load_started
    generation = make_generation(config["rollout"], model.generation_config)
    generation.pad_token_id = tokenizer.pad_token_id
    sampling = sampling_options(config["rollout"])
    generation, actual_sampling = resolve_and_verify(model, generation, sampling)
    optimizer = torch.optim.AdamW([parameter for _, parameter in named], lr=config["learning_rate"], weight_decay=0.0)

    audit_started = time.perf_counter()
    initial_state = adapter_snapshot(model)
    initial_sha = state_fingerprint(initial_state)
    base_before = base_fingerprint(model)
    teacher_before = base_fingerprint(teacher) if teacher is not None else None
    if args.resume:
        assert initial_sha == metadata["initial_adapter_sha256"]
        assert base_before == metadata["base_sha256_before"]
        assert teacher_before == metadata["teacher_sha256_before"]
        set_peft_model_state_dict(model, payload["adapter"])
        assert state_fingerprint(adapter_snapshot(model)) == payload["adapter_sha256"]
        optimizer.load_state_dict(payload["optimizer"])
        history = payload["history"]
        actual_updates = payload["actual_updates"]
        assert actual_updates == sum(metric["optimizer_steps"] for metric in history)
        assert len(history) == start_iteration
        write_jsonl(run / "metrics.jsonl", history)
        for metric in history:
            atomic_json(run / metric["evidence_directory"] / "metrics.json", metric)
        restore_rng(payload["rng"])
        print(f"PASS: Resumed boundary {start_iteration}; actual optimizer steps {actual_updates}.", flush=True)
    else:
        model.save_pretrained(run / "adapter_initial", safe_serialization=True, save_embedding_layers=False)
        assert verify_saved_adapter(model, run / "adapter_initial", initial_state) == initial_sha
        metadata.update(initial_adapter_sha256=initial_sha, base_sha256_before=base_before,
                        teacher_sha256_before=teacher_before, verified_sampling=actual_sampling,
                        trainable_parameters=sum(parameter.numel() for _, parameter in named),
                        student_load_seconds=student_load_seconds, teacher_load_seconds=teacher_load_seconds)
        atomic_json(run / "metadata.json", metadata)
        history, actual_updates = [], 0
    torch.cuda.synchronize()
    pretraining_audit_seconds = time.perf_counter() - audit_started
    sessions = read_jsonl(run / "sessions.jsonl") if (run / "sessions.jsonl").exists() else []
    sessions.append({"resume": args.resume, "start_iteration": start_iteration, "status": "started",
                     "student_load_seconds": student_load_seconds, "teacher_load_seconds": teacher_load_seconds,
                     "pretraining_audit_seconds": pretraining_audit_seconds})
    write_jsonl(run / "sessions.jsonl", sessions)

    def boundary_payload(next_iteration, state):
        return {"next_iteration": next_iteration, "next_question_offset": next_iteration * args.prompts_per_iteration,
                "prompts_per_iteration": args.prompts_per_iteration, "actual_updates": actual_updates,
                "adapter": state, "adapter_sha256": state_fingerprint(state),
                "optimizer": optimizer.state_dict(), "rng": capture_rng(), "history": history}

    if not args.resume:
        initial_marker = save_boundary(run, boundary_payload(0, initial_state), identity, [run / "metadata.json"])
        sessions[-1]["initial_checkpoint_payload_write_seconds"] = initial_marker["checkpoint_payload_write_seconds"]
        write_jsonl(run / "sessions.jsonl", sessions)
    del initial_state

    for iteration in range(start_iteration, stop_at):
        iteration_started = time.perf_counter()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        offset = iteration * args.prompts_per_iteration
        selected = rows[offset:offset + args.prompts_per_iteration]
        iteration_audit_started = time.perf_counter()
        before_state = adapter_snapshot(model)
        policy_sha = state_fingerprint(before_state)
        iteration_audit_seconds = time.perf_counter() - iteration_audit_started
        evidence_dir = run / "iterations" / f"iteration_{iteration + 1:04d}_{uuid.uuid4().hex}"
        evidence_dir.mkdir()
        torch.cuda.synchronize()
        phase_started = time.perf_counter()
        groups = generate_groups(model, tokenizer, generation, sampling, config["rollout"], selected,
                                 config["prompt_instruction"], config["seed"], offset, iteration,
                                 actual_updates, policy_sha, score_answer)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - phase_started
        generation_peak = torch.cuda.max_memory_allocated() / 2**30

        torch.cuda.reset_peak_memory_stats()
        phase_started = time.perf_counter()
        previous = score_groups(model, groups)
        torch.cuda.synchronize()
        old_scoring_seconds = time.perf_counter() - phase_started
        old_scoring_peak = torch.cuda.max_memory_allocated() / 2**30
        targets, teacher_scoring_seconds = None, 0.0
        teacher_scoring_peak = 0.0
        if teacher is not None:
            torch.cuda.reset_peak_memory_stats()
            phase_started = time.perf_counter()
            targets = score_groups(teacher, groups)
            torch.cuda.synchronize()
            teacher_scoring_seconds = time.perf_counter() - phase_started
            teacher_scoring_peak = torch.cuda.max_memory_allocated() / 2**30
            assert all(parameter.grad is None for parameter in teacher.parameters())
        signals = build_signals(groups, previous, targets, args.method)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        phase_started = time.perf_counter()
        result = apply_training_update(model, optimizer, groups, previous, signals)
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - phase_started
        update_peak = torch.cuda.max_memory_allocated() / 2**30
        actual_updates += result["optimizer_steps"]

        iteration_audit_started = time.perf_counter()
        after_state = adapter_snapshot(model)
        after_sha = state_fingerprint(after_state)
        changed = sum(not torch.equal(before_state[key], after_state[key]) for key in before_state)
        assert bool(changed) == bool(result["optimizer_steps"])
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        score_evidence = []
        for group_index, group in enumerate(groups):
            for response_index, answer in enumerate(group["answers"]):
                score_evidence.append({"source_row": group["source_row"], "response": response_index,
                    "iteration": iteration, "policy_step": group["policy_step"],
                    "policy_adapter_sha256": policy_sha, "token_ids": answer["token_ids"],
                    "sequence_advantage": answer["advantage"],
                    "old_student_logprobs": previous[group_index][response_index].cpu().tolist(),
                    "teacher_logprobs": targets[group_index][response_index].cpu().tolist() if targets is not None else None,
                    "effective_token_advantages": signals[group_index][response_index].cpu().tolist()})
        write_jsonl(evidence_dir / "rollouts.jsonl", groups)
        write_jsonl(evidence_dir / "token_scores.jsonl", score_evidence)
        evidence_hashes = {str(path.relative_to(run)): digest(path)
                           for path in (evidence_dir / "rollouts.jsonl", evidence_dir / "token_scores.jsonl")}
        iteration_audit_seconds += time.perf_counter() - iteration_audit_started
        all_answers = [answer for group in groups for answer in group["answers"]]
        metric = {"iteration": iteration + 1, "question_offset": offset, "questions": len(groups),
                  "responses": len(all_answers), "generated_tokens": sum(len(answer["token_ids"]) for answer in all_answers),
                  "correct": sum(answer["reward"] for answer in all_answers),
                  "invalid_answers": sum(not answer["answer_valid"] for answer in all_answers),
                  "truncated_answers": sum(answer["truncated"] for answer in all_answers),
                  "mixed_reward_groups": sum(0 < sum(answer["reward"] for answer in group["answers"]) < len(group["answers"]) for group in groups),
                  **result, "actual_updates_after": actual_updates,
                  "policy_adapter_sha256_before": policy_sha, "policy_adapter_sha256_after": after_sha,
                  "changed_adapter_tensors": changed, "generation_wall_seconds": generation_seconds,
                  "old_policy_scoring_seconds": old_scoring_seconds, "teacher_scoring_seconds": teacher_scoring_seconds,
                  "backward_and_optimizer_seconds": update_seconds,
                  "training_core_seconds": generation_seconds + old_scoring_seconds + teacher_scoring_seconds + update_seconds,
                  "iteration_audit_seconds": iteration_audit_seconds,
                  "generation_peak_allocated_gib": generation_peak, "old_scoring_peak_allocated_gib": old_scoring_peak,
                  "teacher_scoring_peak_allocated_gib": teacher_scoring_peak, "update_peak_allocated_gib": update_peak,
                  "peak_torch_allocated_gib": max(generation_peak, old_scoring_peak, teacher_scoring_peak, update_peak),
                  "evidence_directory": str(evidence_dir.relative_to(run)),
                  "evidence_sha256": evidence_hashes,
                  "iteration_precheckpoint_wall_seconds": time.perf_counter() - iteration_started}
        history.append(metric)
        marker = save_boundary(run, boundary_payload(iteration + 1, after_state), identity,
                               [run / "metadata.json", evidence_dir / "rollouts.jsonl", evidence_dir / "token_scores.jsonl"])
        metric["checkpoint_payload_write_seconds"] = marker["checkpoint_payload_write_seconds"]
        atomic_json(evidence_dir / "metrics.json", metric)
        write_jsonl(run / "metrics.jsonl", history)
        atomic_json(run / "progress.json", {"status": "training", "completed_iterations": iteration + 1,
                                            "actual_optimizer_steps": actual_updates, "identity": identity})
        print(json.dumps(metric, indent=2), flush=True)
        del previous, targets, signals, before_state, after_state, groups, score_evidence

    if stop_at < args.iterations:
        pause_audit_started = time.perf_counter()
        pause_base = base_fingerprint(model)
        pause_teacher = base_fingerprint(teacher) if teacher is not None else None
        assert pause_base == base_before and pause_teacher == teacher_before
        torch.cuda.synchronize()
        sessions[-1].update(status="paused_at_boundary", end_iteration=stop_at,
                            base_sha256_after=pause_base, teacher_sha256_after=pause_teacher,
                            pause_audit_seconds=time.perf_counter() - pause_audit_started,
                            session_wall_seconds=time.perf_counter() - session_started)
        write_jsonl(run / "sessions.jsonl", sessions)
        atomic_json(run / "progress.json", {"status": "paused_at_boundary", "completed_iterations": stop_at,
                                            "actual_optimizer_steps": actual_updates, "identity": identity})
        print(f"Stopped at complete boundary {stop_at}; use --resume with the same configuration.", flush=True)
        return

    final_audit_started = time.perf_counter()
    final_state = adapter_snapshot(model)
    final_sha = state_fingerprint(final_state)
    model.save_pretrained(run / "adapter_final", safe_serialization=True, save_embedding_layers=False)
    assert verify_saved_adapter(model, run / "adapter_final", final_state) == final_sha
    # Loading into the live adapter also verifies the PEFT restore route.
    with torch.no_grad():
        for _, parameter in named:
            parameter.zero_()
    set_peft_model_state_dict(model, final_state)
    assert state_fingerprint(adapter_snapshot(model)) == final_sha
    base_after = base_fingerprint(model)
    teacher_after = base_fingerprint(teacher) if teacher is not None else None
    assert base_before == base_after and teacher_before == teacher_after
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    if teacher is not None:
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    torch.cuda.synchronize()
    final_audit_seconds = time.perf_counter() - final_audit_started
    sessions[-1].update(status="complete", end_iteration=args.iterations,
                        base_sha256_after=base_after, teacher_sha256_after=teacher_after,
                        final_audit_seconds=final_audit_seconds,
                        session_wall_seconds=time.perf_counter() - session_started)
    write_jsonl(run / "sessions.jsonl", sessions)
    # Rebuild the committed metric log on finalization, including after boundary-only resume.
    write_jsonl(run / "metrics.jsonl", history)
    summary = {"status": "complete", "method": args.method, "identity": identity,
               "iterations": len(history), "actual_optimizer_steps": actual_updates,
               "skipped_iterations": sum(metric["optimizer_steps"] == 0 for metric in history),
               "initial_adapter_sha256": initial_sha, "final_adapter_sha256": final_sha,
               "base_sha256_before": base_before, "base_sha256_after": base_after,
               "teacher_sha256_before": teacher_before, "teacher_sha256_after": teacher_after,
               "base_weights_unchanged": True, "teacher_weights_unchanged": teacher is not None,
               "teacher_gradients_absent": teacher is not None, "adapter_save_reload_exact": True,
               "final_audit_seconds": final_audit_seconds,
               "current_session_wall_seconds": time.perf_counter() - session_started,
               "peak_iteration_torch_allocated_gib": max(metric["peak_torch_allocated_gib"] for metric in history),
               "checkpoint_timing_scope": "payload CPU copies, serialization, fsync and checksums; completion-marker commit and retention pruning excluded",
               "iteration_wall_timing_scope": "iteration_precheckpoint_wall_seconds stops before boundary payload construction; checkpoint payload costs are reported separately; sessions include complete wall time",
               "note": "Fresh-rollout repeated engineering pilot. Training samples are not held-out accuracy; equal update count is not equal compute. Evaluation is separate."}
    for key in ("questions", "responses", "generated_tokens", "correct", "mixed_reward_groups",
                "invalid_answers", "truncated_answers", "generation_wall_seconds", "old_policy_scoring_seconds",
                "teacher_scoring_seconds", "backward_and_optimizer_seconds", "training_core_seconds",
                "iteration_audit_seconds", "checkpoint_payload_write_seconds"):
        summary[key] = sum(metric[key] for metric in history)
    atomic_json(run / "summary.json", summary)
    atomic_json(run / "progress.json", {"status": "complete", "completed_iterations": args.iterations,
                                        "actual_optimizer_steps": actual_updates, "identity": identity})
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", run, flush=True)


if __name__ == "__main__":
    main()
