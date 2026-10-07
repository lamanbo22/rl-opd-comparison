"""One OPD or joint RL+OPD adapter update on the validated initial-policy group.

This is an engineering check, not a benchmark or a repeated training loop.
OPD uses a detached token-local teacher-minus-old-student log-probability
advantage in a clipped policy surrogate. It is not an exact full-sequence KL.
Place beside one_rl_update.py in the project's experiments directory.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path

from one_rl_update import (
    adapter_snapshot, base_fingerprint, digest, save_json, surrogate_loss,
    token_logprobs, validate_inputs,
)


def state_fingerprint(state):
    """Hash sorted adapter names, tensor properties, and every tensor byte."""
    import torch
    checksum = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        checksum.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        checksum.update(memoryview(value.view(torch.uint8).numpy()))
    return checksum.hexdigest()


def learning_signals(answers, old, teacher, method, coefficient):
    """Keep target signals detached while differentiating current log-probs."""
    import torch
    assert method in ("opd", "joint")
    assert math.isfinite(coefficient) and coefficient > 0
    assert len(answers) == len(old) == len(teacher)
    signals = []
    for answer, student_scores, teacher_scores in zip(answers, old, teacher):
        assert student_scores.shape == teacher_scores.shape == (len(answer["token_ids"]),)
        value = coefficient * (teacher_scores.detach() - student_scores.detach())
        if method == "joint":
            value = value + answer["advantage"]
        assert not value.requires_grad and torch.isfinite(value).all()
        signals.append(value)
    return signals


def apply_method_update(model, prompt, answers, old, signals, learning_rate):
    """Accumulate one response at a time, then take precisely one AdamW step."""
    import torch
    assert len(answers) == len(old) == len(signals)
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    assert named and all("lora_A." in name or "lora_B." in name for name, _ in named)
    parameters = [parameter for _, parameter in named]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {id(p) for p in parameters}
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    maximum_difference = 0.0
    for index, (answer, previous, signal) in enumerate(zip(answers, old, signals), start=1):
        current = token_logprobs(model, prompt, answer["token_ids"])
        difference = float((current.detach() - previous).abs().max())
        maximum_difference = max(maximum_difference, difference)
        assert difference <= 1e-3, "Pre-update policy mismatch; no optimizer step taken."
        assert not previous.requires_grad and not signal.requires_grad
        loss = surrogate_loss(current, previous, signal, len(answers))
        assert torch.isfinite(loss)
        total_loss += float(loss.detach())
        loss.backward()
        print(f"Backward {index}/{len(answers)} complete; optimizer steps so far: 0", flush=True)
        del current, loss
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    nonzero = {"A": 0, "B": 0}
    for name, parameter in named:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        if torch.count_nonzero(parameter.grad).item():
            nonzero["A" if "lora_A." in name else "B"] += 1
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    assert float(norm) > 0, "No learning signal reached the adapter."
    optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in parameters)
    optimizer.zero_grad(set_to_none=True)
    return {
        "optimizer_steps": 1, "loss_before": total_loss,
        "gradient_norm_before_clipping": float(norm), "gradient_clip_max_norm": 1.0,
        "nonzero_gradient_tensors": nonzero,
        "max_preupdate_logp_difference": maximum_difference,
    }


def check_teacher_tokenizer(config, group):
    from transformers import AutoTokenizer
    tokenizers = [AutoTokenizer.from_pretrained(
        config["models"][role]["local_snapshot"], local_files_only=True, trust_remote_code=False,
    ) for role in ("student", "teacher")]
    student, teacher = tokenizers
    assert student.get_vocab() == teacher.get_vocab()
    assert student.backend_tokenizer.to_str() == teacher.backend_tokenizer.to_str()
    assert student.special_tokens_map == teacher.special_tokens_map
    assert student.chat_template == teacher.chat_template
    text = teacher.apply_chat_template(
        [{"role": "user", "content": config["prompt_instruction"] + group["question"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert teacher(text, add_special_tokens=False)["input_ids"] == group["prompt_token_ids"]
    print("PASS: Teacher and student token IDs, tokenizer backends, and prompt match.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("opd", "joint"), required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--distillation-coefficient", type=float, default=1.0)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    assert math.isfinite(args.distillation_coefficient) and args.distillation_coefficient > 0
    experiment = args.project_root.resolve() / "experiments"
    config, group, source_run = validate_inputs(experiment)
    compatibility_started = time.perf_counter()
    check_teacher_tokenizer(config, group)
    compatibility_seconds = time.perf_counter() - compatibility_started
    if args.check_only:
        print("Model weights loaded: False. Optimizer steps: 0.")
        return

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    from transformers import AutoModelForCausalLM

    assert torch.cuda.is_available()
    name = args.name or f"{args.method}_one_update_01"
    assert Path(name).name == name and name not in (".", "..")
    run = experiment / "runs" / name
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()

    def load(role):
        model = AutoModelForCausalLM.from_pretrained(
            config["models"][role]["local_snapshot"], local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16,
            device_map={"": "cuda:0"}, attn_implementation="sdpa",
        ).eval()
        model.config.use_cache = False
        model.requires_grad_(False)
        return model

    load_started = time.perf_counter()
    base = load("student")
    torch.manual_seed(config["seed"])
    adapter_config = LoraConfig(
        **config["lora"], task_type=TaskType.CAUSAL_LM, bias="none",
        init_lora_weights=True, use_rslora=False,
    )
    model = get_peft_model(base, adapter_config, autocast_adapter_dtype=True,
                           revision=config["models"]["student"]["revision"])
    model.eval()  # No stochastic layers; gradient calculation remains enabled.
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_A." in name or "lora_B." in name for name, _ in trainable)
    assert all(p.dtype == torch.float32 for _, p in trainable)
    b_parameters = [p for name, p in trainable if "lora_B." in name]
    assert b_parameters and all(torch.count_nonzero(p).item() == 0 for p in b_parameters)
    model.print_trainable_parameters()
    torch.cuda.synchronize()
    student_load_seconds = time.perf_counter() - load_started

    teacher_started = time.perf_counter()
    teacher = load("teacher")
    assert not any(p.requires_grad for p in teacher.parameters())
    assert model.config.vocab_size == teacher.config.vocab_size
    torch.cuda.synchronize()
    teacher_load_seconds = time.perf_counter() - teacher_started

    prompt, answers = group["prompt_token_ids"], group["answers"]
    scoring_started = time.perf_counter()
    with torch.no_grad():
        with model.disable_adapter():
            bare = [token_logprobs(model, prompt, answer["token_ids"]).detach() for answer in answers]
        old = [token_logprobs(model, prompt, answer["token_ids"]).detach() for answer in answers]
    initial_difference = max(float((a - b).abs().max()) for a, b in zip(bare, old))
    assert initial_difference <= 1e-5
    torch.cuda.synchronize()
    initial_scoring_seconds = time.perf_counter() - scoring_started

    teacher_scoring_started = time.perf_counter()
    with torch.no_grad():
        teacher_scores = [token_logprobs(teacher, prompt, answer["token_ids"]).detach() for answer in answers]
    assert all(not score.requires_grad and torch.isfinite(score).all() for score in teacher_scores)
    assert all(p.grad is None for p in teacher.parameters())
    signals = learning_signals(answers, old, teacher_scores, args.method, args.distillation_coefficient)
    torch.cuda.synchronize()
    teacher_scoring_seconds = time.perf_counter() - teacher_scoring_started
    print("PASS: No-op initial adapter and detached finite teacher scores.", flush=True)

    audit_started = time.perf_counter()
    initial_state = adapter_snapshot(model)
    initial_adapter_sha = state_fingerprint(initial_state)
    model.save_pretrained(run / "adapter_initial", safe_serialization=True, save_embedding_layers=False)
    print("Hashing all frozen student and teacher parameters...", flush=True)
    base_before, teacher_before = base_fingerprint(model), base_fingerprint(teacher)
    preupdate_audit_seconds = time.perf_counter() - audit_started
    metadata = {
        "config": config, "method": args.method,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "script_sha256": digest(Path(__file__)),
        "rl_helper_sha256": digest(experiment / "one_rl_update.py"),
        "config_sha256": digest(experiment / "configs/pilot.json"),
        "source_rollouts_sha256": digest(source_run / "rollouts.jsonl"),
        "source_metadata_sha256": digest(source_run / "metadata.json"),
        "source_run": str(source_run), "source_row": group["source_row"],
        "selection": "first corrected initial-policy group, engineering check only",
        "seed": config["seed"], "initial_adapter_sha256": initial_adapter_sha,
        "objective": "Clipped policy surrogate; per-response token mean then group mean",
        "distillation_signal": "stop_gradient(teacher_logp - old_student_logp) per sampled token",
        "joint_signal": "sequence_reward_advantage + coefficient * distillation_signal",
        "distillation_coefficient": args.distillation_coefficient,
        "opd_scope": "Token-local surrogate, not exact full-sequence reverse KL; no future-token return",
        "sampling": "Corrected raw initial student distribution; same frozen group as RL validation",
        "clip_epsilon": 0.2, "reference_kl_beta": 0.0,
        "eos_in_loss": True, "prompt_in_loss": False, "microbatch_responses": 1,
        "advantage_std": "population", "optimizer": "AdamW", "weight_decay": 0.0,
        "learning_rate": config["learning_rate"],
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "total_student_parameters": sum(p.numel() for p in model.parameters()),
        "teacher_loaded": True, "teacher_trainable_parameters": 0,
        "checkpoint_scope": "adapter only; optimizer state not saved",
    }
    save_json(run / "metadata.json", metadata)

    torch.cuda.synchronize()
    update_started = time.perf_counter()
    result = apply_method_update(model, prompt, answers, old, signals, config["learning_rate"])
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_started

    verification_started = time.perf_counter()
    after_state = adapter_snapshot(model)
    changed = [name for name in initial_state if not torch.equal(initial_state[name], after_state[name])]
    assert changed
    with torch.no_grad():
        after = [token_logprobs(model, prompt, answer["token_ids"]).detach() for answer in answers]
    after_loss = sum(float(surrogate_loss(current, previous, signal, len(answers)))
                     for current, previous, signal in zip(after, old, signals))
    assert math.isfinite(after_loss)
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
        reloaded = [token_logprobs(model, prompt, answer["token_ids"]).detach() for answer in answers]
    reload_error = max(float((a - b).abs().max()) for a, b in zip(reloaded, after))
    assert reload_error <= 1e-5
    base_after, teacher_after = base_fingerprint(model), base_fingerprint(teacher)
    assert base_before == base_after and teacher_before == teacher_after
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    torch.cuda.synchronize()
    verification_seconds = time.perf_counter() - verification_started

    scores = []
    for number, (answer, previous, target, signal, current) in enumerate(
        zip(answers, old, teacher_scores, signals, after), start=1
    ):
        scores.append({
            "response": number, "token_ids": answer["token_ids"],
            "reward": answer["reward"], "sequence_advantage": answer["advantage"],
            "old_student_logprobs": previous.cpu().tolist(),
            "teacher_logprobs": target.cpu().tolist(),
            "effective_token_advantages": signal.cpu().tolist(),
            "after_student_logprobs": current.cpu().tolist(),
        })
    save_json(run / "token_scores.json", {"source_row": group["source_row"], "answers": scores})
    result.update({
        "status": "complete", "method": args.method,
        "initial_adapter_sha256": initial_adapter_sha,
        "after_adapter_sha256": state_fingerprint(after_state),
        "initial_adapter_max_logp_difference": initial_difference,
        "changed_adapter_tensors": len(changed),
        "adapter_max_absolute_change": max(float((after_state[k] - initial_state[k]).abs().max()) for k in changed),
        "loss_after_on_same_responses": after_loss,
        "mean_logp_changes_by_response": [float((current - previous).mean()) for current, previous in zip(after, old)],
        "mean_teacher_minus_student_logp_by_response": [float((target - previous).mean()) for target, previous in zip(teacher_scores, old)],
        "scored_completion_tokens": sum(len(answer["token_ids"]) for answer in answers),
        "trainable_parameters": metadata["trainable_parameters"],
        "base_weights_unchanged": True, "teacher_weights_unchanged": True,
        "teacher_gradients_absent": True,
        "base_sha256_before": base_before, "base_sha256_after": base_after,
        "teacher_sha256_before": teacher_before, "teacher_sha256_after": teacher_after,
        "adapter_reload_max_logp_difference": reload_error,
        "tokenizer_compatibility_seconds": compatibility_seconds,
        "student_load_seconds": student_load_seconds,
        "teacher_load_seconds": teacher_load_seconds,
        "initial_scoring_seconds": initial_scoring_seconds,
        "teacher_scoring_seconds": teacher_scoring_seconds,
        "preupdate_audit_seconds": preupdate_audit_seconds,
        "update_wall_seconds": update_seconds,
        "verification_wall_seconds": verification_seconds,
        "load_update_and_verification_wall_seconds": time.perf_counter() - started,
        "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "note": "One-group engineering check; includes extra audits, excludes earlier generation; not evidence of accuracy improvement or equal-budget ranking.",
    })
    save_json(run / "summary.json", result)
    print("PASS: One adapter update, frozen student base and teacher, checkpoint reload.", flush=True)
    print(json.dumps(result, indent=2))
    print("Saved:", run)


if __name__ == "__main__":
    main()
