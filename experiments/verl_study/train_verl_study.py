"""Five-arm single-GPU study controller with guarded time budgets.

The bridge owns student backward, optimizer and native checkpoint operations.
This synchronous one-rank controller owns data, fresh HF rollouts, frozen
teacher scoring, objective signals, timing and committed evidence.
"""
import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import uuid
from importlib.metadata import version
from pathlib import Path

# Production location: <project>/experiments/verl_study/train_verl_study.py.
# Explicit --project-root also allows running a reviewed host-side source copy.
PROJECT_DEFAULT = Path(__file__).resolve().parents[2]
for index, argument in enumerate(sys.argv[:-1]):
    if argument == "--project-root":
        PROJECT_DEFAULT = Path(sys.argv[index + 1]).resolve()
        break
sys.path.insert(0, str(PROJECT_DEFAULT / "experiments"))
sys.path.insert(0, str(PROJECT_DEFAULT / "experiments/verl"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_rollouts_fixed import make_generation, resolve_and_verify, sampling_options
from one_rl_update import base_fingerprint, digest, read_jsonl
from part2_methods import state_fingerprint
from train_repeated_pilot import (
    atomic_json, build_signals, capture_rng, cpu_tree, generate_groups, json_digest,
    remove_owned_checkpoint, restore_rng, score_groups, validate_pilot_inputs,
    write_jsonl,
)
from study_schedule import BudgetSchedule, METHODS, TrainingCursor

_active_bridge = None
_stop_requested = False


def request_boundary_stop(signum, frame):
    global _stop_requested
    if _stop_requested:
        raise KeyboardInterrupt("Second stop signal: abandon uncommitted work.")
    _stop_requested = True
    print("Stop requested; finishing the current atomic batch and saving its boundary.", flush=True)


def stream_sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    assert path.is_relative_to(Path(root).resolve())
    return path


def rng_fingerprint(state):
    """Stable fingerprint of Python, NumPy, CPU and CUDA generator state."""
    checksum = hashlib.sha256()
    checksum.update(json.dumps(state["python"], separators=(",", ":")).encode())
    numpy = state["numpy"]
    checksum.update(str(numpy[0]).encode())
    checksum.update(numpy[1].tobytes())
    checksum.update(json.dumps(numpy[2:], separators=(",", ":")).encode())
    for tensor in [state["torch_cpu"], *state["torch_cuda"]]:
        checksum.update(memoryview(tensor.cpu().contiguous().numpy()))
    return checksum.hexdigest()


def validate_frozen_protocol(experiment, protocol, verl_diff_sha256):
    """Pin code and package versions across distinct formal runs, not just resume."""
    required_sources = {
        "verl_study/train_verl_study.py", "verl_study/study_schedule.py",
        "verl/verl_engine_bridge.py", "train_repeated_pilot.py", "one_rl_update.py",
        "part2_methods.py", "collect_rollouts_fixed.py",
    }
    expected_sources = protocol["frozen_source_sha256"]
    assert required_sources <= expected_sources.keys(), "Frozen protocol lacks required source fingerprints."
    for relative, expected in expected_sources.items():
        assert stream_sha(contained(experiment, relative)) == expected, f"Frozen source changed: {relative}"
    loaded_paths = {"verl_study/train_verl_study.py": Path(__file__)}
    for name, relative in (("study_schedule", "verl_study/study_schedule.py"),
                           ("train_repeated_pilot", "train_repeated_pilot.py"),
                           ("one_rl_update", "one_rl_update.py"),
                           ("part2_methods", "part2_methods.py"),
                           ("collect_rollouts_fixed", "collect_rollouts_fixed.py")):
        loaded_paths[relative] = Path(sys.modules[name].__file__)
    for relative, actual in loaded_paths.items():
        assert stream_sha(actual) == expected_sources[relative], f"Unfrozen loaded implementation: {actual}"
    import importlib
    loaded_bridge = importlib.import_module("verl_engine_bridge")
    assert stream_sha(Path(loaded_bridge.__file__)) == expected_sources["verl/verl_engine_bridge.py"], "Unfrozen loaded engine bridge."
    loaded_verl = Path(importlib.import_module("verl").__file__).resolve()
    assert loaded_verl.is_relative_to((experiment.parent / "external/verl/verl").resolve()), "verl imported from an alternate checkout."
    expected_versions = protocol["frozen_versions"]
    assert {"verl", "torch", "transformers", "peft", "numpy"} <= expected_versions.keys()
    for package, expected in expected_versions.items():
        assert version(package) == expected, f"Frozen package version changed: {package}"
    assert verl_diff_sha256 == protocol["frozen_verl_tracked_diff_sha256"], "Tracked verl modifications changed."


def scan_boundaries(run, identity):
    import torch
    boundaries = []
    for marker_path in (run / "checkpoints").glob("boundary_*/complete.json"):
        marker = json.loads(marker_path.read_bytes())
        assert marker["identity"] == identity
        actual_files = {str(path.relative_to(marker_path.parent)) for path in marker_path.parent.rglob("*")
                        if path.is_file() and path != marker_path}
        assert actual_files == set(marker["checkpoint_files_sha256"]), "Native checkpoint file set changed."
        for relative, expected in marker["checkpoint_files_sha256"].items():
            assert stream_sha(contained(marker_path.parent, relative)) == expected
        for relative, expected in marker["evidence_sha256"].items():
            assert stream_sha(contained(run, relative)) == expected
        # Budget markers become authoritative only after checking their exact
        # transition against the already checksummed controller payload.
        payload = torch.load(marker_path.parent / "controller.pt", map_location="cpu", weights_only=False)
        restored = BudgetSchedule(**payload["schedule_before"])
        if marker["budget_accounting"] is not None:
            transition = restored.commit(marker["budget_accounting"]["charged_iteration_seconds"],
                                         payload["history"][-1]["optimizer_steps"])
            assert transition == marker["budget_accounting"]
        assert restored.state_dict() == marker["schedule_after"]
        assert payload["next_iteration"] == marker["boundary"] == restored.completed_rounds
        assert payload["actual_updates"] == marker["actual_updates"] == restored.actual_updates
        del payload
        boundaries.append((marker["boundary"], marker_path.parent, marker))
    boundaries.sort(key=lambda entry: entry[0])
    assert len({entry[0] for entry in boundaries}) == len(boundaries)
    return boundaries


def checkpoint_selections(boundaries):
    """Derive selection from verified clocks, never from a mutable pointer."""
    assert boundaries
    latest = boundaries[-1]
    eligible = [entry for entry in boundaries if BudgetSchedule(**entry[2]["schedule_after"]).primary_eligible()]
    assert eligible
    primary = eligible[-1]
    state = BudgetSchedule(**latest[2]["schedule_after"])
    milestones = {}
    if state.mode == "budgeted":
        for fraction in (0.25, 0.5, 0.75):
            candidates = [entry for entry in boundaries
                          if fraction * state.budget_seconds <= entry[2]["schedule_after"]["charged_seconds"] <= state.budget_seconds]
            if candidates:
                milestones[f"{int(fraction * 100):03d}"] = candidates[0]
    return latest, primary, milestones


def has_unaccounted_work(run, history, sessions):
    """A failed/replayed attempt cannot silently disappear from formal costs."""
    if any(session["status"] == "started" or session.get("unaccounted_prior_failure_detected") for session in sessions):
        return True
    if any((run / "checkpoints").glob(".pending_*")):
        return True
    committed = {metric["evidence_directory"] for metric in history}
    return any(str(path.relative_to(run)) not in committed for path in (run / "iterations").iterdir())


def save_native_boundary(run, bridge, payload, identity, evidence_files, schedule,
                         iteration_started=None, optimizer_steps=0):
    """Commit engine+controller state and the authoritative measured budget.

    The clock includes audits, evidence writing, native saving and checksums.
    The completion marker contains clock-after because its value becomes known
    only after those operations. Marker/rename/pointer/prune/log tails are
    explicitly excluded and measured separately.
    """
    import torch
    checkpoint_root = run / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    unique = uuid.uuid4().hex
    pending = checkpoint_root / (".pending_" + unique)
    complete = checkpoint_root / f"boundary_{payload['next_iteration']:04d}_{unique}"
    pending.mkdir()
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Capture RNG before checkpoint I/O; native save may perform internal work.
    payload = cpu_tree({**payload, "identity": identity, "rng": capture_rng(),
                        "schedule_before": schedule.state_dict(),
                        "optimizer_scheduler_sha256": bridge.optimizer_fingerprint()})
    payload["rng_sha256"] = rng_fingerprint(payload["rng"])
    bridge.save_checkpoint(pending / "engine", payload["actual_updates"])
    torch.cuda.synchronize()
    with (pending / "controller.pt").open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    checkpoint_files = {str(path.relative_to(pending)): stream_sha(path)
                        for path in sorted(pending.rglob("*")) if path.is_file()}
    assert any(name.startswith("engine/") for name in checkpoint_files)
    assert bridge.optimizer_fingerprint() == payload["optimizer_scheduler_sha256"]
    evidence = {str(path.relative_to(run)): stream_sha(path) for path in evidence_files}
    previous_boundaries = scan_boundaries(run, identity)
    restore_rng(payload["rng"])
    assert rng_fingerprint(capture_rng()) == payload["rng_sha256"]
    torch.cuda.synchronize()
    charge_end = time.perf_counter()
    accounting = None
    if iteration_started is not None:
        accounting = schedule.commit(charge_end - iteration_started, optimizer_steps)
    assert schedule.completed_rounds == payload["next_iteration"]
    assert schedule.actual_updates == payload["actual_updates"]
    marker = {"identity": identity, "boundary": payload["next_iteration"],
              "actual_updates": payload["actual_updates"],
              "adapter_sha256": payload["adapter_sha256"],
              "checkpoint_files_sha256": checkpoint_files, "evidence_sha256": evidence,
              "checkpoint_payload_write_seconds": charge_end - started,
              "schedule_after": schedule.state_dict(), "budget_accounting": accounting,
              "timing_scope": "charged iteration covers adapter/cursor audits, generation/reward, scoring/signals/update, evidence, native checkpoint and checksums; marker/rename/pointer/prune/canonical-log tail excluded"}
    atomic_json(pending / "complete.json", marker)
    os.replace(pending, complete)
    atomic_json(checkpoint_root / "latest.json", {"directory": complete.name, **marker})
    retained = [*previous_boundaries, (marker["boundary"], complete, marker)]
    _, primary, milestones = checkpoint_selections(retained)
    atomic_json(checkpoint_root / "primary.json", {"directory": primary[1].name,
                "boundary": primary[0], "charged_seconds": primary[2]["schedule_after"]["charged_seconds"],
                "adapter_sha256": primary[2]["adapter_sha256"], "actual_updates": primary[2]["actual_updates"]})
    pinned = {primary[1].name, *(entry[1].name for entry in milestones.values())}
    for _, directory, _ in retained[:-2]:
        if directory.name in pinned:
            continue
        remove_owned_checkpoint(directory, checkpoint_root)
    return marker, charge_end


def read_native_boundary(run, identity):
    import torch
    completed = scan_boundaries(run, identity)
    assert completed, "No complete verified native checkpoint exists."
    boundary, directory, marker = completed[-1]
    _, primary, milestones = checkpoint_selections(completed)
    atomic_json(run / "checkpoints/primary.json", {"directory": primary[1].name, "boundary": primary[0],
                "charged_seconds": primary[2]["schedule_after"]["charged_seconds"],
                "adapter_sha256": primary[2]["adapter_sha256"], "actual_updates": primary[2]["actual_updates"]})
    payload = torch.load(directory / "controller.pt", map_location="cpu", weights_only=False)
    assert payload["identity"] == identity
    assert payload["next_iteration"] == boundary
    assert payload["next_question_offset"] == boundary * payload["prompts_per_iteration"]
    assert payload["actual_updates"] == marker["actual_updates"]
    assert payload["adapter_sha256"] == marker["adapter_sha256"]
    assert rng_fingerprint(payload["rng"]) == payload["rng_sha256"]
    assert len(payload["history"]) == boundary
    restored_schedule = BudgetSchedule(**payload["schedule_before"])
    if marker["budget_accounting"] is not None:
        expected = restored_schedule.commit(marker["budget_accounting"]["charged_iteration_seconds"],
                                            payload["history"][-1]["optimizer_steps"])
        assert expected == marker["budget_accounting"]
    assert restored_schedule.state_dict() == marker["schedule_after"]
    payload["schedule_after"] = marker["schedule_after"]
    for metric in payload["history"]:
        for relative, expected in metric["evidence_sha256"].items():
            assert stream_sha(contained(run, relative)) == expected
    if payload["history"]:
        payload["history"][-1]["checkpoint_payload_write_seconds"] = marker["checkpoint_payload_write_seconds"]
        payload["history"][-1].update(marker["budget_accounting"])
    return payload, directory, marker


def export_verified_adapter(bridge, destination, expected_state):
    import torch
    from peft.utils.save_and_load import load_peft_weights
    bridge.export_adapter(destination)
    loaded = load_peft_weights(str(destination), device="cpu", local_files_only=True)
    assert loaded.keys() == expected_state.keys()
    assert all(torch.equal(loaded[key], expected_state[key]) for key in loaded)
    return state_fingerprint(loaded)


class GenerationProxy:
    """Give the reference sampler the engine-backed generate implementation."""
    def __init__(self, bridge):
        self.bridge = bridge

    @property
    def device(self):
        import torch
        return torch.device("cuda:0")

    def generate(self, **kwargs):
        return self.bridge.generate(**kwargs)


def main():
    global _active_bridge
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("validation", "budgeted"), required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--validation-rounds", type=int, default=2)
    parser.add_argument("--validation-switch-after", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=10000)
    parser.add_argument("--backend", default="fsdp")
    parser.add_argument("--project-root", type=Path, default=PROJECT_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-rounds", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    assert Path(args.name).name == args.name and args.name not in (".", "..")
    assert args.max_rounds > 0 and (args.stop_after_rounds is None or args.stop_after_rounds > 0)
    project = args.project_root.resolve()
    experiment = project / "experiments"
    config, _ = validate_pilot_inputs(experiment, 8, 4)
    rows = read_jsonl(experiment / config["data"]["train_file"])
    assert len(rows) == 6961
    protocol_path = args.protocol.resolve() if args.protocol else experiment / "verl_study/protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    verl_protocol = json.loads((experiment / "verl/protocol.json").read_bytes())
    assert protocol["source_verl_protocol_sha256"] == digest(experiment / "verl/protocol.json")
    assert protocol["source_config_sha256"] == digest(experiment / "configs/pilot.json")
    assert protocol["prompts_per_iteration"] == protocol["answers_per_prompt"] == 4
    prompts_per_iteration = 4
    initial_adapter = Path(verl_protocol["initial_adapter"])
    assert stream_sha(initial_adapter / "adapter_model.safetensors") == verl_protocol["initial_adapter_file_sha256"]
    assert args.seed in protocol["seeds"]
    assert protocol["joint_coefficient"] == 1.0 and args.method in protocol["methods"]
    commit = subprocess.check_output(["git", "-C", str(project / "external/verl"), "rev-parse", "HEAD"], text=True).strip()
    assert commit == verl_protocol["verl_commit"]
    verl_diff = subprocess.check_output(["git", "-C", str(project / "external/verl"), "diff", "--binary", "HEAD"])
    verl_diff_sha = hashlib.sha256(verl_diff).hexdigest()
    schedule = BudgetSchedule(args.method, args.mode, protocol["budget_seconds"],
                              endpoint_policy=protocol["endpoint_policy"], reserve_seconds=protocol["reserve_seconds"],
                              overrun_tolerance_fraction=protocol.get("overrun_tolerance_fraction", 0.05),
                              validation_rounds=args.validation_rounds, validation_switch_after=args.validation_switch_after)
    if args.mode == "budgeted":
        assert protocol["endpoint_policy"] == "guarded_ceiling"
        assert protocol.get("status") == "frozen", "Budgeted execution needs a frozen protocol."
        validate_frozen_protocol(experiment, protocol, verl_diff_sha)
    if args.check_only:
        print("PASS: Data, reward, sampler, initial adapter, study protocol and verl commit; no model loaded.")
        print(json.dumps({"schedule": schedule.summary(), "cursor": TrainingCursor(rows, args.seed).state_dict()}, indent=2))
        return

    import numpy as np
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl_engine_bridge import VerlEngineBridge

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1, "This study permits exactly one visible GPU."
    assert "5090" in torch.cuda.get_device_name(0), "This study is pinned to the local RTX 5090."
    torch.set_num_threads(4)
    source_files = [Path(__file__), Path(__file__).with_name("study_schedule.py"), experiment / "verl/verl_engine_bridge.py"]
    source_files += [experiment / name for name in ("train_repeated_pilot.py", "one_rl_update.py", "part2_methods.py", "collect_rollouts_fixed.py")]
    versions = {name: version(name) for name in ("verl", "torch", "transformers", "peft", "numpy", "tensordict", "ray", "vllm")}
    spec = {"config": config, "config_sha256": digest(experiment / "configs/pilot.json"),
            "protocol": protocol, "protocol_sha256": digest(protocol_path),
            "reference_protocol_sha256": digest(experiment / "configs/repeated_pilot_v1.json"),
            "source_sha256": {str(path.relative_to(experiment)) if path.is_relative_to(experiment) else path.name: digest(path) for path in source_files},
            "verl_commit": commit, "verl_tracked_diff_sha256": verl_diff_sha,
            "versions": versions, "python_executable": sys.executable,
            "method": args.method, "mode": args.mode, "schedule_initial": schedule.state_dict(),
            "prompts_per_iteration": prompts_per_iteration,
            "group_size": 4, "seed": args.seed, "training_dataset_size": len(rows),
            "initial_cursor": TrainingCursor(rows, args.seed).state_dict(),
            "initial_adapter_seed": 42, "initial_adapter_file_sha256": verl_protocol["initial_adapter_file_sha256"],
            "backend": args.backend, "training_backend": "genuine verl one-rank engine with custom synchronous controller",
            "rollout_backend": "HF generate on current engine module through full-parameter context",
            "teacher_backend": "frozen HF model on same GPU",
            "coefficient": 1.0, "clip_epsilon": 0.2, "gradient_clip_max_norm": 1.0,
            "learning_rate": config["learning_rate"], "weight_decay": 0.0,
            "objective": "completion-token mean per response, then mean over four responses and four questions",
            "opd_signal": "detached teacher_logp minus old_student_logp for each sampled token",
            "joint_signal": "group task advantage plus OPD signal",
            "opd_scope": "token-local clipped surrogate, not exact full-sequence reverse KL gradient",
            "reference_kl_beta": 0.0, "eos_in_loss": True, "prompt_in_loss": False,
            "generation_seed_rule": "training seed plus global zero-based sampled-question index",
            "truncation_policy": "retain all completion tokens; include EOS when present; flag truncation",
            "zero_signal_policy": "skip engine optimizer step entirely if all effective signals are zero",
            "microbatch_responses": 1, "optimizer_epochs_per_batch": 1,
            "gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
            "scheduler_nominal_total_training_steps": 8,
            "scheduler_note": "Frozen bridge uses constant LR with zero warmup; nominal horizon does not stop or decay after 8 updates",
            "budget_scope": "continuous iteration including all training, reward, auditing, evidence and checkpoint/checksum work; final marker/rename/pointer/prune/log tail explicitly excluded",
            "scope": "Forced-switch engineering validation" if args.mode == "validation" else "Common upper training-time allowance on one fixed GPU"}
    identity = json_digest(spec)
    run = experiment / "runs" / args.name
    payload, checkpoint_dir = None, None
    if args.resume:
        metadata = json.loads((run / "metadata.json").read_bytes())
        assert metadata["identity"] == identity and metadata["spec"] == spec
        payload, checkpoint_dir, marker = read_native_boundary(run, identity)
        start_iteration = payload["next_iteration"]
        schedule = BudgetSchedule(**payload["schedule_after"])
        cursor = TrainingCursor.from_state(rows, payload["data_cursor"])
    else:
        run.mkdir(parents=True, exist_ok=False)
        (run / "iterations").mkdir()
        metadata = {"identity": identity, "spec": spec}
        atomic_json(run / "metadata.json", metadata)
        start_iteration = 0
        cursor = TrainingCursor(rows, args.seed)
    sys.path.insert(0, str(experiment / "reward"))
    from text_answer_reward import score_answer

    session_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    load_started = time.perf_counter()
    bridge = VerlEngineBridge({**config, "seed": args.seed}, initial_adapter, backend=args.backend)
    _active_bridge = bridge
    tokenizer = bridge.tokenizer
    tokenizer.padding_side = "left"
    torch.cuda.synchronize()
    student_load_seconds = time.perf_counter() - load_started
    teacher, teacher_load_seconds = None, 0.0
    if args.method != "rl":
        teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"]["local_snapshot"], local_files_only=True, trust_remote_code=False)
        assert tokenizer.get_vocab() == teacher_tokenizer.get_vocab()
        assert tokenizer.backend_tokenizer.to_str() == teacher_tokenizer.backend_tokenizer.to_str()
        assert tokenizer.special_tokens_map == teacher_tokenizer.special_tokens_map
        assert tokenizer.chat_template == teacher_tokenizer.chat_template
        load_started = time.perf_counter()
        teacher = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"]["local_snapshot"],
                  local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
                  device_map={"": "cuda:0"}, attn_implementation="sdpa").eval().requires_grad_(False)
        teacher.config.use_cache = False
        torch.cuda.synchronize()
        teacher_load_seconds = time.perf_counter() - load_started
    generation = make_generation(config["rollout"], bridge.model.generation_config)
    generation.pad_token_id = tokenizer.pad_token_id
    sampling = sampling_options(config["rollout"])
    generation, actual_sampling = resolve_and_verify(bridge.model, generation, sampling)

    audit_started = time.perf_counter()
    initial_state = bridge.adapter_snapshot()
    initial_sha = state_fingerprint(initial_state)
    expected_initial = load_file(str(initial_adapter / "adapter_model.safetensors"), device="cpu")
    assert initial_sha == state_fingerprint(expected_initial)
    del expected_initial
    base_before = bridge.base_fingerprint()
    teacher_before = base_fingerprint(teacher) if teacher is not None else None
    if args.resume:
        assert initial_sha == metadata["initial_adapter_sha256"]
        assert base_before == metadata["base_sha256_before"]
        assert teacher_before == metadata["teacher_sha256_before"]
        bridge.load_checkpoint(checkpoint_dir / "engine")
        assert state_fingerprint(bridge.adapter_snapshot()) == payload["adapter_sha256"]
        assert bridge.optimizer_fingerprint() == payload["optimizer_scheduler_sha256"]
        history, actual_updates = payload["history"], payload["actual_updates"]
        assert actual_updates == sum(metric["optimizer_steps"] for metric in history)
        write_jsonl(run / "metrics.jsonl", history)
        for metric in history:
            atomic_json(run / metric["evidence_directory"] / "metrics.json", metric)
        restore_rng(payload["rng"])
        assert rng_fingerprint(capture_rng()) == payload["rng_sha256"]
        print(f"PASS: Native verl checkpoint resumed at boundary {start_iteration}; updates={actual_updates}.", flush=True)
    else:
        assert export_verified_adapter(bridge, run / "adapter_initial", initial_state) == initial_sha
        metadata.update(initial_adapter_sha256=initial_sha, base_sha256_before=base_before,
                        teacher_sha256_before=teacher_before, verified_sampling=actual_sampling,
                        student_load_seconds=student_load_seconds, teacher_load_seconds=teacher_load_seconds,
                        trainable_parameters=sum(p.numel() for p in bridge.model.parameters() if p.requires_grad))
        metadata["framework"] = bridge.framework_metadata()
        atomic_json(run / "metadata.json", metadata)
        history, actual_updates = [], 0
    torch.cuda.synchronize()
    pretraining_audit_seconds = time.perf_counter() - audit_started
    sessions = read_jsonl(run / "sessions.jsonl") if (run / "sessions.jsonl").exists() else []
    unaccounted_failure = has_unaccounted_work(run, history, sessions)
    sessions.append({"resume": args.resume, "start_iteration": start_iteration, "status": "started",
                     "student_load_seconds": student_load_seconds, "teacher_load_seconds": teacher_load_seconds,
                     "pretraining_audit_seconds": pretraining_audit_seconds,
                     "framework_at_start": bridge.framework_metadata(),
                     "native_optimizer_resume_verified": bool(args.resume),
                     "controller_rng_resume_verified": bool(args.resume)})
    sessions[-1]["unaccounted_prior_failure_detected"] = bool(unaccounted_failure)
    write_jsonl(run / "sessions.jsonl", sessions)

    def boundary_payload(next_iteration, adapter_sha):
        return {"next_iteration": next_iteration, "next_question_offset": cursor.questions_seen,
                "prompts_per_iteration": prompts_per_iteration, "actual_updates": actual_updates,
                "data_cursor": cursor.state_dict(),
                "adapter_sha256": adapter_sha, "history": history,
                "checkpoint_owner": "verl engine", "controller_owner": "train_verl_study.py"}

    if not args.resume:
        marker, _ = save_native_boundary(run, bridge, boundary_payload(0, initial_sha), identity, [run / "metadata.json"], schedule)
        sessions[-1]["initial_checkpoint_payload_write_seconds"] = marker["checkpoint_payload_write_seconds"]
        write_jsonl(run / "sessions.jsonl", sessions)
    del initial_state
    generation_proxy = GenerationProxy(bridge)
    signal.signal(signal.SIGINT, request_boundary_stop)
    signal.signal(signal.SIGTERM, request_boundary_stop)
    commit_tails = read_jsonl(run / "commit_tails.jsonl") if (run / "commit_tails.jsonl").exists() else []
    stop_reason = "budget_or_validation_complete"

    while not schedule.should_stop():
        if _stop_requested or (args.stop_after_rounds is not None and schedule.completed_rounds >= args.stop_after_rounds):
            stop_reason = "requested_boundary_stop"
            break
        if schedule.completed_rounds >= args.max_rounds:
            stop_reason = "safety_max_rounds"
            break
        iteration = schedule.completed_rounds
        torch.cuda.synchronize()
        iteration_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        active_method = schedule.active_method()
        offset = cursor.questions_seen
        audit_started = time.perf_counter()
        cursor_before = cursor.state_dict()
        selected = cursor.take(prompts_per_iteration)
        before_state = bridge.adapter_snapshot()
        optimizer_before_sha = bridge.optimizer_fingerprint()
        policy_sha = state_fingerprint(before_state)
        audit_seconds = time.perf_counter() - audit_started
        evidence_dir = run / "iterations" / f"iteration_{iteration + 1:04d}_{uuid.uuid4().hex}"
        evidence_dir.mkdir()
        torch.cuda.synchronize()
        phase_started = time.perf_counter()
        groups = generate_groups(generation_proxy, tokenizer, generation, sampling, config["rollout"],
                 selected, config["prompt_instruction"], args.seed,
                 offset, iteration, actual_updates, policy_sha, score_answer)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter() - phase_started
        generation_peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        phase_started = time.perf_counter()
        previous = bridge.score_groups(groups)
        torch.cuda.synchronize()
        old_seconds = time.perf_counter() - phase_started
        old_peak = torch.cuda.max_memory_allocated() / 2**30
        targets, teacher_seconds, teacher_peak = None, 0.0, 0.0
        if active_method != "rl":
            assert teacher is not None
            torch.cuda.reset_peak_memory_stats()
            phase_started = time.perf_counter()
            targets = score_groups(teacher, groups)
            torch.cuda.synchronize()
            teacher_seconds = time.perf_counter() - phase_started
            teacher_peak = torch.cuda.max_memory_allocated() / 2**30
            assert all(parameter.grad is None for parameter in teacher.parameters())
        signals = build_signals(groups, previous, targets, active_method, coefficient=1.0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        phase_started = time.perf_counter()
        result = bridge.update(groups, previous, signals)
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - phase_started
        update_peak = torch.cuda.max_memory_allocated() / 2**30
        assert result["optimizer_steps"] in (0, 1)
        assert all(math.isfinite(result[key]) for key in ("loss_before", "gradient_norm_before_clipping", "max_preupdate_logp_difference"))
        assert 0 <= result["max_preupdate_logp_difference"] <= 1e-3
        actual_updates += result["optimizer_steps"]
        assert all(math.isclose(group["lr"], config["learning_rate"], rel_tol=0, abs_tol=1e-12)
                   for group in bridge.optimizer.param_groups), "Unexpected LR change."
        audit_started = time.perf_counter()
        after_state = bridge.adapter_snapshot()
        after_sha = state_fingerprint(after_state)
        changed = sum(not torch.equal(before_state[key], after_state[key]) for key in before_state)
        assert bool(changed) == bool(result["optimizer_steps"])
        score_evidence = []
        for gi, group in enumerate(groups):
            for ai, answer in enumerate(group["answers"]):
                score_evidence.append({"source_row": group["source_row"], "response": ai, "iteration": iteration,
                    "policy_step": group["policy_step"], "policy_adapter_sha256": policy_sha,
                    "token_ids": answer["token_ids"], "sequence_advantage": answer["advantage"],
                    "old_student_logprobs": previous[gi][ai].cpu().tolist(),
                    "teacher_logprobs": targets[gi][ai].cpu().tolist() if targets is not None else None,
                    "effective_token_advantages": signals[gi][ai].cpu().tolist()})
        write_jsonl(evidence_dir / "rollouts.jsonl", groups)
        write_jsonl(evidence_dir / "token_scores.jsonl", score_evidence)
        evidence_files = [evidence_dir / "rollouts.jsonl", evidence_dir / "token_scores.jsonl"]
        evidence_hashes = {str(path.relative_to(run)): stream_sha(path) for path in evidence_files}
        audit_seconds += time.perf_counter() - audit_started
        answers = [answer for group in groups for answer in group["answers"]]
        metric = {"iteration": iteration + 1, "question_offset": offset, "questions": len(groups),
                  "responses": len(answers), "generated_tokens": sum(len(a["token_ids"]) for a in answers),
                  "correct": sum(a["reward"] for a in answers), "invalid_answers": sum(not a["answer_valid"] for a in answers),
                  "truncated_answers": sum(a["truncated"] for a in answers),
                  "mixed_reward_groups": sum(0 < sum(a["reward"] for a in group["answers"]) < len(group["answers"]) for group in groups),
                  **result, "actual_updates_after": actual_updates, "changed_adapter_tensors": changed,
                  "active_method": active_method, "data_cursor_before": cursor_before,
                  "data_cursor_after": cursor.state_dict(),
                  "optimizer_scheduler_sha256_before_update": optimizer_before_sha,
                  "optimizer_scheduler_sha256_after_update": bridge.optimizer_fingerprint(),
                  "policy_adapter_sha256_before": policy_sha, "policy_adapter_sha256_after": after_sha,
                  "generation_wall_seconds": generation_seconds, "old_policy_scoring_seconds": old_seconds,
                  "teacher_scoring_seconds": teacher_seconds, "backward_and_optimizer_seconds": update_seconds,
                  "training_core_seconds": generation_seconds + old_seconds + teacher_seconds + update_seconds,
                  "iteration_audit_seconds": audit_seconds,
                  "generation_peak_allocated_gib": generation_peak, "old_scoring_peak_allocated_gib": old_peak,
                  "teacher_scoring_peak_allocated_gib": teacher_peak, "update_peak_allocated_gib": update_peak,
                  "peak_torch_allocated_gib": max(generation_peak, old_peak, teacher_peak, update_peak),
                  "evidence_directory": str(evidence_dir.relative_to(run)), "evidence_sha256": evidence_hashes,
                  "iteration_precheckpoint_wall_seconds": time.perf_counter() - iteration_started,
                  "framework_calls_this_process": bridge.framework_metadata()["call_counts_this_process"],
                  "student_update_owner": "verl engine"}
        history.append(metric)
        marker, charge_end = save_native_boundary(run, bridge, boundary_payload(iteration + 1, after_sha), identity,
                                      [run / "metadata.json", *evidence_files], schedule,
                                      iteration_started=iteration_started, optimizer_steps=result["optimizer_steps"])
        metric["checkpoint_payload_write_seconds"] = marker["checkpoint_payload_write_seconds"]
        metric.update(marker["budget_accounting"])
        assert metric["active_method"] == active_method
        atomic_json(evidence_dir / "metrics.json", metric)
        write_jsonl(run / "metrics.jsonl", history)
        atomic_json(run / "progress.json", {"status": "training", "completed_iterations": iteration + 1,
                    "actual_optimizer_steps": actual_updates, "identity": identity})
        print(json.dumps(metric, indent=2), flush=True)
        del previous, targets, signals, before_state, after_state, groups, score_evidence
        commit_tails.append({"iteration": iteration + 1, "uncharged_commit_tail_seconds": time.perf_counter() - charge_end,
                             "scope": "completion-marker/rename/pointer/prune/metrics/progress/console tail; excludes persisting this diagnostic record"})
        write_jsonl(run / "commit_tails.jsonl", commit_tails)

    final_started = time.perf_counter()
    final_state = bridge.adapter_snapshot()
    final_sha = state_fingerprint(final_state)
    base_after = bridge.base_fingerprint()
    teacher_after = base_fingerprint(teacher) if teacher is not None else None
    assert base_before == base_after and teacher_before == teacher_after
    if teacher is not None:
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    terminal_round = schedule.completed_rounds
    if not schedule.should_stop():
        status = "paused_at_boundary"
        final_seconds = time.perf_counter() - final_started
    else:
        final_payload, final_checkpoint, _ = read_native_boundary(run, identity)
        final_optimizer_sha = bridge.optimizer_fingerprint()
        assert final_optimizer_sha == final_payload["optimizer_scheduler_sha256"]
        verified = scan_boundaries(run, identity)
        _, primary_entry, milestone_entries = checkpoint_selections(verified)
        primary_round, primary_directory, primary_marker = primary_entry
        selected = {"primary": primary_entry, **{f"milestone_{key}": value for key, value in milestone_entries.items()}}
        checkpoint_exports = {}
        failed_compute = unaccounted_failure or any(s.get("unaccounted_prior_failure_detected") for s in sessions)
        phases_executed = not schedule.sequential or (schedule.phase1_seconds > 0 and schedule.phase2_seconds > 0)
        diagnostic_only = (args.mode == "budgeted" and
                           (schedule.summary()["budget_status"] != "within_allowance" or failed_compute or not phases_executed))
        if primary_round != terminal_round:
            assert export_verified_adapter(bridge, run / "adapter_terminal", final_state) == final_sha
        for label, (boundary, directory, selected_marker) in selected.items():
            bridge.load_checkpoint(directory / "engine")
            selected_state = bridge.adapter_snapshot()
            selected_sha = state_fingerprint(selected_state)
            assert selected_sha == selected_marker["adapter_sha256"]
            if label == "primary":
                destination = run / ("adapter_diagnostic_primary" if diagnostic_only else "adapter_final")
            else:
                destination = run / ("adapter_diagnostic_milestones" if diagnostic_only else "adapter_milestones") / label
            assert export_verified_adapter(bridge, destination, selected_state) == selected_sha
            checkpoint_exports[label] = {"boundary": boundary, "checkpoint_directory": str(directory),
                "adapter_directory": str(destination), "adapter_sha256": selected_sha,
                "charged_seconds": selected_marker["schedule_after"]["charged_seconds"],
                "actual_updates": selected_marker["actual_updates"],
                "schedule": selected_marker["schedule_after"], "diagnostic_only": bool(diagnostic_only),
                "eligible_for_formal_evaluation": args.mode == "budgeted" and not diagnostic_only,
                "selection_purpose": "engineering_validation" if args.mode == "validation" else
                                     "invalid_budget_diagnostic" if diagnostic_only else "predeclared_time_endpoint"}
        atomic_json(run / "checkpoint_selections.json", checkpoint_exports)
        # Restore terminal engine state after exporting time-selected checkpoints.
        bridge.load_checkpoint(final_checkpoint / "engine")
        assert state_fingerprint(bridge.adapter_snapshot()) == final_sha == final_payload["adapter_sha256"]
        assert bridge.optimizer_fingerprint() == final_optimizer_sha
        restore_rng(final_payload["rng"])
        assert rng_fingerprint(capture_rng()) == final_payload["rng_sha256"]
        assert bridge.base_fingerprint() == base_before
        torch.cuda.synchronize()
        final_seconds = time.perf_counter() - final_started
        status = "complete"
    sessions[-1].update(status=status, end_iteration=terminal_round, stop_reason=stop_reason,
                        schedule_after=schedule.state_dict(), base_sha256_after=base_after,
                        teacher_sha256_after=teacher_after,
                        final_audit_seconds=final_seconds if status == "complete" else 0.0,
                        pause_audit_seconds=final_seconds if status != "complete" else 0.0,
                        session_wall_seconds=time.perf_counter() - session_started,
                        framework_at_end=bridge.framework_metadata())
    write_jsonl(run / "sessions.jsonl", sessions)
    write_jsonl(run / "metrics.jsonl", history)
    atomic_json(run / "progress.json", {"status": status, "completed_iterations": terminal_round,
                "actual_optimizer_steps": actual_updates, "identity": identity, "schedule": schedule.summary(),
                "data_cursor": cursor.state_dict(), "stop_reason": stop_reason})
    if status != "complete":
        print(f"Stopped at complete native verl boundary {terminal_round}; use --resume.", flush=True)
        return
    budget = schedule.summary()
    any_failed_compute = unaccounted_failure or any(s.get("unaccounted_prior_failure_detected") for s in sessions)
    both_phases = not schedule.sequential or (schedule.phase1_seconds > 0 and schedule.phase2_seconds > 0)
    accounting_valid = (args.mode == "budgeted" and budget["budget_status"] == "within_allowance"
                        and both_phases and not any_failed_compute)
    summary = {"status": "complete", "method": args.method, "identity": identity, "mode": args.mode,
               "budget": budget, "budget_accounting_valid": accounting_valid,
               "unaccounted_failed_compute": bool(any_failed_compute),
               "both_sequential_phases_executed": both_phases,
               "selected_primary": checkpoint_exports["primary"], "data_cursor": cursor.state_dict(),
               "training_backend": "verl " + args.backend, "versions": versions,
               "iterations": len(history), "actual_optimizer_steps": actual_updates,
               "skipped_iterations": sum(metric["optimizer_steps"] == 0 for metric in history),
               "initial_adapter_sha256": initial_sha,
               "final_adapter_sha256": checkpoint_exports["primary"]["adapter_sha256"],
               "terminal_adapter_sha256": final_sha,
               "base_sha256_before": base_before, "base_sha256_after": base_after,
               "teacher_sha256_before": teacher_before, "teacher_sha256_after": teacher_after,
               "base_weights_unchanged": True, "teacher_weights_unchanged": teacher is not None,
               "teacher_gradients_absent": teacher is not None, "adapter_save_reload_exact": True,
               "native_adapter_reload_exact": True, "native_optimizer_scheduler_reload_exact": True,
               "controller_rng_restore_exact": True,
               "final_optimizer_scheduler_sha256": final_optimizer_sha,
               "framework": bridge.framework_metadata(), "final_audit_seconds": final_seconds,
               "peak_iteration_torch_allocated_gib": max((metric["peak_torch_allocated_gib"] for metric in history), default=0.0),
               "charged_training_seconds": schedule.charged_seconds,
               "uncharged_commit_tail_seconds": sum(record["uncharged_commit_tail_seconds"] for record in commit_tails),
               "charged_timing_scope": spec["budget_scope"],
               "checkpoint_retention": "latest two, timing-selected primary, and first crossing of 25/50/75 percent within allowance",
               "note": "Validation forces phase boundaries and is not equal-budget evidence. Budgeted mode uses a common upper measured-time allowance; unused allowance is not filled with waiting. Unexpected overrun invalidates budget compliance."}
    for key in ("questions", "responses", "generated_tokens", "correct", "mixed_reward_groups", "invalid_answers",
                "truncated_answers", "generation_wall_seconds", "old_policy_scoring_seconds", "teacher_scoring_seconds",
                "backward_and_optimizer_seconds", "training_core_seconds", "iteration_audit_seconds", "checkpoint_payload_write_seconds"):
        summary[key] = sum(metric[key] for metric in history)
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", run, flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # The executable owns its one-rank process group, including error paths.
        if _active_bridge is not None:
            _active_bridge.close()
        distributed = sys.modules.get("torch.distributed")
        if distributed is not None and distributed.is_initialized():
            distributed.destroy_process_group()
