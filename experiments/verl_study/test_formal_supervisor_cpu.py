"""CPU-only process/control tests. No model imports, tensors, or GPU APIs."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import run_verl_formal_study as subject


def rejects(call):
    try:
        call()
    except RuntimeError:
        return
    raise AssertionError("Expected an explicit rejection")


def wait_for(path, timeout=5):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"Timed out waiting for {path}")
        time.sleep(0.01)


wrapper = Path(__file__).with_name("formal_child.py").resolve()
with tempfile.TemporaryDirectory(prefix="formal-supervisor-cpu-") as temporary:
    root = Path(temporary)
    # Every official test job must follow all 15 train/dev jobs and their audits.
    methods = ["rl", "opd", "joint", "opd_then_rl", "rl_then_opd"]
    declared = [{"order": i + 1, "method": method, "seed": seed, "name": f"train_{method}_{seed}"}
                for i, (method, seed) in enumerate((m, s) for s in (42, 43, 44) for m in methods)]
    jobs = subject.plan({"methods": methods, "seeds": [42, 43, 44], "run_order": {"runs": declared}})
    assert len(jobs) == 122 and sum(j["kind"] == "train" for j in jobs) == 15
    assert sum(j["kind"] == "eval" for j in jobs) == 46
    first_test = next(i for i, job in enumerate(jobs) if job.get("split") == "test")
    assert first_test == 90 and all(job.get("split") == "test" for job in jobs[first_test:])

    # A child retains the flock even after its parent closes its own descriptor.
    lockpath = root / "inherited.lock"
    child_code = "import sys; sys.stdin.read(1)"
    with subject.JobLock(lockpath) as lock:
        rejects(lambda: subject.JobLock(lockpath).__enter__())
        child = subprocess.Popen([sys.executable, "-c", child_code], stdin=subprocess.PIPE, pass_fds=(lock.fd,))
    rejects(lambda: subject.JobLock(lockpath).__enter__())
    child.communicate(b"x", timeout=5)
    with subject.JobLock(lockpath):
        pass

    # A SIGINT during simulated startup is deferred to the real target handler.
    target = root / "signal_target.py"
    entered, result = root / "entered", root / "signal-result.json"
    target.write_text("""import json, signal, sys, time
from pathlib import Path
entered, result = map(Path, sys.argv[1:])
entered.write_text('ready')
time.sleep(0.5)
observed = []
def request_boundary_stop(signum, frame):
    observed.append(signum)
signal.signal(signal.SIGINT, request_boundary_stop)
signal.signal(signal.SIGTERM, request_boundary_stop)
result.write_text(json.dumps(observed))
""", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(wrapper), "--program", str(target), "--", str(entered), str(result)])
    wait_for(entered)
    child.send_signal(signal.SIGINT)
    assert child.wait(timeout=5) == 0
    assert subject.read(result) == [signal.SIGINT]

    # Use real fake child processes to test pause, ledger and restart controls.
    project = root / "project"
    directory = project / "experiments/verl_study/formal_supervisor"
    directory.mkdir(parents=True)
    fake = root / "fake_child.py"
    fake.write_text("""import argparse, json, signal, time, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--resume',action='store_true');p.add_argument('--fail',action='store_true');p.add_argument('--early-pause',action='store_true');a=p.parse_args()
time.sleep(0.8)
pause=False
def request_boundary_stop(signum, frame):
    global pause
    pause=True
signal.signal(signal.SIGINT, request_boundary_stop)
signal.signal(signal.SIGTERM, request_boundary_stop)
if pause and a.early_pause: sys.exit(0)
if a.fail: raise SystemExit(7)
for i in range(3):
    if pause: break
    time.sleep(0.1)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'status':'paused' if pause else 'complete','resumed':a.resume}))
""", encoding="utf-8")
    tiny_jobs = [{"id": f"00{i + 1}_train_fake{i}", "kind": "train", "name": f"fake{i}", "method": "rl", "seed": 42}
                 for i in range(2)]
    identity = {"protocol_sha256": "cpu-test", "execution_manifest_sha256": "cpu-test", "jobs": tiny_jobs}

    def initial():
        return {"status": "starting", "identity": identity, "identity_sha256": subject.canonical(identity),
                "jobs": [{"job": j, "status": "pending", "attempts": []} for j in tiny_jobs]}

    def model_state(project, job, unused):
        output = project / (job["name"] + ".json")
        if not output.exists():
            return "new", {}
        return subject.read(output)["status"], {"artifacts_sha256": subject.artifact_manifest([output])}

    def fake_command(project, job, state, attempt_number, resume=False):
        command = [sys.executable, str(wrapper), "--program", str(fake), "--", "--output", str(project / (job["name"] + ".json"))]
        return command + (["--resume"] if resume else [])

    def fake_result(project, job, command, unused):
        return model_state(project, job, None)

    with patch.object(subject, "frozen_inputs", return_value=(None, None, identity)), \
         patch.object(subject, "training_state", side_effect=model_state), \
         patch.object(subject, "command_for", side_effect=fake_command), \
         patch.object(subject, "completed_result", side_effect=fake_result):
        state = initial()
        with subject.JobLock(directory / "local_gpu_job.lock") as lock:
            supervisor = subject.Supervisor(project, directory, identity, state, lock.fd, root / "status.json")
            timer = threading.Timer(0.1, lambda: subject.atomic(directory / "pause.request", {"requested": True}))
            timer.start()
            supervisor.execute()
            timer.join()
        assert state["status"] == "paused" and state["jobs"][0]["status"] == "paused"
        assert state["jobs"][1]["attempts"] == []
        assert state["jobs"][0]["attempts"][0]["pause_signal_sent_utc"]
        assert subject.read(root / "status.json")["pause_requested"] is True
        (directory / "pause.request").unlink()
        with subject.JobLock(directory / "local_gpu_job.lock") as lock:
            restored = subject.read(directory / "state.json")
            resumed = subject.Supervisor(project, directory, identity, restored, lock.fd)
            resumed.recover()
            resumed.execute()
        assert restored["status"] == "complete"
        assert [len(j["attempts"]) for j in restored["jobs"]] == [2, 1]
        assert subject.read(project / "fake0.json")["resumed"] is True
        assert len(list((project / "experiments/run_logs").glob("*.log"))) == 3
        assert all(a["process_wall_seconds"] > 0 for j in restored["jobs"] for a in j["attempts"])
        assert all(a["whole_job_wall_seconds"] >= a["process_wall_seconds"] for j in restored["jobs"] for a in j["attempts"])

        # A dead supervisor may reconcile a clean completed child, explicitly
        # recording that its exact whole-process wall duration is unavailable.
        interrupted = copy.deepcopy(restored)
        interrupted["status"] = "running"
        interrupted["jobs"][0]["status"] = "running"
        with subject.JobLock(directory / "local_gpu_job.lock") as lock:
            recovery = subject.Supervisor(project, directory, identity, interrupted, lock.fd)
            recovery.recover()
        assert interrupted["jobs"][0]["status"] == "complete"
        assert interrupted["jobs"][0]["attempts"][-1]["process_wall_seconds"] is None
        # An unclean child is rejected rather than replayed.
        interrupted["jobs"][0]["status"] = "running"
        with patch.object(subject, "completed_result", side_effect=RuntimeError("unclean training")):
            rejects(recovery.recover)

    # A requested evaluator pause before model/run setup must remain a fresh job,
    # not be mislabeled as failed or resumed with a nonexistent run directory.
    early_project = root / "early-project"
    early_directory = early_project / "experiments/verl_study/formal_supervisor"
    early_jobs = [{"id":"001_eval_early", "kind":"eval", "name":"early", "split":"dev64", "selection":"initial"}]
    early_identity = {**identity, "jobs": early_jobs}
    early_state = {"status":"starting", "jobs":[{"job":early_jobs[0],"status":"pending","attempts":[]}]}
    def early_command(*args, **kwargs):
        return fake_command(*args, **kwargs) + ["--early-pause"]
    with patch.object(subject, "frozen_inputs", return_value=(None,None,early_identity)), \
         patch.object(subject, "command_for", side_effect=early_command), \
         subject.JobLock(early_directory / "local_gpu_job.lock") as lock:
        early = subject.Supervisor(early_project,early_directory,early_identity,early_state,lock.fd)
        timer=threading.Timer(0.1,lambda:subject.atomic(early_directory / "pause.request",{"requested":True}))
        timer.start()
        early.execute()
        timer.join()
    assert early_state["status"]=="paused" and early_state["jobs"][0]["status"]=="pending"
    assert early_state["jobs"][0]["attempts"][0]["status"]=="paused_before_work"

    # Audit-selected files must stay exactly bound to the selected checkpoint.
    train = project / "experiments/runs/audited_train"
    adapter = train / "adapter_final"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"cpu fixture")
    entry = {"boundary": 12, "adapter_directory": str(adapter), "adapter_sha256": "tensor-state-hash",
             "eligible_for_formal_evaluation": True, "diagnostic_only": False}
    subject.atomic(train / "checkpoint_selections.json", {"primary": entry})
    report = {"status": "pass", "run_name": "audited_train", "selected_primary": {
        **entry, "adapter_files_sha256": {"adapter_model.safetensors": subject.sha(adapter / "adapter_model.safetensors")}}}
    audit_path = root / "audit.json"
    subject.atomic(audit_path, report)
    ledger = {"jobs": [{"job": {"kind": "train_audit", "name": "audited_train"}, "status": "complete",
                        "result": {"audit_output": str(audit_path), "artifacts_sha256": subject.artifact_manifest([audit_path])}}]}
    assert subject.audit_selection(project, "audited_train", "primary", ledger) == adapter
    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    rejects(lambda: subject.audit_selection(project, "audited_train", "primary", ledger))

    # Actual failing child propagates failure before a second model process starts.
    failed_project = root / "failed-project"
    failed_directory = failed_project / "experiments/verl_study/formal_supervisor"
    failed_state = initial()
    def failing_command(*args, **kwargs):
        return fake_command(*args, **kwargs) + ["--fail"]
    with patch.object(subject, "frozen_inputs", return_value=(None, None, identity)), \
         patch.object(subject, "training_state", return_value=("new", {})), \
         patch.object(subject, "command_for", side_effect=failing_command), \
         subject.JobLock(failed_directory / "local_gpu_job.lock") as lock:
        runner = subject.Supervisor(failed_project, failed_directory, identity, failed_state, lock.fd)
        rejects(runner.execute)
    assert failed_state["jobs"][0]["attempts"][0]["exit_code"] == 7
    assert failed_state["jobs"][1]["attempts"] == []

print(json.dumps({"status": "PASS", "gpu_used": False, "formal_plan_jobs": 122,
                  "tests": ["official_test_phase_after_all_training_and_dev_audits", "exclusive_and_inherited_flock",
                            "deferred_startup_sigint_once", "clean_pause_then_separate_process_resume",
                            "durable_ledger_and_exclusive_attempt_logs", "whole_job_wall_measurements",
                            "clean_child_recovery_with_unknown_wall_explicit", "unclean_child_resume_rejected",
                            "evaluation_pause_before_model_setup", "audited_adapter_tamper_rejected",
                            "child_failure_stops_queue"]}, indent=2))
