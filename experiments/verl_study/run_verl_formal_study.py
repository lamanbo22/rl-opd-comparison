"""Durable serial supervisor for the frozen fifteen-run local RTX 5090 study.

Default/--check-only validates and prints the plan. --execute starts it;
--execute --resume continues only verified complete or cleanly paused work.
--request-pause never launches a child. No training configuration is changed.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), f"Path escapes project: {relative}")
    return path


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "linux_start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "exited_before_identity_read": True}


class JobLock:
    """A child inherits this FD, keeping the GPU lock if its supervisor dies."""
    def __init__(self, path):
        self.path = Path(path)
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError("Another supervisor or its child still holds the local GPU job lock.")
        return self

    def __exit__(self, *unused):
        # Closing preserves an inherited child's lock; explicit LOCK_UN would not.
        if self.fd is not None:
            os.close(self.fd)


def plan(protocol):
    declared = protocol["run_order"]["runs"]
    require(len(declared) == 15 and [r["order"] for r in declared] == list(range(1, 16)),
            "Frozen plan must contain fifteen ordered runs.")
    require({(r["method"], r["seed"]) for r in declared} ==
            {(m, s) for m in protocol["methods"] for s in protocol["seeds"]}, "Method/seed plan differs.")
    require(len({r["name"] for r in declared}) == 15, "Duplicate training name.")
    jobs = []

    def add(kind, name, **extra):
        require(Path(name).name == name and name not in (".", ".."), "Unsafe run name.")
        jobs.append({"id": f"{len(jobs) + 1:03d}_{kind}_{name}", "kind": kind, "name": name, **extra})

    for run in declared:
        name = run["name"]
        add("train", name, method=run["method"], seed=run["seed"], train_name=name)
        add("train_audit", name, train_name=name)
        for split, suffix, selection in (("dev64", "dev64", "milestone_050"),
                                         ("dev512", "dev512", "primary")):
            eval_name = f"{name}_{suffix}"
            add("eval", eval_name, train_name=name, split=split, selection=selection)
            add("eval_audit", eval_name, train_name=name, split=split, selection=selection)
    add("eval", "verl_study_initial_test_01", split="test", selection="initial")
    add("eval_audit", "verl_study_initial_test_01", split="test", selection="initial")
    for run in declared:
        name = f"{run['name']}_test"
        add("eval", name, train_name=run["name"], split="test", selection="primary")
        add("eval_audit", name, train_name=run["name"], split="test", selection="primary")
    return jobs


def frozen_inputs(project, require_self=True):
    experiment = project / "experiments"
    study = experiment / "verl_study"
    protocol_path = study / "protocol.json"
    manifest_path = study / "execution_manifest.json"
    protocol, manifest = read(protocol_path), read(manifest_path)
    require(protocol["status"] == manifest["status"] == "frozen", "Both protocols must be frozen.")
    require(manifest["protocol_sha256"] == sha(protocol_path), "Original protocol hash differs.")
    require(protocol["endpoint_policy"] == "guarded_ceiling" and protocol["budget_seconds"] == 7200,
            "Unexpected frozen endpoint policy.")
    hardware = protocol["hardware"]
    require(hardware["gpu_count"] == hardware["distributed_world_size"] == 1 and
            not hardware["remote_compute_allowed"] and hardware["jobs_run_sequentially"],
            "Only the one local GPU plan is supported.")
    required = {"verl_study/run_verl_formal_study.py", "verl_study/formal_child.py",
                "verl_study/evaluate_verl_formal.py", "verl_study/audit_verl_formal.py"}
    require(required <= manifest["source_sha256"].keys(), "Supplemental source manifest is incomplete.")
    for mapping in (protocol["frozen_source_sha256"], manifest["source_sha256"]):
        for relative, expected in mapping.items():
            require(sha(contained(experiment, relative)) == expected, f"Frozen source changed: {relative}")
    require(sha(study / "formal_test/manifest.json") == manifest["formal_test_manifest_sha256"],
            "Official test manifest differs.")
    if require_self:
        require(sha(__file__) == manifest["source_sha256"]["verl_study/run_verl_formal_study.py"],
                "The actual supervisor source is not pinned.")
    require((project / ".venv-verl/bin/python").is_file(), "Pinned interpreter is missing.")
    jobs = plan(protocol)
    identity = {"protocol_sha256": sha(protocol_path), "execution_manifest_sha256": sha(manifest_path),
                "project_root": str(project), "jobs": jobs}
    return protocol, manifest, identity


def artifact_manifest(paths):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def verify_artifacts(result):
    for path, expected in result.get("artifacts_sha256", {}).items():
        require(sha(path) == expected, f"Previously verified artifact changed: {path}")


def training_state(project, job, protocol_sha):
    """Only clean boundaries can be resumed; comprehensive completion audit is next."""
    run = project / "experiments/runs" / job["name"]
    if not run.exists():
        return "new", {}
    metadata = read(run / "metadata.json")
    spec = metadata["spec"]
    require(spec["protocol_sha256"] == protocol_sha and spec["mode"] == "budgeted" and
            spec["method"] == job["method"] and spec["seed"] == job["seed"], "Training identity mismatch.")
    sessions, history = rows(run / "sessions.jsonl"), rows(run / "metrics.jsonl")
    require(sessions and all(s["status"] in ("paused_at_boundary", "complete") and
                            not s.get("unaccounted_prior_failure_detected") for s in sessions),
            "Unclean training interruption: automatic resume is prohibited.")
    require(not any((run / "checkpoints").glob(".pending_*")), "Uncommitted training checkpoint exists.")
    committed = {m["evidence_directory"] for m in history}
    require({str(p.relative_to(run)) for p in (run / "iterations").iterdir()} == committed,
            "Uncommitted training iteration exists; its cost cannot be ignored.")
    progress = read(run / "progress.json")
    require(progress["completed_iterations"] == len(history), "Training progress/history mismatch.")
    require(progress["data_cursor"]["questions_seen"] == 4 * len(history), "Training cursor mismatch.")
    markers = [read(p) for p in (run / "checkpoints").glob("boundary_*/complete.json")]
    latest = max(markers, key=lambda value: value["boundary"])
    require(latest["identity"] == metadata["identity"] and latest["boundary"] == len(history),
            "Last complete native boundary does not match history.")
    require(latest["schedule_after"]["charged_seconds"] == progress["schedule"]["charged_seconds"],
            "Committed budget and progress differ.")
    common = [run / n for n in ("metadata.json", "sessions.jsonl", "metrics.jsonl", "progress.json")]
    if progress["status"] == "paused_at_boundary":
        require(not (run / "summary.json").exists(), "Paused training has a stale final summary.")
        return "paused", {"artifacts_sha256": artifact_manifest(common), "completed_rounds": len(history)}
    require(progress["status"] == "complete", "Training did not close at a safe boundary.")
    summary, selections = read(run / "summary.json"), read(run / "checkpoint_selections.json")
    selected = selections["primary"]
    require(summary["status"] == "complete" and summary["budget_accounting_valid"] and
            not summary["unaccounted_failed_compute"] and summary["both_sequential_phases_executed"],
            "Training is incomplete or ineligible; stop without a replacement run.")
    require(summary["selected_primary"] == selected and selected["eligible_for_formal_evaluation"] and
            not selected["diagnostic_only"] and selected["boundary"] == len(history),
            "Selected endpoint is not the valid completed training state.")
    require(summary["budget"]["budget_status"] == "within_allowance" and
            6720 < summary["charged_training_seconds"] <= 7200, "Guarded training allowance mismatch.")
    require((run / "adapter_final").resolve() == Path(selected["adapter_directory"]).resolve(),
            "Unexpected final adapter directory.")
    common += [run / "summary.json", run / "checkpoint_selections.json"]
    common += list((run / "adapter_final").glob("*"))
    return "complete", {"artifacts_sha256": artifact_manifest(p for p in common if p.is_file()),
                         "completed_rounds": len(history), "charged_training_seconds": summary["charged_training_seconds"],
                         "selected_primary": selected}


def evaluation_state(project, job):
    run = project / "experiments/runs" / job["name"]
    if not run.exists():
        return "new", {}
    sessions = [read(p) for p in sorted((run / "sessions").glob("*.json"))]
    require(sessions and all(s["status"] in ("paused", "complete") for s in sessions),
            "Unclean evaluation interruption needs inspection before resume.")
    require(all(s["all_parameter_hash_before"] == s["all_parameter_hash_after"] for s in sessions),
            "Evaluation session lacks closing unchanged-weight verification.")
    if not (run / "summary.json").exists():
        require(sessions[-1]["status"] == "paused", "Partial evaluation did not pause cleanly.")
        return "paused", {}
    summary = read(run / "summary.json")
    require(summary["status"] == "complete" and summary["questions"] == {"dev64": 64, "dev512": 512, "test": 1319}[job["split"]],
            "Evaluation completeness mismatch.")
    require(summary.get("all_committed_batches_have_closing_full_hash") is True,
            "Evaluation contains unclosed batch sessions.")
    paths = [run / n for n in ("metadata.json", "initial_weights.json", "responses.jsonl", "summary.json")]
    paths += list((run / "batches").glob("*.json")) + list((run / "sessions").glob("*.json"))
    return "complete", {"artifacts_sha256": artifact_manifest(paths), "questions": summary["questions"]}


def audit_selection(project, train_name, selection, state):
    matches = [j for j in state["jobs"] if j["job"]["kind"] == "train_audit" and j["job"]["name"] == train_name]
    require(len(matches) == 1 and matches[0]["status"] == "complete", "Training has no successful formal audit.")
    result = matches[0]["result"]
    verify_artifacts(result)
    report = read(result["audit_output"])
    require(report["status"] == "pass" and report["run_name"] == train_name, "Training audit did not pass.")
    entry = report["selected_primary" if selection == "primary" else "milestone_050"]
    actual = read(project / "experiments/runs" / train_name / "checkpoint_selections.json")[selection]
    for key in ("boundary", "adapter_directory", "adapter_sha256"):
        require(actual[key] == entry[key], f"Audit-selected adapter differs: {key}")
    require(actual["eligible_for_formal_evaluation"] and not actual["diagnostic_only"],
            "Adapter is diagnostic-only.")
    directory = Path(actual["adapter_directory"]).resolve()
    require(directory.is_relative_to((project / "experiments/runs" / train_name).resolve()),
            "Selected adapter is outside its training run.")
    # The audit report binds these files; never trust an adapter path alone.
    file_hashes = entry.get("adapter_files_sha256")
    require(isinstance(file_hashes, dict) and "adapter_model.safetensors" in file_hashes,
            "Audit-selected adapter lacks file hashes.")
    for relative, expected in file_hashes.items():
        require(sha(contained(directory, relative)) == expected, "Audited adapter file changed.")
    return directory


def command_for(project, job, state, attempt_number, resume=False):
    study = project / "experiments/verl_study"
    python = str(project / ".venv-verl/bin/python")
    base = [python, "-B", "-u"]
    if job["kind"] == "train":
        command = base + [str(study / "formal_child.py"), "--program", str(study / "train_verl_study.py"), "--",
            "--project-root", str(project), "--protocol", str(study / "protocol.json"),
            "--mode", "budgeted", "--method", job["method"], "--seed", str(job["seed"]), "--name", job["name"]]
    elif job["kind"] == "train_audit":
        output = study / "formal_supervisor/audits" / f"{job['id']}_attempt{attempt_number:03d}.json"
        require(not output.exists(), "Refusing to overwrite an audit output.")
        command = base + [str(study / "audit_verl_formal.py"), "--project-root", str(project),
                         "--run-name", job["name"], "--output", str(output)]
    else:
        args = ["--project-root", str(project), "--split", job["split"], "--name", job["name"]]
        if job["selection"] != "initial":
            args += ["--adapter", str(audit_selection(project, job["train_name"], job["selection"], state))]
        if job["kind"] == "eval_audit":
            command = base + [str(study / "evaluate_verl_formal.py"), *args, "--audit-only"]
        else:
            command = base + [str(study / "formal_child.py"), "--program", str(study / "evaluate_verl_formal.py"), "--", *args]
    if resume:
        require(job["kind"] in ("train", "eval"), "Only model jobs resume native state.")
        command.append("--resume")
    return command


def completed_result(project, job, command, identity):
    if job["kind"] == "train":
        return training_state(project, job, identity["protocol_sha256"])
    if job["kind"] in ("eval", "eval_audit"):
        return evaluation_state(project, job)
    output = Path(command[command.index("--output") + 1])
    report = read(output)
    require(report["status"] == "pass" and report["run_name"] == job["name"] and
            report["protocol_sha256"] == identity["protocol_sha256"], "Formal training audit failed.")
    run = project / "experiments/runs" / job["name"]
    names = {"summary_sha256": "summary.json", "metadata_sha256": "metadata.json", "metrics_sha256": "metrics.jsonl",
             "sessions_sha256": "sessions.jsonl", "checkpoint_selections_sha256": "checkpoint_selections.json"}
    for field, filename in names.items():
        require(report[field] == sha(run / filename), f"Audit bound different {filename}.")
    evidence = report["evidence_files_sha256"]
    require(evidence and any(path.startswith("checkpoints/") for path in evidence), "Native evidence manifest is missing.")
    for relative, expected in evidence.items():
        require(sha(contained(run, relative)) == expected, f"Audited evidence changed: {relative}")
    return "complete", {"audit_output": str(output),
                         "artifacts_sha256": {**artifact_manifest([output, *[run / n for n in names.values()]]),
                                              **{str(contained(run, key)): value for key, value in evidence.items()}}}


class Supervisor:
    def __init__(self, project, state_dir, identity, state, lock_fd, status_copy=None):
        self.project, self.directory, self.identity, self.state = project, state_dir, identity, state
        self.lock_fd, self.status_copy = lock_fd, status_copy
        self.pause = False
        self.child = None
        self.current_attempt = None
        self.child_started = None

    def signal_pause(self, signum, frame):
        self.pause = True

    def pause_requested(self):
        return self.pause or (self.directory / "pause.request").exists()

    def drain_child_after_error(self):
        """A failed status write must not leave a live, uncontrollable GPU job."""
        if self.child is None:
            return
        attempt = self.current_attempt
        if self.child.poll() is None and not attempt.get("pause_signal_sent_utc"):
            try:
                self.child.send_signal(signal.SIGINT)
                attempt["pause_signal_sent_utc"] = utc()
            except ProcessLookupError:
                pass
        code = self.child.wait()  # finish the boundary; never kill or time out it
        attempt.update(exit_code=code, ended_utc=utc(),
                       process_wall_seconds=time.monotonic() - self.child_started,
                       status="closed_after_supervisor_error")
        self.child = None
        self.state["active_child"] = None

    def compact(self):
        jobs = self.state["jobs"]
        result = {key: self.state.get(key) for key in ("status", "updated_utc", "supervisor", "active_child", "current_job", "error")}
        result.update(completed_training_runs=sum(j["status"] == "complete" and j["job"]["kind"] == "train" for j in jobs),
                      audited_training_runs=sum(j["status"] == "complete" and j["job"]["kind"] == "train_audit" for j in jobs),
                      completed_evaluations=sum(j["status"] == "complete" and j["job"]["kind"] == "eval" for j in jobs),
                      completed_jobs=sum(j["status"] == "complete" for j in jobs), total_jobs=len(jobs),
                      pause_requested=self.pause_requested(), execution_manifest_sha256=self.identity["execution_manifest_sha256"])
        current = self.state.get("current_job")
        if current and current["kind"] == "train":
            path = self.project / "experiments/runs" / current["name"] / "progress.json"
            if path.exists():
                result["training_progress"] = read(path)
        return result

    def save(self, event=None):
        self.state["updated_utc"] = utc()
        atomic(self.directory / "state.json", self.state)
        if event:
            atomic(self.directory / "events" / f"{time.time_ns()}_{uuid.uuid4().hex[:8]}.json",
                   {"at_utc": utc(), "event": event, "status": self.state["status"],
                    "current_job": self.state.get("current_job"), "active_child": self.state.get("active_child")})
        if self.status_copy:
            atomic(self.status_copy, self.compact())

    def recover(self):
        """Reconcile a dead supervisor only from closed child evidence."""
        require(self.state["status"] != "failed", "Failed suite requires inspection; no automatic rescue/replacement.")
        for entry in self.state["jobs"]:
            job = entry["job"]
            if entry["status"] == "complete":
                verify_artifacts(entry["result"])
            elif entry["status"] == "running":
                attempt = entry["attempts"][-1]
                if job["kind"] in ("train", "eval"):
                    status, result = completed_result(self.project, job, attempt["command"], self.identity)
                    early_pause = (status == "new" and job["kind"] == "eval" and attempt.get("pause_signal_sent_utc"))
                    require(status in ("complete", "paused") or early_pause, "Abandoned child did not close safely.")
                    entry.update(status="pending" if early_pause else status, result=result)
                else:
                    # CPU audits are idempotent and create a new output/log per attempt.
                    entry["status"] = "pending"
                attempt.update(status="supervisor_recovery", process_wall_seconds=None,
                               whole_job_wall_seconds=None,
                               cost_accounting="Supervisor died before recording child exit; exact whole-job wall time is unknown.",
                               recovered_utc=utc())
                self.save("reconciled_clean_child_after_supervisor_restart")
            elif entry["status"] not in ("pending", "paused"):
                raise RuntimeError("Failed or unknown job state prevents automatic resume.")
        self.state.update(active_child=None, current_job=None)

    def execute(self):
        environment = dict(os.environ, OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                           HF_HUB_OFFLINE="1", WANDB_MODE="disabled", NCCL_DEBUG="WARN")
        for entry in self.state["jobs"]:
            job = entry["job"]
            if entry["status"] == "complete":
                continue
            if self.pause_requested():
                break
            job_started = time.monotonic()
            _, _, current_identity = frozen_inputs(self.project)
            require(current_identity == self.identity, "Pinned execution inputs changed during the study.")
            if job.get("split") == "test":
                require(all(j["status"] == "complete" for j in self.state["jobs"]
                            if j["job"].get("split") != "test"), "Test access cannot precede all training/development jobs.")
            resume = entry["status"] == "paused"
            if job["kind"] in ("train", "eval"):
                status, result = (training_state(self.project, job, self.identity["protocol_sha256"])
                                  if job["kind"] == "train" else evaluation_state(self.project, job))
                require((resume and status == "paused") or (not resume and status == "new"),
                        "Existing model run is not the exact clean paused job in this ledger.")
            number = len(entry["attempts"]) + 1
            command = command_for(self.project, job, self.state, number, resume=resume)
            log = self.project / "experiments/run_logs" / f"formal_{job['id']}_attempt{number:03d}.log"
            require(not log.exists(), "Refusing to overwrite a process log.")
            log.parent.mkdir(parents=True, exist_ok=True)
            attempt = {"number": number, "command": command, "log": str(log), "status": "starting", "started_utc": utc()}
            attempt["supervisor_preparation_seconds"] = time.monotonic() - job_started
            entry["attempts"].append(attempt)
            entry["status"] = "running"
            self.state.update(status="running", current_job=job, active_child=None)
            self.save("job_starting")
            started = time.monotonic()
            self.current_attempt, self.child_started = attempt, started
            with log.open("x", encoding="utf-8") as stream:
                self.child = subprocess.Popen(command, cwd=self.project, env=environment,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(self.lock_fd,))
                attempt.update(status="running", child=process_identity(self.child.pid))
                self.state["active_child"] = attempt["child"]
                self.save("child_started")
                sent_pause = False
                heartbeat = time.monotonic()
                while self.child.poll() is None:
                    if self.pause_requested() and not sent_pause and job["kind"] in ("train", "eval"):
                        try:
                            self.child.send_signal(signal.SIGINT)
                            sent_pause = True
                            attempt["pause_signal_sent_utc"] = utc()
                            self.state["status"] = "pausing"
                            self.save("clean_boundary_pause_requested")
                        except ProcessLookupError:
                            pass
                    if time.monotonic() - heartbeat >= 10:
                        attempt["elapsed_wall_seconds_so_far"] = time.monotonic() - started
                        self.save()
                        heartbeat = time.monotonic()
                    time.sleep(0.5)
                code = self.child.wait()
                elapsed = time.monotonic() - started
                stream.write(f"\nPROCESS_EXIT_CODE: {code}\n")
                stream.flush()
                os.fsync(stream.fileno())
            attempt.update(status="exited", exit_code=code, ended_utc=utc(), process_wall_seconds=elapsed,
                           whole_job_wall_seconds=time.monotonic() - job_started,
                           log_sha256=sha(log))
            self.child = None
            self.state["active_child"] = None
            self.save("child_exited")
            require(code == 0, f"Job failed with exit code {code}; inspect {log}.")
            verification_started = time.monotonic()
            status, result = completed_result(self.project, job, command, self.identity)
            early_pause = status == "new" and job["kind"] == "eval" and sent_pause
            require(status in ("complete", "paused") or early_pause, "Child exited without complete/clean paused evidence.")
            require(status == "complete" or job["kind"] in ("train", "eval"), "An audit did not complete.")
            entry.update(status="pending" if early_pause else status, result=result)
            attempt["postprocess_verification_seconds"] = time.monotonic() - verification_started
            attempt["whole_job_wall_seconds"] = time.monotonic() - job_started
            attempt["status"] = "paused_before_work" if early_pause else status
            self.save("job_" + attempt["status"])
            if status == "paused" or early_pause:
                self.pause = True
            if self.pause_requested():
                break
        all_complete = all(entry["status"] == "complete" for entry in self.state["jobs"])
        self.state.update(status="complete" if all_complete else "paused", current_job=None, active_child=None)
        self.save("suite_" + self.state["status"])


def hardware_check():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"], text=True)
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    require(len(lines) == 1 and "5090" in lines[0], "Formal execution requires exactly one local RTX 5090.")
    return {"nvidia_smi_gpu": lines[0], "host": os.uname().nodename}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/home/enci/project/rl_opd_comparison"))
    parser.add_argument("--status-copy", type=Path, help="Optional compact JSON mirror, e.g. a mounted host output path.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--request-pause", action="store_true")
    modes.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(not args.resume or args.execute, "--resume requires --execute.")
    project = args.project_root.resolve()
    directory = project / "experiments/verl_study/formal_supervisor"
    state_path = directory / "state.json"
    if args.request_pause:
        atomic(directory / "pause.request", {"requested_utc": utc(), "requested_by": "command", "request_id": uuid.uuid4().hex})
        state = read(state_path) if state_path.exists() else {}
        print(json.dumps({"pause_requested": True, "state_path": str(state_path),
                          **{key: state.get(key) for key in ("status", "current_job", "active_child")}}, indent=2))
        return
    if args.status:
        print(json.dumps(read(state_path) if state_path.exists() else {"status": "not_started"}, indent=2))
        return
    protocol, manifest, identity = frozen_inputs(project)
    if not args.execute:
        print(json.dumps({"status": "check_only_pass", "jobs": len(identity["jobs"]),
                          "training_runs": 15, "evaluations": 46, "identity_sha256": canonical(identity),
                          "state_path": str(state_path), "plan": identity["jobs"]}, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with JobLock(directory / "local_gpu_job.lock") as lock:
        pause_path = directory / "pause.request"
        acknowledged_pause = pause_path.read_bytes() if args.resume and pause_path.exists() else None
        if args.resume:
            state = read(state_path)
            require(state["identity"] == identity and state["identity_sha256"] == canonical(identity),
                    "Supervisor execution identity changed.")
        else:
            require(not state_path.exists(), "Supervisor state exists; inspect it and use --execute --resume.")
            for job in identity["jobs"]:
                if job["kind"] in ("train", "eval"):
                    require(not (project / "experiments/runs" / job["name"]).exists(),
                            f"Existing run is outside the new ledger: {job['name']}")
            state = {"schema_version": 1, "identity": identity, "identity_sha256": canonical(identity),
                     "status": "starting", "created_utc": utc(), "jobs": [
                         {"job": job, "status": "pending", "attempts": []} for job in identity["jobs"]]}
        supervisor = Supervisor(project, directory, identity, state, lock.fd, args.status_copy)
        if args.resume:
            supervisor.recover()
            request = directory / "pause.request"
            if acknowledged_pause is not None and request.exists() and request.read_bytes() == acknowledged_pause:
                archived = directory / "acknowledged_pause_requests" / f"{time.time_ns()}.json"
                archived.parent.mkdir(exist_ok=True)
                os.replace(request, archived)
        state.update(supervisor=process_identity(os.getpid()), hardware=hardware_check(), status="running")
        session = {"started_utc": utc(), "supervisor": state["supervisor"], "resume": args.resume, "status": "running"}
        state.setdefault("supervisor_sessions", []).append(session)
        session_started = time.monotonic()
        old_handlers = {sig: signal.signal(sig, supervisor.signal_pause) for sig in (signal.SIGTERM, signal.SIGINT)}
        supervisor.save("supervisor_resumed" if args.resume else "supervisor_started")
        try:
            supervisor.execute()
        except BaseException as error:
            # No kill/terminate path: request one boundary stop and wait for it.
            supervisor.drain_child_after_error()
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            supervisor.save("supervisor_failed")
            raise
        finally:
            session.update(status=state["status"], ended_utc=utc(),
                           known_wall_seconds=time.monotonic() - session_started)
            supervisor.save("supervisor_session_closed")
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        print(json.dumps(supervisor.compact(), indent=2), flush=True)


if __name__ == "__main__":
    main()
