"""Prepare an isolated workspace and run one declared training configuration."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from check_environment import check_assets
from prepare_assets import ROOT, read_json, require, sha256


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def materialize(workspace, assets):
    workspace, assets = Path(workspace).resolve(), Path(assets).resolve()
    require(not workspace.exists(), f"Workspace already exists: {workspace}")
    require(not ROOT.is_relative_to(workspace), "Workspace must not contain the source repository")
    checks = check_assets(assets, models=True)
    failures = [label for label, passed in checks if not passed]
    require(not failures, "Prepare assets first: " + ", ".join(failures))
    original = ROOT / "experiments"
    frozen = read_json(original / "verl_study/protocol.json")
    for relative, expected in frozen["frozen_source_sha256"].items():
        require(sha256(original / relative) == expected, f"Frozen source differs: {relative}")
    workspace.mkdir(parents=True)
    experiment = workspace / "experiments"
    experiment.mkdir()
    for path in original.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".txt"} and "__pycache__" not in path.parts:
            target = experiment / path.relative_to(original)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    (experiment / "data").mkdir()
    for name in ("train", "dev", "test"):
        shutil.copyfile(assets / "data" / f"{name}.jsonl", experiment / "data" / f"{name}.jsonl")
    asset_spec = read_json(ROOT / "configs/assets.json")
    config_path = experiment / "configs/pilot.json"
    config = read_json(config_path)
    for role, spec in asset_spec["models"].items():
        config["models"][role]["local_snapshot"] = str(assets / "models" / role / spec["revision"])
    write_json(config_path, config)
    pilot_path = experiment / "configs/repeated_pilot_v1.json"
    pilot = read_json(pilot_path)
    pilot["source_config_sha256"] = sha256(config_path)
    write_json(pilot_path, pilot)
    verl_path = experiment / "verl/protocol.json"
    verl = read_json(verl_path)
    verl["source_config_sha256"] = sha256(config_path)
    verl["source_protocol_sha256"] = sha256(pilot_path)
    verl["initial_adapter"] = str(assets / "initial_adapter")
    write_json(verl_path, verl)
    protocol_path = experiment / "verl_study/protocol.json"
    protocol = read_json(protocol_path)
    protocol["source_config_sha256"] = sha256(config_path)
    protocol["source_verl_protocol_sha256"] = sha256(verl_path)
    protocol["models_and_initialization"]["models"] = config["models"]
    protocol["models_and_initialization"]["initial_adapter_path"] = verl["initial_adapter"]
    for relative in protocol["frozen_source_sha256"]:
        protocol["frozen_source_sha256"][relative] = sha256(experiment / relative)
    protocol["reproduction"] = {
        "original_protocol_sha256": sha256(original / "verl_study/protocol.json"),
        "changes": "Asset paths and dependent configuration checksums only; Python sources, objectives, budgets, seeds and settings unchanged.",
        "results_scope": "New reproduction runs; not additions to the released study results.",
    }
    write_json(protocol_path, protocol)
    changed = {}
    for relative in ("configs/pilot.json", "configs/repeated_pilot_v1.json", "verl/protocol.json", "verl_study/protocol.json"):
        changed[relative] = {"original_sha256": sha256(original / relative),
                             "runtime_sha256": sha256(experiment / relative)}
    manifest = {"schema_version": 1, "assets": str(assets),
                "asset_manifest_sha256": sha256(ROOT / "configs/assets.json"),
                "original_protocol_sha256": sha256(original / "verl_study/protocol.json"),
                "runtime_protocol_sha256": sha256(protocol_path), "changed_configuration": changed,
                "python_source_sha256": {str(path.relative_to(experiment)): sha256(path)
                                         for path in sorted(experiment.rglob("*.py"))},
                "formal_queue_recreated": False}
    write_json(workspace / "reproduction_manifest.json", manifest)
    return manifest


def verify_workspace(workspace):
    manifest = read_json(workspace / "reproduction_manifest.json")
    require(sha256(workspace / "experiments/verl_study/protocol.json") == manifest["runtime_protocol_sha256"],
            "Runtime protocol changed")
    for relative, expected in manifest["python_source_sha256"].items():
        require(sha256(workspace / "experiments" / relative) == expected, f"Runtime source changed: {relative}")
    for relative, hashes in manifest["changed_configuration"].items():
        require(sha256(workspace / "experiments" / relative) == hashes["runtime_sha256"], f"Runtime config changed: {relative}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Copy frozen sources and relocate assets in a new workspace")
    prepare.add_argument("--workspace", type=Path, default=ROOT / "artifacts/reproduction")
    prepare.add_argument("--assets", type=Path, default=ROOT / "artifacts/assets")
    train = commands.add_parser("train", help="Run a single method and seed using the unchanged trainer")
    train.add_argument("--workspace", type=Path, default=ROOT / "artifacts/reproduction")
    train.add_argument("--method", choices=("rl", "opd", "joint", "opd_then_rl", "rl_then_opd"), required=True)
    train.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    train.add_argument("--name", required=True)
    train.add_argument("--check-only", action="store_true", help="Validate inputs without loading weights or training")
    train.add_argument("--validation", action="store_true", help="Run the existing two-round engineering check")
    train.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.command == "prepare":
        materialize(workspace, args.assets)
        print(f"Prepared {workspace}")
        print("No models were loaded and no training was started.")
        return
    verify_workspace(workspace)
    command = [sys.executable, str(workspace / "experiments/verl_study/train_verl_study.py"),
               "--project-root", str(workspace), "--method", args.method, "--seed", str(args.seed),
               "--name", args.name, "--mode", "validation" if args.validation else "budgeted"]
    for argument in ("check_only", "resume"):
        if getattr(args, argument):
            command.append("--" + argument.replace("_", "-"))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
