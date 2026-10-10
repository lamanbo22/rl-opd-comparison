"""Check prepared assets and the pinned Linux training environment."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

from prepare_assets import ROOT, read_json, sha256


def check_assets(directory, models=False):
    manifest = read_json(ROOT / "configs/assets.json")
    checks = []
    for name, spec in manifest["dataset"]["splits"].items():
        path = directory / "data" / f"{name}.jsonl"
        checks.append((f"{name} data", path.is_file() and sha256(path) == spec["sha256"]))
    if models:
        for role, spec in manifest["models"].items():
            for name, expected in spec["files_sha256"].items():
                path = directory / "models" / role / spec["revision"] / name
                checks.append((f"{role}/{name}", path.is_file() and sha256(path) == expected))
        adapter = directory / "initial_adapter/adapter_model.safetensors"
        checks.append(("initial adapter", adapter.is_file() and sha256(adapter) == manifest["initial_adapter"]["weights_sha256"]))
        config = directory / "initial_adapter/adapter_config.json"
        checks.append(("adapter config", config.is_file() and config.read_bytes()
                       == (ROOT / manifest["initial_adapter"]["config_file"]).read_bytes()))
    return checks


def check_training_environment(verl_source):
    protocol = read_json(ROOT / "experiments/verl_study/protocol.json")
    checks = [("Linux or WSL", platform.system() == "Linux"),
              ("Python " + protocol["frozen_python_version"], platform.python_version() == protocol["frozen_python_version"])]
    for package, expected in protocol["frozen_versions"].items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        checks.append((f"{package}: expected {expected}, found {actual}", actual == expected))
    try:
        commit = subprocess.check_output(["git", "-C", str(verl_source), "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "-C", str(verl_source), "diff", "--binary", "HEAD"])
        checks.append(("verl commit", commit == protocol["frozen_verl_commit"]))
        checks.append(("clean tracked verl source", not diff))
    except (OSError, subprocess.CalledProcessError):
        checks.append(("verl checkout", False))
    spec = importlib.util.find_spec("verl")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    checks.append(("verl import uses the pinned checkout", origin is not None
                   and origin.is_relative_to((verl_source / "verl").resolve())))
    try:
        import torch
        count = torch.cuda.device_count()
        checks.append(("one visible RTX 5090", count == 1 and "5090" in torch.cuda.get_device_name(0)))
    except (ImportError, RuntimeError, OSError):
        checks.append(("CUDA runtime", False))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "artifacts/assets")
    parser.add_argument("--models", action="store_true", help="Verify downloaded model and adapter checksums")
    parser.add_argument("--training", action="store_true", help="Also check CUDA and all pinned training dependencies")
    parser.add_argument("--verl-source", type=Path, default=ROOT / "external/verl")
    args = parser.parse_args()
    checks = check_assets(args.assets.resolve(), args.models or args.training)
    if args.training:
        checks += check_training_environment(args.verl_source.resolve())
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {label}")
    passed = all(ok for _, ok in checks)
    print(json.dumps({"status": "pass" if passed else "fail", "checks": len(checks),
                      "training_environment_checked": args.training}))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
