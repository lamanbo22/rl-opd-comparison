"""Check release checksums and the original study's frozen source files."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check(relative, expected):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Missing release file: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {relative}")


def main():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        check(item["path"], item["sha256"])
    study = ROOT / "experiments/verl_study"
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    execution = json.loads((study / "execution_manifest.json").read_text(encoding="utf-8"))
    for mapping in [protocol["frozen_source_sha256"], execution["source_sha256"]]:
        for relative, expected in mapping.items():
            check(Path("experiments") / relative, expected)
    for name, expected in execution["test_source_sha256"].items():
        if (study / name).is_file():
            check(Path("experiments/verl_study") / name, expected)
    print(f"PASS: {len(manifest['files'])} release files and all frozen study sources match.")


if __name__ == "__main__":
    main()
