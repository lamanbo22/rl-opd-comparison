"""Reconstruct the study splits and fetch pinned model assets."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_bytes())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_identical(path, content):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == content, f"Existing file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def download_checked(url, destination, expected, limit):
    destination = Path(destination)
    if destination.exists():
        require(sha256(destination) == expected, f"Checksum mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream, urllib.request.urlopen(url, timeout=60) as response:
            size = 0
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                require(size <= limit, f"Download exceeds expected size limit: {url}")
                stream.write(block)
        require(sha256(temporary) == expected, f"Downloaded checksum mismatch: {url}")
        if destination.exists():
            require(sha256(destination) == expected, f"Existing file differs: {destination}")
        else:
            os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def question_id(question):
    return hashlib.sha256(" ".join(question.split()).casefold().encode()).hexdigest()


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def read_source(content):
    # splitlines() also splits Unicode line separators inside JSON strings.
    return [json.loads(line) for line in io.StringIO(content.decode("utf-8")) if line.strip()]


def derive_splits(manifest, raw, parse_gold, canonical):
    dataset = manifest["dataset"]
    source = {}
    for name, spec in dataset["raw"].items():
        require(hashlib.sha256(raw[name]).hexdigest() == spec["sha256"], f"Raw {name} checksum differs")
        source[name] = read_source(raw[name])
        require(len(source[name]) == spec["rows"], f"Raw {name} row count differs")
    split_ids, split_rows = {}, {}
    for name, spec in dataset["splits"].items():
        references = spec["source_rows"]
        if references == "all":
            references = list(range(len(source[spec["source"]])))
        require(len(references) == len(set(references)) == spec["rows"], f"Invalid {name} row indices")
        rows = []
        for index in references:
            require(type(index) is int and 0 <= index < len(source[spec["source"]]), f"Invalid {name} source row")
            original = source[spec["source"]][index]
            gold = parse_gold(original["answer"])
            require(gold is not None, f"Invalid gold answer: {name} row {index}")
            row = {"source_row": index, "question_sha256": question_id(original["question"]),
                   "question": original["question"], "gold": canonical(gold)}
            if name == "test":
                row = {"source_split": "test", **row}
            rows.append(row)
        require(hashlib.sha256(jsonl_bytes(rows)).hexdigest() == spec["sha256"], f"Derived {name} checksum differs")
        split_rows[name] = rows
        split_ids[name] = {row["question_sha256"] for row in rows}
        require(len(split_ids[name]) == len(rows), f"Duplicate {name} question")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        require(not split_ids[left] & split_ids[right], f"{left}/{right} overlap")
    require(set(dataset["splits"]["train"]["source_rows"]) | set(dataset["splits"]["dev"]["source_rows"])
            == set(range(dataset["raw"]["train"]["rows"])), "Train/dev do not cover the source split")
    return split_rows


def load_reward():
    config = read_json(ROOT / "experiments/configs/pilot.json")
    for name, checksum in config["reward"]["code_sha256"].items():
        require(sha256(ROOT / "experiments/reward" / name) == checksum, f"Frozen reward changed: {name}")
    spec = importlib.util.spec_from_file_location("study_reward", ROOT / "experiments/reward/text_reward.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_data(manifest, destination, train_source=None, test_source=None):
    raw = {}
    for name, local in (("train", train_source), ("test", test_source)):
        spec = manifest["dataset"]["raw"][name]
        path = destination / "raw" / f"{name}.jsonl"
        if local:
            content = Path(local).read_bytes()
            require(hashlib.sha256(content).hexdigest() == spec["sha256"], f"Local {name} checksum differs")
            save_identical(path, content)
        else:
            download_checked(spec["url"], path, spec["sha256"], 10 * 1024 * 1024)
        raw[name] = path.read_bytes()
    reward = load_reward()
    splits = derive_splits(manifest, raw, reward.parse_gold, reward.canonical)
    for name, rows in splits.items():
        save_identical(destination / "data" / f"{name}.jsonl", jsonl_bytes(rows))
    return {name: {"rows": len(rows), "sha256": manifest["dataset"]["splits"][name]["sha256"]}
            for name, rows in splits.items()}


def prepare_models(manifest, destination):
    from huggingface_hub import snapshot_download

    for role, spec in manifest["models"].items():
        path = destination / "models" / role / spec["revision"]
        snapshot_download(repo_id=spec["id"], revision=spec["revision"], local_dir=path,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
        for name, expected in spec["files_sha256"].items():
            require(sha256(path / name) == expected, f"Model checksum differs: {role}/{name}")


def prepare_adapter(manifest, destination, source=None):
    spec = manifest["initial_adapter"]
    target = destination / "initial_adapter" / "adapter_model.safetensors"
    if source:
        path = Path(source) / target.name
        require(sha256(path) == spec["weights_sha256"], "Initial adapter checksum differs")
        save_identical(target, path.read_bytes())
    else:
        download_checked(spec["url"], target, spec["weights_sha256"], 200 * 1024 * 1024)
    save_identical(target.with_name("adapter_config.json"), (ROOT / spec["config_file"]).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/assets")
    parser.add_argument("--train-source", type=Path, help="Use the pinned raw training file without downloading it")
    parser.add_argument("--test-source", type=Path, help="Use the pinned raw test file without downloading it")
    parser.add_argument("--models", action="store_true", help="Download the pinned student and teacher weights")
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument("--download-adapter", action="store_true")
    adapters.add_argument("--adapter-dir", type=Path, help="Import the exact initial adapter from a local directory")
    args = parser.parse_args()
    manifest = read_json(ROOT / "configs/assets.json")
    destination = args.output.resolve()
    splits = prepare_data(manifest, destination, args.train_source, args.test_source)
    if args.models:
        prepare_models(manifest, destination)
    if args.download_adapter or args.adapter_dir:
        prepare_adapter(manifest, destination, args.adapter_dir)
    receipt = {"schema_version": 1, "asset_manifest_sha256": sha256(ROOT / "configs/assets.json"),
               "splits": splits, "models": {role: str(destination / "models" / role / spec["revision"])
                                            for role, spec in manifest["models"].items()},
               "initial_adapter": str(destination / "initial_adapter")}
    save_identical(destination / "manifest.json", (json.dumps(receipt, indent=2) + "\n").encode())
    print(json.dumps({"status": "prepared", "output": str(destination), "splits": splits,
                      "models_downloaded": args.models,
                      "adapter_prepared": bool(args.download_adapter or args.adapter_dir)}, indent=2))


if __name__ == "__main__":
    main()
