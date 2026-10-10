"""Evaluate a reproduction adapter with the study's prompt, decoding and reward."""
import argparse
import importlib.metadata
import importlib.util
import json
import sys
import time
from pathlib import Path

from prepare_assets import ROOT, read_json, read_source, require, sha256
from run_reproduction import verify_workspace, write_json


def select_rows(rows, split, limit):
    require(split in {"dev", "test"}, "Expected dev or test split")
    require(limit is None or split == "dev", "A question limit is allowed only for development checks")
    require(limit is None or 0 < limit <= len(rows), "Question limit is out of range")
    return rows if limit is None else rows[:limit]


def completion_tokens(sequence, eos_ids):
    stop = next((index for index, token in enumerate(sequence) if token in eos_ids), None)
    tokens = sequence if stop is None else sequence[:stop + 1]
    require(bool(tokens), "Empty generated response")
    return tokens, stop is not None, stop is None and len(tokens) >= 1024


def prepare(args):
    workspace = args.workspace.resolve()
    manifest = verify_workspace(workspace)
    assets = Path(manifest["assets"])
    asset_spec = read_json(ROOT / "configs/assets.json")
    experiment = workspace / "experiments"
    config = read_json(experiment / "configs/pilot.json")
    path = experiment / "data" / f"{args.split}.jsonl"
    split_spec = asset_spec["dataset"]["splits"][args.split]
    require(sha256(path) == split_spec["sha256"], "Evaluation split checksum differs")
    rows = read_source(path.read_bytes())
    require(len(rows) == split_spec["rows"], "Evaluation split length differs")
    rows = select_rows(rows, args.split, args.limit)
    snapshot = Path(config["models"]["student"]["local_snapshot"])
    for name, expected in asset_spec["models"]["student"]["files_sha256"].items():
        require(sha256(snapshot / name) == expected, f"Student model file differs: {name}")
    for name, expected in config["reward"]["code_sha256"].items():
        require(sha256(experiment / "reward" / name) == expected, f"Reward code differs: {name}")
    adapter = args.adapter.resolve() if args.adapter else assets / "initial_adapter"
    weights = adapter / "adapter_model.safetensors"
    adapter_hash = sha256(weights)
    if args.adapter is None:
        require(adapter_hash == asset_spec["initial_adapter"]["weights_sha256"], "Initial adapter checksum differs")
    adapter_config = read_json(adapter / "adapter_config.json")
    for name in ("r", "lora_alpha", "lora_dropout"):
        require(adapter_config[name] == config["lora"][name], f"Adapter {name} differs")
    require(set(adapter_config["target_modules"]) == set(config["lora"]["target_modules"]), "Adapter target modules differ")
    identity = {"split": args.split, "questions": len(rows), "development_subset": args.limit,
                "data_sha256": split_spec["sha256"], "batch_size": 4, "seed": 42,
                "max_new_tokens": 1024, "enable_thinking": False, "do_sample": False,
                "adapter": str(adapter), "adapter_sha256": adapter_hash,
                "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
                "initial_adapter": args.adapter is None,
                "model": config["models"]["student"],
                "model_files_sha256": asset_spec["models"]["student"]["files_sha256"],
                "config_sha256": sha256(experiment / "configs/pilot.json"),
                "runtime_manifest_sha256": sha256(workspace / "reproduction_manifest.json"),
                "evaluator_sha256": sha256(Path(__file__)),
                "scope": "Independent reproduction evaluation; excluded from the original study results."}
    return workspace, config, rows, snapshot, adapter, identity


def run(args, prepared):
    workspace, config, rows, snapshot, adapter, identity = prepared
    protocol = read_json(workspace / "experiments/verl_study/protocol.json")
    versions = {name: importlib.metadata.version(name) for name in protocol["frozen_versions"]}
    require(versions == protocol["frozen_versions"], "Use the pinned runtime environment")
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    require(torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0),
            "Expected one visible RTX 5090")
    helper_path = workspace / "experiments/verl_study/evaluate_verl_formal.py"
    spec = importlib.util.spec_from_file_location("frozen_evaluator", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sys.path.insert(0, str(workspace / "experiments/reward"))
    from text_answer_reward import score_answer

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity.update(versions=versions, gpu=torch.cuda.get_device_name(0),
                    frozen_helpers_sha256=sha256(helper_path), prompt_instruction=config["prompt_instruction"])
    write_json(output / "metadata.json", identity)
    started = time.perf_counter()
    generation_seconds = 0.0
    records = []
    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                  trust_remote_code=False, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
                                                    dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                                    attn_implementation="sdpa").eval()
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True,
                                         is_trainable=False, autocast_adapter_dtype=True).eval()
        saved = load_peft_weights(adapter, device="cpu", local_files_only=True)
        actual = get_peft_model_state_dict(model, save_embedding_layers=False)
        require(saved.keys() == actual.keys() and all(torch.equal(saved[key], actual[key].detach().cpu()) for key in saved),
                "Loaded adapter values differ")
        del saved, actual
        model.requires_grad_(False)
        generation = helper.resolved_generation(model, tokenizer)
        eos = generation.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        initial_hash = helper.parameter_hash(model)
        versions_before = helper.parameter_versions(model)
        prompts = [helper.prompt(tokenizer, config, row) for row in rows]
        identity["generation"] = generation.to_dict()
        write_json(output / "metadata.json", identity)
        with (output / "responses.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for begin in range(0, len(rows), 4):
                batch = rows[begin:begin + 4]
                inputs = tokenizer(prompts[begin:begin + 4], add_special_tokens=False,
                                   padding=True, return_tensors="pt").to("cuda:0")
                width = inputs.input_ids.shape[1]
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=generation, **helper.OPTIONS)
                torch.cuda.synchronize()
                generation_seconds += time.perf_counter() - tick
                for row, sequence in zip(batch, generated[:, width:].cpu().tolist()):
                    tokens, terminated, truncated = completion_tokens(sequence, eos_ids)
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    record = {**row, **score_answer(text, row["gold"]), "completion": text,
                              "token_ids": tokens, "eos_terminated": terminated, "truncated": truncated}
                    records.append(record)
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                require(helper.parameter_versions(model) == versions_before, "Evaluation weights changed")
                del inputs, generated
                print(f"Evaluated {len(records)}/{len(rows)} questions", flush=True)
        final_hash = helper.parameter_hash(model)
        require(final_hash == initial_hash, "Evaluation parameter hash changed")
        correct = sum(record["correct"] for record in records)
        summary = {"status": "complete", "split": args.split, "questions": len(records), "correct": correct,
                   "accuracy": correct / len(records), "invalid_answers": sum(not record["answer_valid"] for record in records),
                   "truncated_answers": sum(record["truncated"] for record in records),
                   "generation_seconds": generation_seconds, "evaluation_wall_seconds": time.perf_counter() - started,
                   "peak_torch_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                   "parameter_sha256": final_hash, "responses_sha256": sha256(output / "responses.jsonl"),
                   "optimizer_updates": 0, "development_subset": args.limit}
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    except BaseException as error:
        write_json(output / "summary.json", {"status": "incomplete", "completed_questions": len(records),
                   "evaluation_wall_seconds": time.perf_counter() - started, "generation_seconds": generation_seconds,
                   "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "artifacts/reproduction")
    parser.add_argument("--adapter", type=Path, help="Defaults to the exact shared initial adapter")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--limit", type=int, help="First N development questions for an engineering check")
    parser.add_argument("--output", type=Path, help="New evaluation directory; required for inference")
    parser.add_argument("--check-only", action="store_true", help="Check inputs without loading a model or accessing CUDA")
    args = parser.parse_args()
    require(args.check_only or args.output is not None, "--output is required for inference")
    prepared = prepare(args)
    if args.check_only:
        print(json.dumps({"status": "check_pass", **prepared[-1]}, indent=2))
        return
    run(args, prepared)


if __name__ == "__main__":
    main()
