import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_assets as assets
import run_reproduction
import evaluate_adapter


class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.raw = {
            "train": assets.jsonl_bytes([{"question": "First?", "answer": "1"},
                                         {"question": "Second\u2028question?", "answer": "2"}]),
            "test": assets.jsonl_bytes([{"question": "Third?", "answer": "3"}]),
        }
        self.manifest = {"dataset": {"raw": {}, "splits": {}}}
        for name, content in self.raw.items():
            self.manifest["dataset"]["raw"][name] = {
                "sha256": hashlib.sha256(content).hexdigest(), "rows": 2 if name == "train" else 1}
        for name, source, indices in (("train", "train", [1]), ("dev", "train", [0]), ("test", "test", "all")):
            originals = assets.read_source(self.raw[source])
            rows = []
            for index in range(len(originals)) if indices == "all" else indices:
                original = originals[index]
                row = {"source_row": index, "question_sha256": assets.question_id(original["question"]),
                       "question": original["question"], "gold": original["answer"]}
                rows.append({"source_split": "test", **row} if name == "test" else row)
            self.manifest["dataset"]["splits"][name] = {
                "source": source, "source_rows": indices, "rows": 1,
                "sha256": hashlib.sha256(assets.jsonl_bytes(rows)).hexdigest()}

    def derive(self):
        return assets.derive_splits(self.manifest, self.raw, int, str)

    def test_exact_order_and_unicode_separator(self):
        result = self.derive()
        self.assertEqual(result["train"][0]["source_row"], 1)
        self.assertEqual(result["train"][0]["question"], "Second\u2028question?")
        self.assertEqual(result["dev"][0]["gold"], "1")
        self.assertEqual(result["test"][0]["source_split"], "test")

    def test_changed_raw_bytes_rejected(self):
        self.raw["train"] += b"\n"
        with self.assertRaisesRegex(ValueError, "Raw train checksum"):
            self.derive()

    def test_changed_split_order_rejected(self):
        self.manifest["dataset"]["splits"]["train"]["source_rows"] = [0]
        with self.assertRaisesRegex(ValueError, "Derived train checksum"):
            self.derive()

    def test_invalid_indices_rejected(self):
        self.manifest["dataset"]["splits"]["train"]["source_rows"] = [-1]
        with self.assertRaisesRegex(ValueError, "Invalid train source row"):
            self.derive()

    def test_existing_different_file_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            assets.save_identical(path, b"original")
            assets.save_identical(path, b"original")
            with self.assertRaisesRegex(ValueError, "Existing file differs"):
                assets.save_identical(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")

    def test_pinned_indices_partition_training_source(self):
        manifest = assets.read_json(ROOT / "configs/assets.json")
        train = manifest["dataset"]["splits"]["train"]["source_rows"]
        dev = manifest["dataset"]["splits"]["dev"]["source_rows"]
        self.assertEqual(len(train), 6961)
        self.assertEqual(len(dev), 512)
        self.assertFalse(set(train) & set(dev))
        self.assertEqual(set(train) | set(dev), set(range(7473)))

    def test_official_test_cannot_use_a_subset(self):
        with self.assertRaisesRegex(ValueError, "only for development"):
            evaluate_adapter.select_rows([{}, {}], "test", 1)
        self.assertEqual(evaluate_adapter.select_rows([1, 2], "dev", 1), [1])
        self.assertEqual(evaluate_adapter.select_rows([1, 2], "test", None), [1, 2])

    def test_evaluation_eos_and_truncation(self):
        self.assertEqual(evaluate_adapter.completion_tokens([2, 3, 99, 0, 0], {99}), ([2, 3, 99], True, False))
        tokens, ended, truncated = evaluate_adapter.completion_tokens([2] * 1024, {99})
        self.assertEqual(len(tokens), 1024)
        self.assertFalse(ended)
        self.assertTrue(truncated)

    def test_runtime_changes_only_paths_and_hash_links(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            asset_dir = base / "assets"
            (asset_dir / "data").mkdir(parents=True)
            for name in ("train", "dev", "test"):
                (asset_dir / "data" / f"{name}.jsonl").write_text("{}\n")
            workspace = base / "new_run"
            with patch.object(run_reproduction, "check_assets", return_value=[("fixtures", True)]):
                result = run_reproduction.materialize(workspace, asset_dir)
            self.assertFalse(result["formal_queue_recreated"])
            run_reproduction.verify_workspace(workspace)
            original = assets.read_json(ROOT / "experiments/verl_study/protocol.json")
            runtime = assets.read_json(workspace / "experiments/verl_study/protocol.json")
            for name in ("methods", "seeds", "budget_seconds", "reserve_seconds", "run_order", "statistical_analysis"):
                self.assertEqual(original[name], runtime[name])
            for relative, expected in result["python_source_sha256"].items():
                self.assertEqual(assets.sha256(ROOT / "experiments" / relative), expected)
            with self.assertRaisesRegex(ValueError, "Workspace already exists"):
                run_reproduction.materialize(workspace, asset_dir)


if __name__ == "__main__":
    unittest.main()
