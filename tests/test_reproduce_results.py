"""Check public results, pairing, and malformed input handling."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reproduce_results', ROOT / 'scripts/reproduce_results.py')
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


class PublicResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.outcomes = reproduce.read_inputs(ROOT / 'results')

    def test_all_tables_match_the_audited_study(self):
        tables, digest = reproduce.build_tables(self.manifest, self.outcomes)
        self.assertEqual(digest, '2e7d8aab4ff57d467c80e03b6eb9b1068f05600699c18bfbeb67bf85279d0643')
        for name, rows in tables.items():
            with self.subTest(table=name):
                reproduce.check_table(ROOT / 'results' / name, rows)

    def test_shared_draws_preserve_identical_predictions(self):
        baseline = np.array([1, 0, 0, 1, 1, 0, 1], dtype=np.int8)
        predictions = np.broadcast_to(baseline, (2, 3, len(baseline)))
        scores, base_scores = reproduce.crossed_bootstrap(predictions, baseline, replicates=100)
        np.testing.assert_array_equal(scores[:, 0], scores[:, 1])
        np.testing.assert_array_equal(scores[:, 0], base_scores)

    def test_chunk_size_does_not_change_draws(self):
        predictions = np.random.default_rng(4).integers(0, 2, size=(2, 3, 19))
        baseline = predictions[0, 0]
        first = reproduce.crossed_bootstrap(predictions, baseline, replicates=101, chunk_size=7)
        second = reproduce.crossed_bootstrap(predictions, baseline, replicates=101, chunk_size=32)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_invalid_predictions_fail(self):
        with self.assertRaisesRegex(ValueError, 'binary'):
            reproduce.crossed_bootstrap(np.full((2, 3, 4), 2), np.zeros(4))
        with self.assertRaisesRegex(ValueError, 'method-by-seed'):
            reproduce.crossed_bootstrap(np.zeros((2, 3, 4)), np.zeros(3))

    def test_source_results_cannot_be_an_output_directory(self):
        for output in [ROOT / 'results', ROOT / 'results/new']:
            with self.assertRaisesRegex(ValueError, 'outside the source results'):
                reproduce.validate_output(output, ROOT / 'results')

    def test_cost_ledger_retains_the_failed_attempt_once(self):
        costs = self.manifest['costs']
        failed = [r for r in costs['attempts'] if r['excluded_from_accuracy']]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]['job_id'], '107_eval_verl_study_rl_seed44_01_test')
        self.assertEqual(failed[0]['number'], 1)
        self.assertEqual(failed[0]['status'], 'exited')
        same_job = [r for r in costs['attempts'] if r['job_id'] == failed[0]['job_id']]
        self.assertEqual([r['number'] for r in same_job], [1, 2])
        self.assertEqual(same_job[1]['status'], 'complete')
        for kind, totals in costs['known_attempt_totals_by_kind'].items():
            for field, expected in totals.items():
                observed = sum(r[field] for r in costs['attempts'] if r['kind'] == kind and r[field] is not None)
                self.assertAlmostEqual(observed, expected, places=7)


class InvalidExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        for name in ['analysis_manifest.json', 'question_outcomes.csv']:
            shutil.copyfile(ROOT / 'results' / name, self.directory / name)

    def edit_manifest(self, edit):
        path = self.directory / 'analysis_manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        edit(manifest)
        path.write_text(json.dumps(manifest), encoding='utf-8')

    def edit_rows(self, edit):
        path = self.directory / 'question_outcomes.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        edit(rows)
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        self.edit_manifest(lambda m: m.update(question_outcomes_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_changed_outcome_fails_even_with_updated_file_hash(self):
        self.edit_rows(lambda rows: rows[0].update(rl_seed42=str(1 - int(rows[0]['rl_seed42']))))
        with self.assertRaisesRegex(ValueError, 'Correctness vector differs'):
            reproduce.read_inputs(self.directory)

    def test_changed_pairing_fails_even_with_updated_file_hash(self):
        def swap(rows):
            rows[0]['question_sha256'], rows[1]['question_sha256'] = rows[1]['question_sha256'], rows[0]['question_sha256']
        self.edit_rows(swap)
        with self.assertRaisesRegex(ValueError, 'Question pairing differs'):
            reproduce.read_inputs(self.directory)

    def test_duplicate_evaluation_fails(self):
        self.edit_manifest(lambda m: m['evaluations'].append(m['evaluations'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicated evaluations'):
            reproduce.read_inputs(self.directory)

    def test_missing_question_fails(self):
        self.edit_rows(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, 'Missing or extra outcome rows'):
            reproduce.read_inputs(self.directory)

    def test_missing_evaluation_fails(self):
        self.edit_manifest(lambda m: m['evaluations'].pop())
        with self.assertRaisesRegex(ValueError, 'Missing or duplicated evaluations'):
            reproduce.read_inputs(self.directory)

    def test_nonbinary_outcome_fails(self):
        self.edit_rows(lambda rows: rows[0].update(rl_seed42='2'))
        with self.assertRaisesRegex(ValueError, 'binary'):
            reproduce.read_inputs(self.directory)

    def test_changed_released_table_fails(self):
        path = self.directory / 'table.csv'
        path.write_text('method,accuracy\nrl,79.0\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Table mismatch'):
            reproduce.check_table(path, [{'method': 'rl', 'accuracy': 80.0}])


if __name__ == '__main__':
    unittest.main()
