"""Reproduce the published tables from question outcomes on a CPU."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ['rl', 'opd', 'joint', 'opd_then_rl', 'rl_then_opd']
SEEDS = [42, 43, 44]
SPLITS = {'dev64': 64, 'dev512': 512, 'test': 1319}
COLUMNS = ['initial_student'] + [f'{method}_seed{seed}' for method in METHODS for seed in SEEDS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_inputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'analysis_manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema_version'] == 1, 'Unsupported analysis manifest version')
    require(manifest['methods'] == METHODS and manifest['seeds'] == SEEDS, 'Method or seed plan differs')
    require(manifest['split_sizes'] == SPLITS, 'Split sizes differ')
    path = directory / 'question_outcomes.csv'
    require(sha256(path) == manifest['question_outcomes_sha256'], 'Question outcomes checksum differs')
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['split', 'question_index', 'source_row', 'question_sha256', *COLUMNS],
                'Unexpected outcome columns')
        rows = list(reader)
    require(len(rows) == sum(SPLITS.values()), 'Missing or extra outcome rows')
    outcomes = {}
    offset = 0
    for split, count in SPLITS.items():
        section = rows[offset:offset + count]
        offset += count
        require(all(r['split'] == split and r['question_index'] == str(i) for i, r in enumerate(section)),
                'Question order differs in ' + split)
        ids = [r['question_sha256'] for r in section]
        require(len(set(ids)) == count, 'Duplicate question in ' + split)
        require(all(len(q) == 64 and all(c in '0123456789abcdef' for c in q) for q in ids),
                'Invalid question identifier')
        require(all(r['source_row'].isdigit() for r in section), 'Invalid source row')
        require(len({r['source_row'] for r in section}) == count, 'Duplicate source row')
        expected_columns = COLUMNS if split == 'test' else COLUMNS[1:]
        if split != 'test':
            require(all(r['initial_student'] == '' for r in section), 'Unexpected development baseline')
        for column in expected_columns:
            values = [r[column] for r in section]
            require(all(v in ('0', '1') for v in values), 'Outcomes must be binary: ' + column)
            outcomes[split, column] = np.array(values, dtype=np.int8)
        expected = {(split, c) for c in expected_columns}
        records = [r for r in manifest['evaluations'] if r['split'] == split]
        require(len(records) == len(expected) and {(split, r['column']) for r in records} == expected,
                'Missing or duplicated evaluations in ' + split)
        for record in records:
            column = ('initial_student' if record['method'] == 'initial_student'
                      else f"{record['method']}_seed{record['seed']}")
            require(record['column'] == column, 'Evaluation label differs: ' + record['name'])
            values = outcomes[split, record['column']]
            require(hashlib.sha256(values.tobytes()).hexdigest() == record['correct_mask_sha256'],
                    'Correctness vector differs: ' + record['name'])
            require(canonical_sha256(ids) == record['ordered_question_ids_sha256'],
                    'Question pairing differs: ' + record['name'])
            require(record['questions'] == count and record['correct'] == int(values.sum()),
                    'Evaluation count differs: ' + record['name'])
            require(all(type(record[k]) is int and record[k] >= 0 for k in
                        ('questions', 'correct', 'invalid_answers', 'truncated_answers', 'generated_tokens')),
                    'Invalid evaluation counts: ' + record['name'])
            require(record['invalid_answers'] <= count and record['truncated_answers'] <= count,
                    'Evaluation flags exceed question count')
    require(len(manifest['evaluations']) == 46, 'Unexpected evaluation count')
    require(len({r['name'] for r in manifest['evaluations']}) == 46, 'Duplicate evaluation name')
    training = manifest['training']
    require(len(training) == 15 and {(r['method'], r['seed']) for r in training} ==
            set(itertools.product(METHODS, SEEDS)), 'Missing or duplicated training runs')
    return manifest, outcomes


def crossed_bootstrap(predictions, baseline, replicates=20000, seed=20261003, chunk_size=128):
    """Use the same seed and question draws for every strategy."""
    data, base = np.asarray(predictions), np.asarray(baseline)
    require(data.ndim == 3 and base.shape == (data.shape[2],) and data.shape[1] >= 2,
            'Expected method-by-seed-by-question outcomes and one baseline vector')
    require(np.isin(data, [0, 1]).all() and np.isin(base, [0, 1]).all(), 'Outcomes must be binary')
    require(replicates > 0 and chunk_size > 0 and data.shape[2] > 0, 'Invalid bootstrap size')
    data, base = data.astype(np.int8), base.astype(np.int8)
    method_count, seed_count, question_count = data.shape
    rng = np.random.default_rng(seed)
    # Drawing seeds first preserves the original study's random-number sequence.
    seed_draws = rng.integers(0, seed_count, size=(replicates, seed_count))
    scores = np.empty((replicates, method_count))
    baselines = np.empty(replicates)
    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        questions = rng.integers(0, question_count, size=(end - start, question_count))
        sampled = data[:, seed_draws[start:end, :, None], questions[:, None, :]]
        scores[start:end] = sampled.mean(axis=(2, 3)).T * 100
        baselines[start:end] = base[questions].mean(axis=1) * 100
    return scores, baselines


def statistical_tables(outcomes, plan):
    require(plan['replicates'] == 20000 and plan['seed'] == 20261003 and
            plan['quantile_method'] == 'linear', 'Bootstrap plan differs')
    data = np.stack([[outcomes['test', f'{method}_seed{seed}'] for seed in SEEDS] for method in METHODS])
    base = outcomes['test', 'initial_student']
    scores, bootbase = crossed_bootstrap(data, base, plan['replicates'], plan['seed'])
    digest = hashlib.sha256(scores.astype('<f8').tobytes() + bootbase.astype('<f8').tobytes()).hexdigest()
    require(digest == plan['score_array_sha256'], 'Bootstrap draws differ from the audited analysis')
    counts = data.sum(axis=2)
    accuracy = counts / data.shape[2] * 100
    means = counts.sum(axis=1) / (len(SEEDS) * data.shape[2]) * 100
    baseline_accuracy = float(base.sum() / len(base) * 100)
    methods = []
    for i, method in enumerate(METHODS):
        row = dict(method=method, mean_accuracy_percent=float(means[i]),
                   sample_sd_percentage_points=float(accuracy[i].std(ddof=1)),
                   minimum_accuracy_percent=float(accuracy[i].min()),
                   maximum_accuracy_percent=float(accuracy[i].max()))
        for j, seed in enumerate(SEEDS):
            row[f'seed{seed}_correct'] = int(counts[i, j])
            row[f'seed{seed}_accuracy_percent'] = float(accuracy[i, j])
        methods.append(row)
    contrasts = []
    for i, j in itertools.combinations(range(len(METHODS)), 2):
        low, high = np.quantile(scores[:, i] - scores[:, j], [.025, .975], method='linear')
        contrasts.append(dict(left=METHODS[i], right=METHODS[j], difference_pp=float(means[i] - means[j]),
                              ci_low_pp=float(low), ci_high_pp=float(high)))
    for i, method in enumerate(METHODS):
        low, high = np.quantile(scores[:, i] - bootbase, [.025, .975], method='linear')
        contrasts.append(dict(left=method, right='initial_student',
                              difference_pp=float(means[i] - baseline_accuracy),
                              ci_low_pp=float(low), ci_high_pp=float(high)))
    return methods, contrasts, digest


def cost_rows(manifest):
    """Export nested timers with their containing clock to avoid double counting."""
    rows = []
    def add(scope, name, method, seed, metric, seconds, contained_in):
        require(seconds is None or (math.isfinite(seconds) and seconds >= 0), 'Invalid cost: ' + metric)
        rows.append(dict(scope=scope, name=name, method=method, seed=seed, metric=metric,
                         seconds='' if seconds is None else seconds, contained_in=contained_in))
    for record in manifest['training']:
        cost = record['costs']
        fixed = {
            'charged_seconds': 'session_wall_seconds',
            'unused_allowance_seconds': 'budget_allowance_seconds',
            'training_core_seconds': 'charged_seconds',
            'within_charge_audit_seconds': 'charged_seconds',
            'within_charge_checkpoint_payload_seconds': 'charged_seconds',
            'uncharged_commit_tail_seconds': 'session_wall_seconds',
            'independent_audit_internal_seconds': 'train_audit.process_wall_seconds',
        }
        values = [(k, cost[k], parent) for k, parent in fixed.items()]
        values += [(k, v, 'training_core_seconds') for k, v in cost['core_components'].items()]
        values += [(k, v, 'train.process_wall_seconds' if k == 'session_wall_seconds' else 'session_wall_seconds')
                   for k, v in cost['session_costs'].items()]
        values += [('phase_' + k + '_seconds', v, 'charged_seconds') for k, v in cost['phase_seconds'].items()]
        for metric, seconds, parent in values:
            add('training', record['name'], record['method'], record['seed'], metric, seconds, parent)
    for record in manifest['evaluations']:
        for metric, seconds in record['costs'].items():
            parent = 'eval.process_wall_seconds' if metric in (
                'known_session_wall_seconds', 'preparation_seconds', 'recovery_validation_seconds') else 'known_session_wall_seconds'
            add('evaluation', record['name'], record['method'], record['seed'], metric, seconds, parent)
    for attempt in manifest['costs']['attempts']:
        for metric in ['process_wall_seconds', 'whole_job_wall_seconds',
                       'supervisor_preparation_seconds', 'postprocess_verification_seconds']:
            parent = 'supervisor_session_wall_seconds' if metric == 'whole_job_wall_seconds' else 'whole_job_wall_seconds'
            add(attempt['kind'] + '_attempt', attempt['job_id'] + '_attempt' + str(attempt['number']),
                '', '', metric, attempt[metric], parent)
    return rows


def build_tables(manifest, outcomes):
    methods, contrasts, digest = statistical_tables(outcomes, manifest['bootstrap'])
    evaluations = []
    for record in manifest['evaluations']:
        values = outcomes[record['split'], record['column']]
        evaluations.append(dict(name=record['name'], split=record['split'], correct=int(values.sum()),
            questions=len(values), accuracy_percent=100 * int(values.sum()) / len(values),
            invalid_answers=record['invalid_answers'], truncated_answers=record['truncated_answers'],
            generated_tokens=record['generated_tokens'],
            generation_tokens_per_second=record['generated_tokens'] / record['costs']['generation_seconds']))
    training = []
    for record in manifest['training']:
        cost = record['costs']
        training.append(dict(method=record['method'], seed=record['seed'], charged_seconds=cost['charged_seconds'],
            unused_seconds=cost['unused_allowance_seconds'], session_wall_seconds=cost['session_costs']['session_wall_seconds'],
            updates=record['counts']['actual_optimizer_steps'], rounds=record['counts']['iterations'],
            rl_fraction=cost['phase_fractions'].get('rl', 0), opd_fraction=cost['phase_fractions'].get('opd', 0),
            joint_fraction=cost['phase_fractions'].get('joint', 0)))
    return {'method_results.csv': methods, 'paired_contrasts.csv': contrasts,
            'evaluation_results.csv': evaluations, 'training_costs.csv': training,
            'cost_breakdown.csv': cost_rows(manifest)}, digest


def check_table(path, expected):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == list(expected[0]), 'Table columns differ: ' + str(path))
        actual = list(reader)
    require(len(actual) == len(expected), 'Table row count differs: ' + str(path))
    for index, (left, right) in enumerate(zip(actual, expected)):
        for key, value in right.items():
            if isinstance(value, (int, float)):
                try:
                    matches = math.isclose(float(left[key]), value, rel_tol=1e-12, abs_tol=1e-10)
                except ValueError:
                    matches = False
            else:
                matches = left[key] == ('' if value is None else str(value))
            require(matches, f'Table mismatch: {path}, row {index + 1}, {key}')


def write_tables(tables, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with (directory / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def validate_output(output, results):
    output = Path(output).resolve()
    for source in [Path(results).resolve(), (ROOT / 'results').resolve()]:
        require(not output.is_relative_to(source), 'Use an output directory outside the source results')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results', help='Released outcomes and analysis manifest')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/reproduced', help='Directory for regenerated tables')
    parser.add_argument('--check', action='store_true', help='Compare regenerated values with the released CSV tables')
    args = parser.parse_args()
    validate_output(args.output, args.results)
    manifest, outcomes = read_inputs(args.results)
    tables, digest = build_tables(manifest, outcomes)
    if args.check:
        for name, rows in tables.items():
            check_table(args.results / name, rows)
    write_tables(tables, args.output)
    print('PASS: 46 evaluations and 20,000 paired bootstrap resamples match the audited study.')
    if args.check:
        print('PASS: all five released tables match.')
    print('Bootstrap SHA256:', digest)
    print('Tables:', args.output.resolve())


if __name__ == '__main__':
    main()
