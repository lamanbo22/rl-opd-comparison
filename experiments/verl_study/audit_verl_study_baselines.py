"""Independent CPU audit of the two complete 512-question study baselines.

This module does not import the evaluator or load models. It independently
checks provenance, decodes and regrades records, reconstructs aggregates, and
checks clean session closure. The optional CLI output is the only file written.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path


EVALUATOR_SHA256 = 'c1102f7db9603371c5994516db92eb81bd4be9ddf147ef2e0f38805a74426347'
RUN_NAMES = {'student': 'verl_study_student_dev_01', 'teacher': 'verl_study_teacher_dev_01'}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def manifest(directory, suffixes=None):
    return {path.name: sha(path) for path in sorted(Path(directory).iterdir())
            if path.is_file() and (suffixes is None or path.suffix in suffixes)}


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        check(math.isfinite(value), 'Non-finite recorded numeric value')


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def adapter_state_hash(path):
    import torch
    from safetensors.torch import load_file
    value = hashlib.sha256()
    state = load_file(str(path), device='cpu')
    for name in sorted(state):
        tensor = state[name].contiguous()
        check(tensor.dtype == torch.float32, 'Initial adapter storage is not FP32')
        value.update(f'{name}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        value.update(memoryview(tensor.view(torch.uint8).numpy()))
    return value.hexdigest()


def audit_baselines(project: Path) -> dict:
    project = Path(project).resolve()
    experiment = project / 'experiments'
    runs = {role: experiment / 'runs' / name for role, name in RUN_NAMES.items()}
    # Do not inspect partial evaluations as if they were final results.
    summaries = {role: read(run / 'summary.json') for role, run in runs.items()}
    check(all(item['status'] == 'complete' for item in summaries.values()),
          'Both baseline evaluations must be complete before this audit runs')
    config_path = experiment / 'configs/pilot.json'
    protocol_path = experiment / 'verl/protocol.json'
    config, protocol = read(config_path), read(protocol_path)
    evaluator = experiment / 'verl_study/evaluate_verl_study.py'
    check(sha(evaluator) == EVALUATOR_SHA256, 'Evaluator differs from the reviewed frozen source')
    check(sha(config_path) == protocol['source_config_sha256'], 'Pilot config/protocol mismatch')
    check(config['reward']['version'] == 3, 'Wrong reward version')
    datasets = {}
    for split, count in [('train', 6961), ('dev', 512)]:
        path = experiment / config['data'][split + '_file']
        check(sha(path) == config['data'][split + '_sha256'], 'Frozen ' + split + ' bytes changed')
        datasets[split] = rows(path)
        check(len(datasets[split]) == count, split + ' count differs')
        ids = [row['question_sha256'] for row in datasets[split]]
        check(len(set(ids)) == count, 'Duplicated ' + split + ' question')
        for row in datasets[split]:
            normalized = ' '.join(row['question'].split()).casefold()
            check(hashlib.sha256(normalized.encode()).hexdigest() == row['question_sha256'],
                  'Question ID does not match normalized source')
    check(not ({r['question_sha256'] for r in datasets['train']} &
               {r['question_sha256'] for r in datasets['dev']}), 'Train/dev overlap')
    for name, expected in config['reward']['code_sha256'].items():
        check(sha(experiment / 'reward' / name) == expected, 'Frozen reward source changed')
    sys.path.insert(0, str(experiment / 'reward'))
    from text_answer_reward import score_answer
    from transformers import AutoTokenizer

    tokenizers = {role: AutoTokenizer.from_pretrained(config['models'][role]['local_snapshot'],
                  local_files_only=True, trust_remote_code=False, padding_side='left')
                  for role in RUN_NAMES}
    prompts = {}
    for role, tokenizer in tokenizers.items():
        text = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': config['prompt_instruction'] + row['question']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for row in datasets['dev']]
        prompts[role] = tokenizer(text, add_special_tokens=False)['input_ids']
    check(tokenizers['student'].get_vocab() == tokenizers['teacher'].get_vocab(), 'Teacher token vocabulary mismatch')
    check(tokenizers['student'].all_special_ids == tokenizers['teacher'].all_special_ids, 'Teacher special-token mismatch')
    check(prompts['student'] == prompts['teacher'], 'Teacher prompt encoding differs')
    baseline = experiment / 'runs/verl_dev_before_01'
    old_metadata = read(baseline / 'metadata.json')
    old_records = rows(baseline / 'responses.jsonl')
    check(len(old_records) == 64, 'Pilot baseline incomplete')
    for index, record in enumerate(old_records):
        check(record['prompt_token_ids'] == prompts['student'][index], 'Pilot prompt token mismatch')
        check(record['question_sha256'] == datasets['dev'][index]['question_sha256'], 'Pilot question order mismatch')

    report, all_records = {}, {}
    for role, run in runs.items():
        metadata, summary = read(run / 'metadata.json'), summaries[role]
        identity = metadata['identity']
        identity_sha = canonical(identity)
        check(metadata['identity_sha256'] == identity_sha, 'Metadata identity checksum differs')
        check(identity['role'] == role and identity['questions'] == 512 and identity['batch_size'] == 4,
              'Unexpected baseline role or exposure')
        check(identity['seed'] == 42 and identity['optimizer_steps'] == 0 and
              identity['enable_thinking'] is False and identity['train_or_test_questions_used'] is False,
              'Unexpected inference protocol')
        check(identity['config_sha256'] == sha(config_path) and
              identity['source_protocol_sha256'] == sha(protocol_path) and
              identity['script_sha256'] == EVALUATOR_SHA256, 'Source identity mismatch')
        check(identity['data'] == config['data'] and identity['reward'] == config['reward'] and
              identity['prompt_instruction'] == config['prompt_instruction'], 'Data/reward/prompt identity mismatch')
        check(identity['source_rows'] == [r['source_row'] for r in datasets['dev']] and
              identity['question_sha256'] == [r['question_sha256'] for r in datasets['dev']], 'Frozen row order mismatch')
        check(identity['prompt_token_ids_sha256'] == canonical(prompts[role]), 'Prompt hash mismatch')
        for package, version in identity['versions'].items():
            check(importlib.metadata.version(package) == version, 'Audit environment differs: ' + package)
        for package, version in old_metadata['versions'].items():
            check(identity['versions'][package] == version, 'Pilot baseline package mismatch')
        check(identity['pilot_baseline_metadata_sha256'] == sha(baseline / 'metadata.json') and
              identity['pilot_baseline_responses_sha256'] == sha(baseline / 'responses.jsonl'), 'Pilot evidence changed')
        check(identity['pilot_first_64_prompt_token_ids_equal'] is True, 'Missing pilot prompt proof')
        compatibility = identity['tokenizer_compatibility']
        check(all(compatibility[key] is True for key in ['vocabulary_and_token_ids_equal',
              'special_token_ids_equal', 'all_512_prompt_token_ids_equal']), 'Tokenizer compatibility claim differs')
        check(compatibility['vocabulary_sha256'] == {r: canonical(t.get_vocab()) for r, t in tokenizers.items()},
              'Tokenizer vocabulary checksums differ')
        check(identity['model'] == config['models'][role], 'Wrong pinned model')
        snapshot = Path(identity['model']['local_snapshot'])
        check(snapshot.name == identity['model']['revision'], 'Snapshot revision differs')
        check(identity['model_files_sha256'] == manifest(snapshot, {'.json', '.safetensors', '.jinja', '.txt', '.model'}),
              'Pinned checkpoint or tokenizer file changed')
        weights = read(run / 'initial_weights.json')
        frozen = weights['all_parameter_sha256']
        check(valid_hash(frozen) and weights['identity_sha256'] == identity_sha, 'Initial weight provenance invalid')
        if role == 'student':
            adapter = Path(protocol['initial_adapter'])
            check(identity['adapter_path'] == str(adapter.resolve()), 'Student is not the initial adapter')
            check(identity['adapter_files_sha256'] == manifest(adapter, {'.json', '.safetensors'}), 'Adapter files changed')
            check(sha(adapter / 'adapter_model.safetensors') == protocol['initial_adapter_file_sha256'], 'Wrong initial adapter')
            check(weights['adapter_state_sha256'] == adapter_state_hash(adapter / 'adapter_model.safetensors'),
                  'Runtime loaded adapter checksum differs')
        else:
            check(identity['adapter_path'] is None and identity['adapter_files_sha256'] is None and
                  weights['adapter_state_sha256'] is None, 'Teacher unexpectedly adapted')
        generation = identity['generation']
        for key, expected in {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 1024, 'use_cache': True}.items():
            check(generation[key] == expected and old_metadata['generation'][key] == expected, 'Greedy options differ')
        tokenizer = tokenizers[role]
        check(generation['pad_token_id'] == tokenizer.pad_token_id, 'Unexpected pad token')
        pretrained_generation = read(snapshot / 'generation_config.json')
        check(generation['eos_token_id'] == pretrained_generation['eos_token_id'], 'EOS policy differs from pinned model')
        eos_value = generation['eos_token_id']
        eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
        sessions = [read(path) for path in sorted((run / 'sessions').glob('*.json'))]
        expected_statuses = ['paused', 'complete'] if role == 'student' else ['complete']
        check([s['status'] for s in sessions] == expected_statuses, 'Unexpected session lifecycle')
        check(len({s['session_id'] for s in sessions}) == len(sessions), 'Duplicate session IDs')
        expected_ranges = [(0, 2, False), (2, 128, True)] if role == 'student' else [(0, 128, False)]
        for session, (begin, end, resumed) in zip(sessions, expected_ranges):
            finite(session)
            check(session['identity_sha256'] == identity_sha and session['optimizer_steps'] == 0, 'Session identity differs')
            check(session['all_parameter_hash_before'] == session['all_parameter_hash_after'] == frozen,
                  'Session lacks exact full-weight closure')
            check(session['start_completed_batches'] == begin and session['end_completed_batches'] == end and
                  session['last_committed_questions'] == 4 * end and session['resumed'] is resumed, 'Session cursor differs')
            check('RTX 5090' in session['gpu_name'] and session['gpu_total_memory_bytes'] > 30 * 2**30, 'Wrong evaluation GPU')
            check(datetime.fromisoformat(session['started_at']) <= datetime.fromisoformat(session['ended_at']),
                  'Session timestamps reversed')
        session_map = {s['session_id']: s for s in sessions}
        batch_paths = sorted((run / 'batches').glob('*.json'))
        check([p.name for p in batch_paths] == [f'{i:04d}.json' for i in range(128)], 'Missing/extra committed batch')
        batch_records, batches, version_by_session = [], [], {}
        vocabulary_size = len(tokenizer)
        for index, path in enumerate(batch_paths):
            batch = read(path)
            finite(batch)
            check(batch['identity_sha256'] == identity_sha and batch['optimizer_steps'] == 0, 'Batch identity differs')
            check(batch['begin'] == 4 * index and batch['end'] == 4 * (index + 1), 'Batch range differs')
            check(batch['records_sha256'] == canonical(batch['records']) and len(batch['records']) == 4, 'Batch records changed')
            check(batch['session_initial_parameter_hash'] == frozen, 'Batch used a different frozen model')
            version = batch['parameter_versions_before']
            check(valid_hash(version) and batch['parameter_versions_after'] == version, 'Per-batch parameter versions differ')
            sid = batch['session_id']
            check(sid in session_map, 'Batch refers to absent session')
            if sid in version_by_session:
                check(version_by_session[sid] == version, 'Parameter versions changed between batches')
            version_by_session[sid] = version
            session = session_map[sid]
            check(session['start_completed_batches'] <= index < session['end_completed_batches'], 'Batch/session cursor mismatch')
            check(datetime.fromisoformat(session['started_at']) <= datetime.fromisoformat(batch['committed_at']) <=
                  datetime.fromisoformat(session['ended_at']), 'Batch timestamp lies outside session')
            for key in ['generation_seconds', 'weight_audit_seconds', 'peak_torch_allocated_gib']:
                check(batch[key] >= 0, 'Negative batch timing/memory')
            for j, record in enumerate(batch['records'], index * 4):
                source = datasets['dev'][j]
                check(all(record[key] == value for key, value in source.items()), 'Response source row differs')
                check(record['prompt_token_ids'] == prompts[role][j], 'Actual prompt encoding differs')
                tokens = record['token_ids']
                check(isinstance(tokens, list) and 0 < len(tokens) <= 1024 and
                      all(type(t) is int and 0 <= t < vocabulary_size for t in tokens), 'Invalid response token sequence')
                stops = [k for k, token in enumerate(tokens) if token in eos]
                check(not stops or stops == [len(tokens) - 1], 'Internal/multiple EOS')
                check(bool(stops) or len(tokens) == 1024, 'Response stopped without EOS or length cap')
                check(record['eos_terminated'] == bool(stops) and record['truncated'] == (not stops and len(tokens) == 1024),
                      'Stopping flags differ')
                completion = tokenizer.decode(tokens, skip_special_tokens=True)
                check(completion == record['completion'], 'Token decoding differs')
                grade = score_answer(completion, source['gold'])
                check(all(record[key] == value for key, value in grade.items()) and record['reward'] == int(grade['correct']),
                      'Independent frozen reward recomputation differs')
                batch_records.append(record)
            batches.append(batch)
        check(len(batch_records) == 512 and rows(run / 'responses.jsonl') == batch_records, 'Final response export differs')
        if role == 'student':
            mismatches = [i for i in range(64) if batch_records[i]['token_ids'] != old_records[i]['token_ids']]
            check(not mismatches, 'Initial student first64 greedy token parity failed: ' + str(mismatches))
            pilot = summary['pilot_first64']
            check(pilot['applicable'] is True and pilot['questions'] == 64 and pilot['sampled_token_ids_equal'] is True and
                  pilot['mismatch_zero_based_indices'] == [] and pilot['correct'] == sum(r['correct'] for r in batch_records[:64]) and
                  pilot['pilot_correct'] == sum(r['correct'] for r in old_records), 'Pilot parity summary differs')
        else:
            check(summary['pilot_first64']['applicable'] is False, 'Teacher cannot claim student token parity')
        expected = {'status': 'complete', 'identity_sha256': identity_sha, 'role': role, 'questions': 512,
            'correct': sum(r['correct'] for r in batch_records), 'accuracy': sum(r['correct'] for r in batch_records) / 512,
            'invalid_answers': sum(not r['answer_valid'] for r in batch_records),
            'truncated_answers': sum(r['truncated'] for r in batch_records),
            'generated_tokens': sum(len(r['token_ids']) for r in batch_records),
            'generation_seconds': sum(b['generation_seconds'] for b in batches),
            'batch_weight_audit_seconds': sum(b['weight_audit_seconds'] for b in batches),
            'peak_torch_allocated_gib': max(b['peak_torch_allocated_gib'] for b in batches),
            'optimizer_steps': 0, 'all_parameter_hash_before': frozen, 'all_parameter_hash_after': frozen,
            'sessions': len(sessions), 'session_statuses': expected_statuses,
            'batch_sessions_without_closing_full_hash': [], 'all_committed_batches_have_closing_full_hash': True,
            'batch_files_sha256': {p.name: sha(p) for p in batch_paths},
            'session_files_sha256': manifest(run / 'sessions', {'.json'}),
            'initial_weights_file_sha256': sha(run / 'initial_weights.json'), 'responses_sha256': sha(run / 'responses.jsonl')}
        for key in ['initial_weight_audit_seconds', 'final_weight_audit_seconds', 'model_and_adapter_load_seconds',
                    'preparation_seconds', 'recovery_validation_seconds']:
            expected[key] = sum(s[key] for s in sessions)
        expected['known_session_wall_seconds'] = sum(s['session_wall_seconds'] for s in sessions)
        finite(summary)
        for key, value in expected.items():
            check(summary[key] == value, f'{role} summary field differs: {key}')
        all_records[role] = batch_records
        report[role] = {'status': 'pass', 'run': str(run), 'questions': 512, 'correct': expected['correct'],
            'accuracy': expected['accuracy'], 'invalid_answers': expected['invalid_answers'],
            'truncated_answers': expected['truncated_answers'], 'generated_tokens': expected['generated_tokens'],
            'generation_seconds': expected['generation_seconds'], 'sessions': len(sessions),
            'session_statuses': expected_statuses, 'full_weight_hash_closure': True,
            'identity_sha256': identity_sha, 'summary_sha256': sha(run / 'summary.json'),
            'all_parameter_sha256': frozen, 'batches': 128}
    gains = sum(not s['correct'] and t['correct'] for s, t in zip(all_records['student'], all_records['teacher']))
    losses = sum(s['correct'] and not t['correct'] for s, t in zip(all_records['student'], all_records['teacher']))
    return {'status': 'pass', 'gpu_operations': 0, 'auditor_source_sha256': sha(Path(__file__)),
            'evaluator_source_sha256': EVALUATOR_SHA256, 'roles': report,
            'student_clean_pause_after_two_batches_and_resume': True,
            'student_first64_token_parity': True, 'teacher_student_all512_prompt_token_parity': True,
            'teacher_paired_gains': gains, 'teacher_paired_losses': losses,
            'teacher_minus_student_percentage_points': (gains - losses) / 512 * 100,
            'scope': 'Independent CPU evidence audit; no model inference or official-test evaluation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit_baselines(args.project_root)
    document = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(document, encoding='utf-8')
    print(document)
