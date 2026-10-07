"""Resumable, inference-only evaluation on all 512 frozen development questions.

Examples (run in the pinned .venv-verl environment):
  python evaluate_verl_study.py --project-root ROOT --role student --name study_student_dev_01 --check-only
  python evaluate_verl_study.py --project-root ROOT --role student --name study_student_dev_01 --stop-after-batches 2
  python evaluate_verl_study.py --project-root ROOT --role student --name study_student_dev_01 --resume
  python evaluate_verl_study.py --project-root ROOT --role teacher --name study_teacher_dev_01

Completed batch files are the authoritative records. responses.jsonl and summary.json
are derived only after all 512 questions finish. No test questions or optimizer are used.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

OPTIONS = dict(do_sample=False, num_beams=1, max_new_tokens=1024, use_cache=True)
PACKAGES = ('torch', 'transformers', 'peft', 'verl', 'tokenizers', 'safetensors')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    """Commit one durable file; an interrupted temporary file is never evidence."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def jsonl(path):
    # A physical line is a record; str.splitlines() would split embedded U+2028.
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_manifest(directory, suffixes):
    return {p.name: sha(p) for p in sorted(Path(directory).iterdir())
            if p.is_file() and p.suffix in suffixes}


def parameter_hash(model):
    """Same ordered parameter-byte fingerprint as the pilot evaluator."""
    import torch
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        require(not parameter.requires_grad, 'Evaluation parameter unexpectedly trainable: ' + name)
        require(parameter.grad is None, 'Unexpected gradient on ' + name)
        h.update(f'{name}:{parameter.dtype}:{tuple(parameter.shape)}'.encode())
        h.update(memoryview(parameter.detach().contiguous().cpu().view(torch.uint8).numpy()))
    return h.hexdigest()


def state_hash(state):
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        h.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        h.update(memoryview(value.view(torch.uint8).numpy()))
    return h.hexdigest()


def parameter_versions(model):
    """Cheap in-place mutation check; closing full hashes remain authoritative."""
    values = []
    for name, parameter in model.named_parameters():
        require(not parameter.requires_grad and parameter.grad is None, 'Trainable parameter or gradient during inference')
        values.append([name, parameter._version])
    return digest(values)


def resolved_generation(model_or_dummy, tokenizer):
    from transformers import GenerationConfig
    requested = GenerationConfig(**OPTIONS, temperature=None, top_p=None, top_k=None, min_p=None)
    requested.eos_token_id = model_or_dummy.generation_config.eos_token_id
    requested.pad_token_id = tokenizer.pad_token_id
    requested.bos_token_id = model_or_dummy.generation_config.bos_token_id
    effective, unused = model_or_dummy._prepare_generation_config(requested, **OPTIONS)
    require(not unused, 'Unexpected unused generation parameters')
    require(all(getattr(effective, key) == value for key, value in OPTIONS.items()),
            'Resolved greedy generation configuration differs')
    return effective


def prompt(tokenizer, config, row):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': config['prompt_instruction'] + row['question']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def prepare(args):
    """CPU-only source, data, tokenizer and decoding validation."""
    from transformers import AutoTokenizer, GenerationConfig
    from transformers.generation.utils import GenerationMixin
    experiment = args.project_root.resolve() / 'experiments'
    config_path, protocol_path = experiment / 'configs/pilot.json', experiment / 'verl/protocol.json'
    config, protocol = read_json(config_path), read_json(protocol_path)
    require(sha(config_path) == protocol['source_config_sha256'], 'Pilot config changed')
    data = {}
    for split, count in (('train', 6961), ('dev', 512)):
        path = experiment / config['data'][split + '_file']
        require(sha(path) == config['data'][split + '_sha256'], split + ' data hash changed')
        data[split] = jsonl(path)
        require(len(data[split]) == count, split + ' row count changed')
        require(len({r['question_sha256'] for r in data[split]}) == count, 'Duplicate questions')
        for row in data[split]:
            normalized = ' '.join(row['question'].split()).casefold()
            require(hashlib.sha256(normalized.encode()).hexdigest() == row['question_sha256'],
                    'Question fingerprint mismatch')
    require(not ({r['question_sha256'] for r in data['train']} &
                 {r['question_sha256'] for r in data['dev']}), 'Train/dev overlap')
    require(config['reward']['version'] == 3, 'Expected frozen reward v3')
    for filename, expected in config['reward']['code_sha256'].items():
        require(sha(experiment / 'reward' / filename) == expected, 'Reward source changed: ' + filename)
    sys.path.insert(0, str(experiment / 'reward'))
    from text_answer_reward import score_answer

    tokenizers = {role: AutoTokenizer.from_pretrained(config['models'][role]['local_snapshot'],
                  local_files_only=True, trust_remote_code=False, padding_side='left')
                  for role in ('student', 'teacher')}
    all_prompts = {role: [prompt(tok, config, row) for row in data['dev']]
                   for role, tok in tokenizers.items()}
    all_ids = {role: tok(all_prompts[role], add_special_tokens=False)['input_ids']
               for role, tok in tokenizers.items()}
    left, right = tokenizers['student'], tokenizers['teacher']
    compatibility = {
        'vocabulary_and_token_ids_equal': left.get_vocab() == right.get_vocab(),
        'special_token_ids_equal': left.all_special_ids == right.all_special_ids,
        'all_512_prompt_token_ids_equal': all_ids['student'] == all_ids['teacher'],
        'vocabulary_sha256': {role: digest(tok.get_vocab()) for role, tok in tokenizers.items()},
        'scope': 'Vocabulary/token IDs, special IDs, and the 512 actual prompt encodings; each model uses its own tokenizer.'}
    require(all(compatibility[key] for key in ('vocabulary_and_token_ids_equal',
            'special_token_ids_equal', 'all_512_prompt_token_ids_equal')), 'Student/teacher tokenizer mismatch')

    baseline = experiment / 'runs/verl_dev_before_01'
    baseline_meta, baseline_rows = read_json(baseline / 'metadata.json'), jsonl(baseline / 'responses.jsonl')
    require(len(baseline_rows) == 64, 'Fresh pilot baseline is not complete')
    for index, row in enumerate(baseline_rows):
        require(row['source_row'] == data['dev'][index]['source_row'] and
                row['question_sha256'] == data['dev'][index]['question_sha256'], 'Pilot dev order differs')
        require(row['prompt_token_ids'] == all_ids['student'][index], 'Pilot prompt token parity failed')
    require(baseline_meta['config_sha256'] == sha(config_path) and
            baseline_meta['protocol_sha256'] == sha(protocol_path), 'Baseline provenance differs')
    require(all(baseline_meta['generation'][key] == value for key, value in OPTIONS.items()),
            'Pilot greedy settings differ')

    adapter = None
    adapter_files = None
    if args.role == 'student':
        adapter = (args.adapter or Path(protocol['initial_adapter'])).resolve()
        require((adapter / 'adapter_model.safetensors').is_file(), 'Adapter file missing')
        adapter_files = file_manifest(adapter, {'.json', '.safetensors'})
        if args.adapter is None:
            require(adapter_files['adapter_model.safetensors'] == protocol['initial_adapter_file_sha256'],
                    'Initial adapter hash differs from protocol')
    else:
        require(args.adapter is None, 'Teacher evaluation must not use an adapter')

    spec = config['models'][args.role]
    snapshot = Path(spec['local_snapshot'])
    require(snapshot.name == spec['revision'], 'Model snapshot is not the pinned revision')
    tokenizer = tokenizers[args.role]
    class ConfigOnly(GenerationMixin):
        pass
    dummy = ConfigOnly()
    dummy.generation_config = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
    effective = resolved_generation(dummy, tokenizer)
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(all(versions[name] == value for name, value in baseline_meta['versions'].items()),
            'Environment differs from fresh verl pilot baseline')
    identity = {
        'schema_version': 1, 'role': args.role, 'questions': 512, 'batch_size': args.batch_size,
        'seed': 42, 'model': spec, 'model_files_sha256': file_manifest(snapshot, {'.json', '.safetensors', '.jinja', '.txt', '.model'}),
        'adapter_path': str(adapter) if adapter else None, 'adapter_files_sha256': adapter_files,
        'config_sha256': sha(config_path), 'source_protocol_sha256': sha(protocol_path),
        'script_sha256': sha(Path(__file__)), 'versions': versions,
        'selection': 'all 512 questions in frozen development order',
        'source_rows': [r['source_row'] for r in data['dev']],
        'question_sha256': [r['question_sha256'] for r in data['dev']],
        'data': config['data'], 'reward': config['reward'], 'prompt_instruction': config['prompt_instruction'],
        'prompt_token_ids_sha256': digest(all_ids[args.role]), 'generation': effective.to_dict(),
        'sampling_api': 'Transformers 5 explicit resolved GenerationConfig', 'enable_thinking': False,
        'evaluation': 'greedy, one answer per question; BF16 model; SDPA; left padding',
        'train_or_test_questions_used': False, 'optimizer_steps': 0,
        'tokenizer_compatibility': compatibility,
        'pilot_baseline_metadata_sha256': sha(baseline / 'metadata.json'),
        'pilot_baseline_responses_sha256': sha(baseline / 'responses.jsonl'),
        'pilot_first_64_prompt_token_ids_equal': True,
        'training_backend': 'none; frozen Hugging Face model inference in the pinned verl environment',
        'hardware_requirement': 'one visible NVIDIA GeForce RTX 5090',
        'weight_audit_method': 'Full parameter-byte hash at session start/end; requires_grad=False, grad=None, and unchanged tensor._version after each batch. A hard-killed session may lack its closing full hash; such sessions are listed explicitly.',
    }
    return experiment, config, data['dev'], tokenizer, all_prompts[args.role], all_ids[args.role], effective, score_answer, identity


def validate_batches(run, identity, rows, tokenizer, prompt_ids, score_answer, expected_weights=None):
    """Validate every committed batch before trusting its cursor or records."""
    batches, records = [], []
    eos_value = identity['generation']['eos_token_id']
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    vocabulary_size = len(tokenizer)
    for batch_index, path in enumerate(sorted((run / 'batches').glob('*.json'))):
        require(path.name == f'{batch_index:04d}.json', 'Missing, duplicate, or reordered batch')
        batch = read_json(path)
        require(batch['identity_sha256'] == digest(identity), 'Batch identity mismatch')
        require(batch['optimizer_steps'] == 0, 'Evaluation batch reports an optimizer step')
        begin = batch_index * identity['batch_size']
        end = min(begin + identity['batch_size'], 512)
        require(begin < 512 and batch['begin'] == begin and batch['end'] == end, 'Batch range mismatch')
        require(len(batch['records']) == end - begin, 'Partial committed batch')
        require(digest(batch['records']) == batch['records_sha256'], 'Batch record hash mismatch')
        require(batch['parameter_versions_after'] == batch['parameter_versions_before'], 'Parameter version changed within batch')
        if expected_weights is not None:
            require(batch['session_initial_parameter_hash'] == expected_weights, 'Batch frozen model hash mismatch')
        for key in ('generation_seconds', 'weight_audit_seconds', 'peak_torch_allocated_gib'):
            require(math.isfinite(batch[key]) and batch[key] >= 0, 'Invalid timing/memory field')
        for index, record in enumerate(batch['records'], begin):
            row = rows[index]
            require(all(record[key] == value for key, value in row.items()), 'Recorded source row differs')
            require(record['prompt_token_ids'] == prompt_ids[index], 'Recorded prompt differs')
            tokens = record['token_ids']
            require(isinstance(tokens, list) and 0 < len(tokens) <= 1024 and
                    all(type(t) is int and 0 <= t < vocabulary_size for t in tokens), 'Invalid completion token IDs')
            eos_positions = [i for i, token in enumerate(tokens) if token in eos]
            require(not eos_positions or eos_positions == [len(tokens) - 1], 'Unexpected internal EOS')
            require(record['eos_terminated'] == bool(eos_positions), 'EOS flag mismatch')
            require(record['truncated'] == (not eos_positions and len(tokens) >= 1024), 'Truncation flag mismatch')
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            require(record['completion'] == text, 'Decoded completion differs')
            grade = score_answer(text, row['gold'])
            require(all(record[key] == value for key, value in grade.items()) and
                    record['reward'] == int(grade['correct']), 'Frozen reward recomputation differs')
        batches.append(batch)
        records.extend(batch['records'])
    return batches, records


def compare_pilot(experiment, identity, records):
    if identity['role'] != 'student' or identity['batch_size'] != 4:
        return {'applicable': False, 'reason': 'Only same student, adapter, and batch-four path is comparable'}
    baseline = experiment / 'runs/verl_dev_before_01'
    meta = read_json(baseline / 'metadata.json')
    if identity['adapter_files_sha256']['adapter_model.safetensors'] != meta['adapter_file_sha256']:
        return {'applicable': False, 'reason': 'Adapter differs from initial pilot baseline'}
    if len(records) < 64:
        return {'applicable': False, 'reason': 'Fewer than 64 completed questions'}
    old = jsonl(baseline / 'responses.jsonl')
    changed = [i for i in range(64) if records[i]['token_ids'] != old[i]['token_ids']]
    return {'applicable': True, 'questions': 64, 'sampled_token_ids_equal': not changed,
            'mismatch_zero_based_indices': changed,
            'correct': sum(r['correct'] for r in records[:64]),
            'pilot_correct': sum(r['correct'] for r in old),
            'note': 'Observed greedy token parity; differences are reported rather than overwritten.'}


def initial_weight_hash(run, identity):
    weights = read_json(run / 'initial_weights.json')
    require(weights['identity_sha256'] == digest(identity), 'Initial weight identity mismatch')
    require(isinstance(weights['all_parameter_sha256'], str) and
            len(weights['all_parameter_sha256']) == 64, 'Invalid initial weight fingerprint')
    return weights['all_parameter_sha256']


def audit_summary(run, identity, batches, records, frozen):
    require(len(records) == 512, 'Evaluation is incomplete')
    summary = read_json(run / 'summary.json')
    expected = {'status': 'complete', 'identity_sha256': digest(identity), 'role': identity['role'],
        'questions': 512, 'correct': sum(r['correct'] for r in records),
        'accuracy': sum(r['correct'] for r in records) / 512,
        'invalid_answers': sum(not r['answer_valid'] for r in records),
        'truncated_answers': sum(r['truncated'] for r in records),
        'generated_tokens': sum(len(r['token_ids']) for r in records),
        'generation_seconds': sum(b['generation_seconds'] for b in batches),
        'batch_weight_audit_seconds': sum(b['weight_audit_seconds'] for b in batches),
        'peak_torch_allocated_gib': max(b['peak_torch_allocated_gib'] for b in batches),
        'optimizer_steps': 0, 'all_parameter_hash_before': frozen, 'all_parameter_hash_after': frozen,
        'responses_sha256': sha(run / 'responses.jsonl'),
        'batch_files_sha256': {p.name: sha(p) for p in sorted((run / 'batches').glob('*.json'))},
        'session_files_sha256': {p.name: sha(p) for p in sorted((run / 'sessions').glob('*.json'))},
        'initial_weights_file_sha256': sha(run / 'initial_weights.json')}
    require(all(summary[key] == value for key, value in expected.items()), 'Recomputed summary or file manifest differs')
    require(jsonl(run / 'responses.jsonl') == records, 'Response export differs from committed batches')
    sessions = [read_json(p) for p in sorted((run / 'sessions').glob('*.json'))]
    require(summary['sessions'] == len(sessions) and summary['session_statuses'] == [s['status'] for s in sessions],
            'Session summary differs')
    session_ids = {s['session_id'] for s in sessions}
    require(all(b['session_id'] in session_ids for b in batches), 'Committed batch has no session record')
    for session in sessions:
        require(session['identity_sha256'] == digest(identity) and session['optimizer_steps'] == 0,
                'Session identity/optimizer mismatch')
        if session['status'] in ('paused', 'complete'):
            require(session['all_parameter_hash_before'] == session['all_parameter_hash_after'] == frozen,
                    'Finished session weight hashes differ')
    unclosed = [s['session_id'] for s in sessions if s['status'] not in ('paused', 'complete') and
                any(b['session_id'] == s['session_id'] for b in batches)]
    require(summary['batch_sessions_without_closing_full_hash'] == unclosed and
            summary['all_committed_batches_have_closing_full_hash'] == (not unclosed), 'Closing hash coverage differs')
    for key in ('initial_weight_audit_seconds', 'model_and_adapter_load_seconds', 'preparation_seconds',
                'recovery_validation_seconds', 'final_weight_audit_seconds'):
        require(summary[key] == sum(s.get(key, 0) for s in sessions), 'Session cost total differs: ' + key)
    require(summary['known_session_wall_seconds'] == sum(s.get('session_wall_seconds', 0) for s in sessions),
            'Session wall total differs')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--role', choices=('student', 'teacher'), required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--adapter', type=Path, help='Student only; defaults to the protocol initial adapter')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-batches', type=int, help='Pause after this many NEW batches in this process')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true')
    modes.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    require(args.name and Path(args.name).name == args.name and args.name not in ('.', '..'), 'Invalid run name')
    require(args.batch_size > 0 and args.batch_size <= 512, 'Batch size must be 1..512')
    require(args.stop_after_batches is None or args.stop_after_batches > 0, 'Pause count must be positive')
    preparation_started = time.perf_counter()
    experiment, config, rows, tokenizer, prompts, prompt_ids, effective, score_answer, identity = prepare(args)
    preparation_seconds = time.perf_counter() - preparation_started
    print('PASS: 512 frozen disjoint development rows, reward v3, pinned sources, tokenizer compatibility, and pilot prompt parity.', flush=True)
    if args.check_only:
        print(json.dumps({'status': 'check_only_pass', 'identity_sha256': digest(identity),
                          'role': args.role, 'questions': 512, 'gpu_operations': 0,
                          'tokenizer_compatibility': identity['tokenizer_compatibility'],
                          'preparation_seconds': preparation_seconds}, indent=2), flush=True)
        return
    run = experiment / 'runs' / args.name
    if args.audit_only:
        metadata = read_json(run / 'metadata.json')
        require(metadata['identity'] == identity and metadata['identity_sha256'] == digest(identity), 'Run identity differs')
        frozen = initial_weight_hash(run, identity)
        batches, records = validate_batches(run, identity, rows, tokenizer, prompt_ids, score_answer, frozen)
        summary = audit_summary(run, identity, batches, records, frozen)
        require(summary['pilot_first64'] == compare_pilot(experiment, identity, records), 'Pilot parity summary differs')
        print(json.dumps({'status': 'audit_pass', 'questions': len(records), 'correct': summary['correct'],
                          'batches': len(batches), 'gpu_operations': 0,
                          'pilot_first64': compare_pilot(experiment, identity, records)}, indent=2), flush=True)
        return

    import fcntl
    if args.resume:
        require(run.is_dir(), 'Resume run does not exist')
    else:
        run.mkdir(parents=True, exist_ok=False)
    with (run / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run / 'batches').mkdir(exist_ok=True)
        (run / 'sessions').mkdir(exist_ok=True)
        metadata_path = run / 'metadata.json'
        if metadata_path.exists():
            stored = read_json(metadata_path)
            require(stored['identity'] == identity and stored['identity_sha256'] == digest(identity),
                    'Resume identity changed: source, model, adapter, config, tokenizer, versions, or batch size')
        else:
            require(not list((run / 'batches').glob('*.json')), 'Missing metadata with existing results')
            atomic_json(metadata_path, {'identity': identity, 'identity_sha256': digest(identity), 'created_at': utc()})
        recovery_tic = time.perf_counter()
        known_weights = initial_weight_hash(run, identity) if (run / 'initial_weights.json').exists() else None
        require(known_weights is not None or not list((run / 'batches').glob('*.json')),
                'Committed batches exist without an initial weight fingerprint')
        batches, records = validate_batches(run, identity, rows, tokenizer, prompt_ids, score_answer, known_weights)
        recovery_seconds = time.perf_counter() - recovery_tic
        if (run / 'summary.json').exists() and read_json(run / 'summary.json')['status'] == 'complete':
            summary = audit_summary(run, identity, batches, records, known_weights)
            require(summary['pilot_first64'] == compare_pilot(experiment, identity, records), 'Pilot parity summary differs')
            print('Already complete; all records, hashes, and summary fields passed CPU recomputation.', flush=True)
            return
        import torch
        from transformers import AutoModelForCausalLM
        require(torch.cuda.is_available(), 'CUDA unavailable')
        require(torch.cuda.device_count() == 1 and 'RTX 5090' in torch.cuda.get_device_name(0),
                'Expected exactly one visible RTX 5090 GPU')
        torch.set_num_threads(4)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        session_id = utc().replace(':', '').replace('.', '') + '_' + uuid.uuid4().hex[:8]
        session_path = run / 'sessions' / (session_id + '.json')
        session = {'session_id': session_id, 'identity_sha256': digest(identity), 'status': 'starting',
                   'started_at': utc(), 'start_completed_batches': len(batches), 'resumed': args.resume,
                   'preparation_seconds': preparation_seconds, 'recovery_validation_seconds': recovery_seconds,
                   'optimizer_steps': 0, 'gpu_name': torch.cuda.get_device_name(0),
                   'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory}
        atomic_json(session_path, session)
        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(identity['model']['local_snapshot'],
                local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
                device_map={'': 'cuda:0'}, attn_implementation='sdpa').eval()
            adapter_state = None
            if args.role == 'student':
                from peft import PeftModel, get_peft_model_state_dict
                from peft.utils.save_and_load import load_peft_weights
                model = PeftModel.from_pretrained(model, identity['adapter_path'], local_files_only=True,
                    is_trainable=False, autocast_adapter_dtype=True).eval()
                saved = load_peft_weights(identity['adapter_path'], device='cpu', local_files_only=True)
                actual = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model, save_embedding_layers=False).items()}
                require(saved.keys() == actual.keys() and all(torch.equal(saved[k], actual[k]) for k in saved),
                        'Loaded student adapter differs from saved values')
                adapter_state = state_hash(actual)
                del actual, saved
            model.requires_grad_(False)
            require(not model.training and not any(p.requires_grad for p in model.parameters()), 'Model is not frozen/eval')
            require(resolved_generation(model, tokenizer).to_dict() == effective.to_dict(), 'Loaded generation config differs')
            torch.cuda.synchronize()
            session['model_and_adapter_load_seconds'] = time.perf_counter() - started
            audit_tic = time.perf_counter()
            frozen = parameter_hash(model)
            session['initial_weight_audit_seconds'] = time.perf_counter() - audit_tic
            weights = {'all_parameter_sha256': frozen, 'adapter_state_sha256': adapter_state,
                       'identity_sha256': digest(identity)}
            if (run / 'initial_weights.json').exists():
                require(read_json(run / 'initial_weights.json') == weights, 'Reloaded model weights differ')
            else:
                atomic_json(run / 'initial_weights.json', weights)
            require(all(b['session_initial_parameter_hash'] == frozen for b in batches), 'Earlier batch weights differ')
            version_hash = parameter_versions(model)
            session.update(status='running', all_parameter_hash_before=frozen)
            atomic_json(session_path, session)
            eos = effective.eos_token_id
            eos_set = set(eos if isinstance(eos, list) else [eos])
            new_batches = 0
            for begin in range(len(records), 512, args.batch_size):
                end = min(begin + args.batch_size, 512)
                inputs = tokenizer(prompts[begin:end], add_special_tokens=False, padding=True,
                                   return_tensors='pt').to('cuda:0')
                width = inputs.input_ids.shape[1]
                torch.cuda.synchronize()
                tic = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(**inputs, generation_config=effective, **OPTIONS)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tic
                completed = []
                for index, sequence in enumerate(output[:, width:].cpu().tolist(), begin):
                    stop = next((i for i, token in enumerate(sequence) if token in eos_set), None)
                    tokens = sequence if stop is None else sequence[:stop + 1]
                    require(tokens, 'Empty generated response')
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    grade = score_answer(text, rows[index]['gold'])
                    completed.append({**rows[index], **grade, 'reward': int(grade['correct']),
                        'completion': text, 'token_ids': tokens, 'prompt_token_ids': prompt_ids[index],
                        'eos_terminated': stop is not None, 'truncated': stop is None and len(tokens) >= 1024})
                del inputs, output
                audit_tic = time.perf_counter()
                after_versions = parameter_versions(model)
                require(after_versions == version_hash, 'Model parameter versions changed during evaluation')
                audit_seconds = time.perf_counter() - audit_tic
                batch = {'identity_sha256': digest(identity), 'session_id': session_id,
                         'begin': begin, 'end': end, 'records': completed, 'records_sha256': digest(completed),
                         'generation_seconds': elapsed, 'weight_audit_seconds': audit_seconds,
                         'peak_torch_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                         'session_initial_parameter_hash': frozen,
                         'parameter_versions_before': version_hash, 'parameter_versions_after': after_versions,
                         'optimizer_steps': 0, 'committed_at': utc()}
                atomic_json(run / 'batches' / f'{len(batches):04d}.json', batch)
                batches.append(batch)
                records.extend(completed)
                new_batches += 1
                session.update(end_completed_batches=len(batches), last_committed_questions=len(records))
                atomic_json(session_path, session)
                print(f'{args.role} greedy dev {len(records)}/512: {sum(r["correct"] for r in records)} correct; '
                      f'batch generation {elapsed:.2f}s; mutation check {audit_seconds:.3f}s', flush=True)
                if args.stop_after_batches and new_batches >= args.stop_after_batches:
                    break
            audit_tic = time.perf_counter()
            frozen_after = parameter_hash(model)
            require(frozen_after == frozen, 'Final session full parameter hash changed')
            session['final_weight_audit_seconds'] = time.perf_counter() - audit_tic
            session.update(status='complete' if len(records) == 512 else 'paused', ended_at=utc(),
                           session_wall_seconds=time.perf_counter() - started,
                           all_parameter_hash_after=frozen_after,
                           peak_torch_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            atomic_json(session_path, session)
        except BaseException as error:
            session.update(status='failed', ended_at=utc(), exception_type=type(error).__name__,
                           error=str(error), session_wall_seconds=time.perf_counter() - started)
            atomic_json(session_path, session)
            raise
        if len(records) < 512:
            print(f'PAUSED: {len(records)}/512 committed. Resume with the same arguments plus --resume.', flush=True)
            return
        batches, records = validate_batches(run, identity, rows, tokenizer, prompt_ids, score_answer, frozen)
        temporary = run / ('responses.jsonl.tmp-' + uuid.uuid4().hex)
        with temporary.open('x', encoding='utf-8') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, run / 'responses.jsonl')
        sessions = [read_json(p) for p in sorted((run / 'sessions').glob('*.json'))]
        summary = {'status': 'complete', 'identity_sha256': digest(identity), 'role': args.role,
            'questions': 512, 'correct': sum(r['correct'] for r in records),
            'accuracy': sum(r['correct'] for r in records) / 512,
            'invalid_answers': sum(not r['answer_valid'] for r in records),
            'truncated_answers': sum(r['truncated'] for r in records),
            'generated_tokens': sum(len(r['token_ids']) for r in records),
            'generation_seconds': sum(b['generation_seconds'] for b in batches),
            'batch_weight_audit_seconds': sum(b['weight_audit_seconds'] for b in batches),
            'initial_weight_audit_seconds': sum(s.get('initial_weight_audit_seconds', 0) for s in sessions),
            'final_weight_audit_seconds': sum(s.get('final_weight_audit_seconds', 0) for s in sessions),
            'model_and_adapter_load_seconds': sum(s.get('model_and_adapter_load_seconds', 0) for s in sessions),
            'known_session_wall_seconds': sum(s.get('session_wall_seconds', 0) for s in sessions),
            'preparation_seconds': sum(s['preparation_seconds'] for s in sessions),
            'recovery_validation_seconds': sum(s['recovery_validation_seconds'] for s in sessions),
            'peak_torch_allocated_gib': max(b['peak_torch_allocated_gib'] for b in batches),
            'sessions': len(sessions), 'session_statuses': [s['status'] for s in sessions],
            'batch_sessions_without_closing_full_hash': [s['session_id'] for s in sessions
                if s['status'] not in ('paused', 'complete') and any(b['session_id'] == s['session_id'] for b in batches)],
            'all_committed_batches_have_closing_full_hash': all(s['status'] in ('paused', 'complete')
                for s in sessions if any(b['session_id'] == s['session_id'] for b in batches)),
            'timing_limit': 'Generation/audit sums cover committed batches. Crashes may lose uncommitted work and wall-time records; preparation, model loading, and CPU recovery validation are reported separately. Process peak memory is cumulative, not per-batch incremental memory.',
            'optimizer_steps': 0, 'all_parameter_hash_before': frozen, 'all_parameter_hash_after': frozen,
            'batch_files_sha256': {p.name: sha(p) for p in sorted((run / 'batches').glob('*.json'))},
            'session_files_sha256': {p.name: sha(p) for p in sorted((run / 'sessions').glob('*.json'))},
            'initial_weights_file_sha256': sha(run / 'initial_weights.json'),
            'responses_sha256': sha(run / 'responses.jsonl'),
            'pilot_first64': compare_pilot(experiment, identity, records),
            'note': 'Full frozen development evaluation; no official test use. Teacher/student calibration, not an equal-budget training result.'}
        atomic_json(run / 'summary.json', summary)
        audit_summary(run, identity, batches, records, frozen)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
