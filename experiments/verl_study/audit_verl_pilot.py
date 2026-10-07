"""CPU-only audit of the genuine-verl pilot and its separately reloaded evaluations.

No model inference, optimizer operations, checkpoint changes or project writes.
The single explicit --output is the only file written. Native checkpoint bytes
are hashed; controller state is read only after verifying its committed digest.
"""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    assert path.is_relative_to(root.resolve()), (root, relative)
    return path


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


def close(actual, expected, tolerance=1e-7):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=tolerance), (actual, expected)


def state_sha(state):
    import torch
    result = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        result.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        result.update(memoryview(value.view(torch.uint8).numpy()))
    return result.hexdigest()


def adapter_sha(path):
    from safetensors.torch import load_file
    return state_sha(load_file(str(path / 'adapter_model.safetensors'), device='cpu'))


def rng_sha(state):
    result = hashlib.sha256()
    result.update(json.dumps(state['python'], separators=(',', ':')).encode())
    result.update(str(state['numpy'][0]).encode())
    result.update(state['numpy'][1].tobytes())
    result.update(json.dumps(state['numpy'][2:], separators=(',', ':')).encode())
    for tensor in [state['torch_cpu'], *state['torch_cuda']]:
        result.update(memoryview(tensor.cpu().contiguous().numpy()))
    return result.hexdigest()


def tree_sha(value):
    import torch
    result = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(f'tensor:{item.dtype}:{tuple(item.shape)}'.encode())
            result.update(memoryview(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, dict):
            result.update(b'dict')
            for key in sorted(item, key=lambda v: (type(v).__name__, repr(v))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update((type(item).__name__ + ':' + json.dumps(item, allow_nan=False)).encode())
    visit(value)
    return result.hexdigest()


def audit_framework(framework, project, protocol):
    assert framework['verl_commit'] == protocol['verl_commit']
    assert framework['upstream_engine_class'] == 'verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithLMHead'
    assert framework['train_batch_implementation'].startswith('verl.')
    assert Path(framework['train_batch_source']).resolve().is_relative_to((project / 'external/verl').resolve())
    assert framework['optimizer'] == 'torch.optim.adamw.AdamW'
    assert framework['base_storage'] == framework['forward_autocast'] == 'bfloat16'
    assert framework['adapter_master_storage'] == framework['selected_log_probability_dtype'] == 'float32'
    assert framework['native_microbatch_responses'] == 1 and framework['checkpoint_lora_only'] is True
    assert framework['checkpoint_contents'] == ['model', 'optimizer', 'extra']
    counts = framework['call_counts_this_process']
    assert set(counts) == {'train_batch', 'infer_batch', 'generate', 'save_checkpoint', 'load_checkpoint'}
    assert all(type(number) is int and number >= 0 for number in counts.values())
    if 'effective_engine_config' in framework:
        engine, optimizer, model = (framework['effective_engine_config'], framework['effective_optimizer_config'],
                                    framework['effective_model_config'])
        for key, expected in dict(strategy='fsdp', fsdp_size=1, use_orig_params=False, model_dtype='bf16',
                                  micro_batch_size_per_gpu=1, infer_micro_batch_size_per_gpu=1,
                                  use_torch_compile=False, use_remove_padding=False, use_fused_kernels=False,
                                  use_dynamic_bsz=False, seed=42).items():
            assert engine[key] == expected
        assert engine['mixed_precision'] == dict(param_dtype='bf16', reduce_dtype='fp32', buffer_dtype='fp32')
        for key, expected in dict(lr=1e-5, weight_decay=0.0, clip_grad=1.0, lr_scheduler_type='constant',
                                  lr_warmup_steps=0, total_training_steps=8).items():
            assert optimizer[key] == expected
        assert optimizer['betas'] == [0.9, 0.999] and optimizer['override_optimizer_config']['eps'] == 1e-8
        assert model['lora_rank'] == 16 and model['lora_alpha'] == 32
        assert model['lora_adapter_path'] == protocol['initial_adapter']
        assert model['override_config']['attn_implementation'] == 'sdpa'
    return counts


def prompt_ids(row, config, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': config['prompt_instruction'] + row['question']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(prompt, add_special_tokens=False)['input_ids']


def audit_answer(answer, gold, tokenizer, score_answer, eos, vocab_size):
    tokens = answer['token_ids']
    assert 0 < len(tokens) <= 1024
    assert all(type(token) is int and 0 <= token < vocab_size for token in tokens)
    stops = [index for index, token in enumerate(tokens) if token in eos]
    assert not stops or stops == [len(tokens) - 1]
    assert answer['eos_terminated'] == bool(stops)
    assert answer['truncated'] == (not stops and len(tokens) == 1024)
    assert stops or len(tokens) == 1024
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    assert decoded == answer['completion']
    grade = score_answer(decoded, gold)
    assert all(answer[key] == value for key, value in grade.items())
    assert answer['reward'] == int(grade['correct'])


def native_checkpoints(run, metadata, metrics, summary):
    import torch
    result = []
    for path in sorted((run / 'checkpoints').glob('boundary_*/complete.json')):
        marker = read(path)
        assert marker['identity'] == metadata['identity']
        claimed = marker['checkpoint_files_sha256']
        actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*')
                  if p.is_file() and p.name != 'complete.json'}
        assert actual == set(claimed), 'Native checkpoint inventory mismatch'
        assert 'controller.pt' in claimed
        native = [relative for relative in claimed if relative.startswith('engine/')]
        assert native and any('model' in name for name in native) and any('optim' in name for name in native)
        for relative, expected in claimed.items():
            assert sha(contained(path.parent, relative)) == expected
        for relative, expected in marker['evidence_sha256'].items():
            assert sha(contained(run, relative)) == expected
        # Locally generated controller state; all checkpoint bytes were verified above.
        state = torch.load(path.parent / 'controller.pt', map_location='cpu', weights_only=False)
        boundary = marker['boundary']
        assert state['identity'] == metadata['identity']
        assert state['next_iteration'] == boundary
        assert state['next_question_offset'] == boundary * state['prompts_per_iteration'] == boundary * 4
        assert state['checkpoint_owner'] == 'verl engine'
        assert state['controller_owner'] == 'train_verl_pilot.py'
        assert 'optimizer' not in state and 'adapter' not in state
        assert len(state['history']) == boundary
        assert state['actual_updates'] == marker['actual_updates'] == sum(m['optimizer_steps'] for m in metrics[:boundary])
        expected_adapter = metrics[boundary - 1]['policy_adapter_sha256_after'] if boundary else metadata['initial_adapter_sha256']
        assert state['adapter_sha256'] == marker['adapter_sha256'] == expected_adapter
        for saved, current in zip(state['history'], metrics[:boundary]):
            assert all(current[key] == value for key, value in saved.items())
        assert {'python', 'numpy', 'torch_cpu', 'torch_cuda'} <= state['rng'].keys()
        assert len(state['rng']['torch_cuda']) == 1
        assert rng_sha(state['rng']) == state['rng_sha256']
        model_path = path.parent / 'engine/model_world_size_1_rank_0.pt'
        optimizer_path = path.parent / 'engine/optim_world_size_1_rank_0.pt'
        extra_path = path.parent / 'engine/extra_state_world_size_1_rank_0.pt'
        assert all(p.is_file() for p in (model_path, optimizer_path, extra_path))
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        assert model and all('lora_A.' in name or 'lora_B.' in name for name in model)
        canonical_model = {name.replace('.default.', '.'): tensor for name, tensor in model.items()}
        assert len(canonical_model) == len(model) and state_sha(canonical_model) == expected_adapter
        del model, canonical_model
        optimizer = torch.load(optimizer_path, map_location='cpu', weights_only=False)
        extra = torch.load(extra_path, map_location='cpu', weights_only=False)
        optimizer_sha = tree_sha({'optimizer': optimizer, 'scheduler': extra['lr_scheduler']})
        assert optimizer_sha == state['optimizer_scheduler_sha256']
        assert optimizer['state']
        for group in optimizer['param_groups']:
            assert group['lr'] == 1e-5 and group['weight_decay'] == 0 and group['eps'] == 1e-8
        for parameter_state in optimizer['state'].values():
            assert float(parameter_state['step']) == state['actual_updates']
            assert torch.isfinite(parameter_state['exp_avg']).all() and torch.isfinite(parameter_state['exp_avg_sq']).all()
        if boundary == 8:
            assert optimizer_sha == summary['final_optimizer_scheduler_sha256']
        del optimizer, extra
        result.append({'boundary': boundary, 'adapter_sha256': expected_adapter,
                       'optimizer_scheduler_sha256': optimizer_sha, 'controller_rng_sha256': state['rng_sha256'],
                       'native_files': native, 'files_hashed': len(claimed),
                       'checkpoint_bytes': sum(contained(path.parent, relative).stat().st_size for relative in claimed)})
        del state
    assert sorted(item['boundary'] for item in result) == [7, 8]
    latest = read(run / 'checkpoints/latest.json')
    assert latest['identity'] == metadata['identity'] and latest['boundary'] == 8
    assert (run / 'checkpoints' / latest['directory'] / 'complete.json').is_file()
    return result


def audit_training(project, method, config, protocol, rows, tokenizer, score_answer, eos, initial_sha):
    import numpy as np
    experiment = project / 'experiments'
    run = experiment / 'runs' / f'verl_pilot_{method}_01'
    metadata, summary = read(run / 'metadata.json'), read(run / 'summary.json')
    metrics, sessions = read_rows(run / 'metrics.jsonl'), read_rows(run / 'sessions.jsonl')
    spec = metadata['spec']
    finite([summary, metrics, sessions])
    assert summary['status'] == 'complete' and len(metrics) == 8
    assert metadata['identity'] == summary['identity'] == canonical_sha(spec)
    assert spec['config'] == config and spec['protocol'] == protocol
    for field, path in (('config_sha256', experiment / 'configs/pilot.json'),
                        ('protocol_sha256', experiment / 'verl/protocol.json'),
                        ('reference_protocol_sha256', experiment / 'configs/repeated_pilot_v1.json')):
        assert spec[field] == sha(path)
    for relative, expected in spec['source_sha256'].items():
        assert sha(contained(experiment, relative)) == expected
    assert spec['verl_commit'] == protocol['verl_commit']
    commit = subprocess.check_output(['git', '-C', str(project / 'external/verl'), 'rev-parse', 'HEAD'], text=True).strip()
    diff = subprocess.check_output(['git', '-C', str(project / 'external/verl'), 'diff', '--binary', 'HEAD'])
    assert commit == spec['verl_commit'] and hashlib.sha256(diff).hexdigest() == spec['verl_tracked_diff_sha256']
    assert spec['method'] == summary['method'] == method
    assert spec['backend'] == 'fsdp'
    assert spec['iterations'] == 8 and spec['prompts_per_iteration'] == spec['group_size'] == 4
    assert spec['seed'] == 42 and spec['learning_rate'] == 1e-5 and spec['coefficient'] == 1.0
    assert spec['clip_epsilon'] == 0.2 and spec['gradient_clip_max_norm'] == 1.0
    assert spec['reference_kl_beta'] == 0 and spec['weight_decay'] == 0
    assert spec['eos_in_loss'] is True and spec['prompt_in_loss'] is False
    assert spec['source_rows'] == [row['source_row'] for row in rows]
    assert spec['question_sha256'] == [row['question_sha256'] for row in rows]
    assert metadata['verified_sampling'] == dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, max_new_tokens=1024)
    assert metadata['initial_adapter_sha256'] == summary['initial_adapter_sha256'] == initial_sha
    assert adapter_sha(run / 'adapter_initial') == initial_sha
    final_sha = adapter_sha(run / 'adapter_final')
    assert final_sha == summary['final_adapter_sha256']
    assert metadata['base_sha256_before'] == summary['base_sha256_before'] == summary['base_sha256_after']
    assert metadata['teacher_sha256_before'] == summary['teacher_sha256_before'] == summary['teacher_sha256_after']
    assert summary['base_weights_unchanged'] and summary['adapter_save_reload_exact']
    assert summary['native_adapter_reload_exact'] and summary['native_optimizer_scheduler_reload_exact']
    assert summary['controller_rng_restore_exact']
    audit_framework(metadata['framework'], project, protocol)
    audit_framework(summary['framework'], project, protocol)
    assert 'effective_engine_config' in metadata['framework'] and 'effective_engine_config' in summary['framework']
    assert (summary['teacher_sha256_before'] is None) == (method == 'rl')
    if method != 'rl':
        assert summary['teacher_weights_unchanged'] and summary['teacher_gradients_absent']
    assert sessions[0]['resume'] is False and sessions[0]['start_iteration'] == 0
    assert sessions[-1]['status'] == 'complete' and sessions[-1]['end_iteration'] == 8
    if method == 'rl':
        assert len(sessions) == 2 and sessions[0]['status'] == 'paused_at_boundary'
        assert sessions[0]['end_iteration'] == 1 and sessions[1]['resume'] is True and sessions[1]['start_iteration'] == 1
    else:
        assert len(sessions) == 1
    for session in sessions:
        assert session['base_sha256_after'] == metadata['base_sha256_before']
        assert session['teacher_sha256_after'] == metadata['teacher_sha256_before']
        audit_framework(session['framework_at_start'], project, protocol)
        audit_framework(session['framework_at_end'], project, protocol)
        assert session['native_optimizer_resume_verified'] == session['resume']
        assert session['controller_rng_resume_verified'] == session['resume']
    for session in sessions:
        start, end = session['start_iteration'], session['end_iteration']
        calls = session['framework_at_end']['call_counts_this_process']
        assert calls['generate'] == (end - start) * 4 and calls['infer_batch'] == end - start
        assert calls['train_batch'] == sum(metric['optimizer_steps'] for metric in metrics[start:end])
        assert calls['save_checkpoint'] == end - start + int(not session['resume'])
        assert calls['load_checkpoint'] == int(session['resume']) + int(session['status'] == 'complete')
    vocabulary_size = len(tokenizer)
    updates, current_hash = 0, initial_sha
    for index, metric in enumerate(metrics):
        assert metric['iteration'] == index + 1 and metric['question_offset'] == index * 4
        assert metric['questions'] == 4 and metric['responses'] == 16
        assert metric['student_update_owner'] == 'verl engine'
        session = next(s for s in sessions if s['start_iteration'] <= index < s['end_iteration'])
        local_round = index - session['start_iteration'] + 1
        calls = metric['framework_calls_this_process']
        assert calls['generate'] == local_round * 4 and calls['infer_batch'] == local_round
        assert calls['train_batch'] == sum(m['optimizer_steps'] for m in metrics[session['start_iteration']:index + 1])
        assert calls['save_checkpoint'] == local_round - 1 + int(not session['resume'])
        assert calls['load_checkpoint'] == int(session['resume'])
        assert metric['policy_adapter_sha256_before'] == current_hash
        directory = contained(run, metric['evidence_directory'])
        assert read(directory / 'metrics.json') == metric
        for relative, expected in metric['evidence_sha256'].items():
            assert sha(contained(run, relative)) == expected
        groups, scores = read_rows(directory / 'rollouts.jsonl'), read_rows(directory / 'token_scores.jsonl')
        assert len(groups) == 4 and len(scores) == 16
        answers, mixed, zero_ratio_loss, abs_signal_mean, any_signal = [], 0, 0., 0., False
        inner_generation = 0.
        for within, group in enumerate(groups):
            offset = index * 4 + within
            source = rows[offset]
            assert all(group[key] == value for key, value in source.items())
            assert group['prompt_token_ids'] == prompt_ids(source, config, tokenizer)
            assert group['iteration'] == index and group['question_offset'] == offset
            assert group['policy_step'] == updates and group['batch_seed'] == 42 + offset
            assert group['policy_adapter_sha256'] == current_hash
            assert len(group['answers']) == 4
            rewards = [a['reward'] for a in group['answers']]
            mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
            mixed += 0 < sum(rewards) < 4
            inner_generation += group['generation_seconds']
            for ai, answer in enumerate(group['answers']):
                audit_answer(answer, source['gold'], tokenizer, score_answer, eos, vocabulary_size)
                close(answer['advantage'], (answer['reward'] - mean) / (std + 1e-8), 1e-10)
                score = scores[within * 4 + ai]
                assert score['source_row'] == source['source_row'] and score['response'] == ai
                assert score['iteration'] == index and score['policy_step'] == updates
                assert score['policy_adapter_sha256'] == current_hash and score['token_ids'] == answer['token_ids']
                assert score['sequence_advantage'] == answer['advantage']
                old = np.asarray(score['old_student_logprobs'], dtype=np.float32)
                signal = np.asarray(score['effective_token_advantages'], dtype=np.float32)
                assert old.shape == signal.shape == (len(answer['token_ids']),)
                assert np.isfinite(old).all() and np.isfinite(signal).all() and (old <= 1e-6).all()
                if method == 'rl':
                    assert score['teacher_logprobs'] is None
                    expected = np.full_like(old, answer['advantage'])
                else:
                    teacher = np.asarray(score['teacher_logprobs'], dtype=np.float32)
                    assert teacher.shape == old.shape and np.isfinite(teacher).all() and (teacher <= 1e-6).all()
                    expected = teacher - old
                    if method == 'joint':
                        expected += np.float32(answer['advantage'])
                assert np.array_equal(signal, expected)
                zero_ratio_loss -= float(signal.astype(np.float64).mean()) / 16
                abs_signal_mean += float(np.abs(signal.astype(np.float64)).mean()) / 16
                any_signal |= bool(np.count_nonzero(signal))
                answers.append(answer)
        assert inner_generation <= metric['generation_wall_seconds'] + 1e-6
        counts = dict(generated_tokens=sum(len(a['token_ids']) for a in answers),
                      correct=sum(a['reward'] for a in answers), invalid_answers=sum(not a['answer_valid'] for a in answers),
                      truncated_answers=sum(a['truncated'] for a in answers), mixed_reward_groups=mixed)
        assert all(metric[key] == value for key, value in counts.items())
        assert metric['optimizer_steps'] == int(any_signal)
        assert bool(metric['changed_adapter_tensors']) == any_signal
        assert (metric['policy_adapter_sha256_after'] != current_hash) == any_signal
        difference = metric['max_preupdate_logp_difference']
        assert 0 <= difference <= 1e-3
        close(metric['loss_before'], zero_ratio_loss, math.expm1(difference) * abs_signal_mean + 2e-5)
        assert metric['backward_responses'] == (16 if any_signal else 0)
        if any_signal:
            assert metric['gradient_norm_before_clipping'] > 0 and metric['skip_reason'] is None
        else:
            assert metric['gradient_norm_before_clipping'] == 0 and metric['skip_reason'] == 'all_effective_signals_zero'
        close(metric['training_core_seconds'], sum(metric[key] for key in (
            'generation_wall_seconds', 'old_policy_scoring_seconds', 'teacher_scoring_seconds', 'backward_and_optimizer_seconds')))
        assert metric['iteration_precheckpoint_wall_seconds'] >= metric['training_core_seconds']
        if method == 'rl':
            assert metric['teacher_scoring_seconds'] == 0
        assert metric['peak_torch_allocated_gib'] == max(metric[key] for key in (
            'generation_peak_allocated_gib', 'old_scoring_peak_allocated_gib', 'teacher_scoring_peak_allocated_gib', 'update_peak_allocated_gib'))
        updates += metric['optimizer_steps']
        assert metric['actual_updates_after'] == updates
        current_hash = metric['policy_adapter_sha256_after']
    assert current_hash == final_sha
    assert summary['iterations'] == 8 and summary['questions'] == 32 and summary['responses'] == 128
    assert summary['actual_optimizer_steps'] == updates and summary['skipped_iterations'] == 8 - updates
    for key in ('questions', 'responses', 'generated_tokens', 'correct', 'mixed_reward_groups', 'invalid_answers',
                'truncated_answers', 'generation_wall_seconds', 'old_policy_scoring_seconds', 'teacher_scoring_seconds',
                'backward_and_optimizer_seconds', 'training_core_seconds', 'iteration_audit_seconds', 'checkpoint_payload_write_seconds'):
        close(summary[key], sum(metric[key] for metric in metrics))
    assert summary['peak_iteration_torch_allocated_gib'] == max(metric['peak_torch_allocated_gib'] for metric in metrics)
    checkpoints = native_checkpoints(run, metadata, metrics, summary)
    return {'status': 'pass', 'run': run.name, 'initial_adapter_sha256': initial_sha, 'final_adapter_sha256': final_sha,
            'base_sha256': summary['base_sha256_before'], 'teacher_sha256': summary['teacher_sha256_before'],
            'iterations': 8, 'actual_updates': updates, 'native_checkpoints': checkpoints,
            'framework_call_counts_across_sessions': {
                key: sum(session['framework_at_end']['call_counts_this_process'][key] for session in sessions)
                for key in ('train_batch', 'infer_batch', 'generate', 'save_checkpoint', 'load_checkpoint')},
            'config_and_source_provenance_verified': True, 'prompt_tokens_rewards_advantages_signals_verified': True,
            'fresh_policy_chain_verified': True, 'native_resume_exercised': method == 'rl',
            'metadata_sha256': sha(run / 'metadata.json'), 'summary_sha256': sha(run / 'summary.json')}


def audit_evaluation(project, name, expected_adapter, config, protocol, rows, tokenizer, score_answer, eos):
    experiment = project / 'experiments'
    run = experiment / 'runs' / name
    meta, summary = read(run / 'metadata.json'), read(run / 'summary.json')
    records = read_rows(run / 'responses.jsonl')
    assert summary['status'] == 'complete' and len(records) == 64
    assert meta['protocol_sha256'] == sha(experiment / 'verl/protocol.json')
    assert meta['config_sha256'] == sha(experiment / 'configs/pilot.json')
    assert meta['script_sha256'] == sha(experiment / 'verl/evaluate_verl_pilot.py')
    assert meta['source_rows'] == [row['source_row'] for row in rows]
    assert meta['student'] == config['models']['student'] and meta['reward'] == config['reward']
    assert Path(meta['adapter_path']).resolve() == expected_adapter.resolve()
    assert meta['adapter_file_sha256'] == sha(expected_adapter / 'adapter_model.safetensors')
    assert meta['adapter_state_sha256'] == adapter_sha(expected_adapter)
    assert meta['optimizer_steps'] == summary['optimizer_steps'] == 0
    assert meta['train_or_test_questions_used'] is False
    assert meta['sampling_api'] == 'Transformers 5 explicit resolved GenerationConfig'
    assert meta['enable_thinking'] is False and meta['batch_size'] == 4
    for key, value in dict(do_sample=False, num_beams=1, max_new_tokens=1024, use_cache=True).items():
        assert meta['generation'][key] == value
    assert summary['all_parameter_hash_before'] == summary['all_parameter_hash_after']
    vocabulary_size = len(tokenizer)
    times = []
    for index, (record, source) in enumerate(zip(records, rows)):
        assert all(record[key] == value for key, value in source.items())
        assert record['prompt_token_ids'] == prompt_ids(source, config, tokenizer)
        audit_answer(record, source['gold'], tokenizer, score_answer, eos, vocabulary_size)
        if index % 4 == 0:
            times.append(record['batch_generation_seconds'])
        else:
            close(record['batch_generation_seconds'], times[-1])
    counts = dict(questions=64, correct=sum(r['correct'] for r in records),
                  invalid_answers=sum(not r['answer_valid'] for r in records),
                  truncated_answers=sum(r['truncated'] for r in records), generated_tokens=sum(len(r['token_ids']) for r in records))
    assert all(summary[key] == value for key, value in counts.items())
    close(summary['accuracy'], counts['correct'] / 64)
    close(summary['generation_seconds'], sum(times))
    finite(summary)
    return {'status': 'pass', 'run': name, **counts, 'adapter_state_sha256': meta['adapter_state_sha256'],
            'versions': meta['versions'], 'data_tokens_rewards_and_hashes_verified': True,
            'summary_sha256': sha(run / 'summary.json'), 'responses_sha256': sha(run / 'responses.jsonl')}


def audit_validation(project, name, protocol, initial_sha):
    import ast
    import torch
    run = project / 'experiments/runs' / name
    summary = read(run / 'summary.json')
    checks = read(run / 'cpu_checks.json')
    manifest = read(run / 'source/manifest.json')
    assert summary['status'] == 'complete' and checks['status'] == 'pass'
    for relative, expected in manifest.items():
        assert sha(contained(run / 'source', relative)) == expected
    for field in ('loss_gradient_parity', 'positive_negative_clipping', 'unequal_response_lengths',
                  'prompt_padding_excluded_eos_included', 'all_zero_skips_native_optimizer'):
        assert checks[field] is True
    for field in ('frozen_base_unchanged', 'frozen_teacher_unchanged', 'live_engine_generation_pass'):
        assert summary[field] is True
    audit_framework(read(run / 'framework.json'), project, protocol)
    final_calls = audit_framework(summary['framework_final'], project, protocol)
    assert final_calls == dict(train_batch=3, infer_batch=6, generate=1, save_checkpoint=4, load_checkpoint=6)
    # Successful validation uses a frozen bridge snapshot. Later additions to
    # reporting metadata are allowed; learning/checkpoint implementations must match.
    def implementations(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        selected = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                selected[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name != 'framework_metadata':
                        selected[node.name + '.' + method.name] = ast.dump(method, include_attributes=False)
        return selected
    assert implementations(run / 'source/verl_engine_bridge.py') == implementations(project / 'experiments/verl/verl_engine_bridge.py')
    results, checkpoint_hashes = {}, {}
    assert [result['method'] for result in summary['methods']] == ['rl', 'opd', 'joint']
    for result in summary['methods']:
        method = result['method']
        assert read(run / f'{method}_result.json') == result
        assert result['optimizer_steps'] == 1 and result['backward_responses'] == 4
        assert result['initial_adapter_sha256'] == initial_sha
        assert result['changed_adapter_tensors'] > 0 and result['gradient_norm_before_clipping'] > 0
        assert result['max_preupdate_logp_difference'] <= 1e-3
        assert result['loss_after_on_same_responses'] < result['loss_before']
        for field in ('zero_signal_preserves_adam_and_adapter_after_momentum',
                      'native_model_optimizer_scheduler_rng_roundtrip_exact', 'native_checkpoint_directory_rename_supported'):
            assert result[field] is True
        assert adapter_sha(run / f'adapter_{method}') == result['final_adapter_sha256']
        checkpoint = run / f'checkpoint_{method}_renamed'
        for path in checkpoint.rglob('*'):
            if path.is_file():
                checkpoint_hashes[str(path.relative_to(run))] = sha(path)
        model = torch.load(checkpoint / 'model_world_size_1_rank_0.pt', map_location='cpu', weights_only=True)
        assert state_sha({key.replace('.default.', '.'): value for key, value in model.items()}) == result['final_adapter_sha256']
        del model
        optimizer = torch.load(checkpoint / 'optim_world_size_1_rank_0.pt', map_location='cpu', weights_only=True)
        assert optimizer['state'] and all(float(state['step']) == 1 for state in optimizer['state'].values())
        del optimizer
        results[method] = {'optimizer_steps': 1, 'adapter_and_native_model_match': True,
                           'runtime_roundtrip_assertions_passed': True}
    return {'status': 'pass', 'run': name, 'source_manifest': manifest, 'production_learning_code_matches_validated_code': True,
            'methods': results, 'checkpoint_sha256': checkpoint_hashes,
            'limit': 'Frozen-weight and RNG restore claims are runtime assertions in the verified validation source; model and optimizer checkpoint contents were independently read on CPU.'}


def compare_first_round(project, initial_sha):
    groups = {}
    for method in ('rl', 'opd', 'joint'):
        run = project / 'experiments/runs' / f'verl_pilot_{method}_01'
        metric = read_rows(run / 'metrics.jsonl')[0]
        assert metric['policy_adapter_sha256_before'] == initial_sha
        groups[method] = read_rows(contained(run, metric['evidence_directory']) / 'rollouts.jsonl')
    comparisons = {}
    for method in ('opd', 'joint'):
        mismatches = []
        prompts_equal = True
        for before, after in zip(groups['rl'], groups[method]):
            assert before['source_row'] == after['source_row']
            assert before['question_sha256'] == after['question_sha256']
            assert before['batch_seed'] == after['batch_seed']
            assert before['policy_step'] == after['policy_step'] == 0
            prompts_equal &= before['prompt_token_ids'] == after['prompt_token_ids']
            for index, (old, new) in enumerate(zip(before['answers'], after['answers'])):
                if old['token_ids'] != new['token_ids']:
                    mismatches.append({'source_row': before['source_row'], 'response_index': index,
                                       'rl_tokens': len(old['token_ids']), 'other_tokens': len(new['token_ids'])})
        comparisons[method + '_vs_rl'] = {'prompt_tokens_equal': bool(prompts_equal),
                                         'responses_compared': 16, 'sampled_token_ids_equal': not mismatches,
                                         'mismatched_responses': mismatches}
    matched = all(c['prompt_tokens_equal'] and c['sampled_token_ids_equal'] for c in comparisons.values())
    return {'same_initial_adapter_sha256': initial_sha, 'same_source_questions_and_generation_seeds': True,
            'all_first_round_prompt_and_sampled_token_ids_equal': matched, 'comparisons': comparisons,
            'interpretation': ('The first pre-update round is bitwise identical in its prompts and sampled responses across methods.'
                               if matched else 'Observed first-round token differences are recorded here; equal initial adapters and seeds do not by themselves guarantee identical numerical sampling.')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-run', default=None,
                        help='Optional successful native bridge validation to include, e.g. verl_bridge_validation_02')
    args = parser.parse_args()
    project = args.project_root.resolve()
    experiment = project / 'experiments'
    config, protocol = read(experiment / 'configs/pilot.json'), read(experiment / 'verl/protocol.json')
    assert protocol['formal_equal_budget_study'] is False
    assert protocol['iterations'] == 8 and protocol['prompts_per_iteration'] == protocol['answers_per_prompt'] == 4
    assert protocol['source_protocol_sha256'] == sha(experiment / 'configs/repeated_pilot_v1.json')
    assert config['reward']['version'] == 3
    splits = {}
    for split in ('train', 'dev'):
        path = experiment / config['data'][f'{split}_file']
        assert sha(path) == config['data'][f'{split}_sha256']
        splits[split] = read_rows(path)
    assert len(splits['train']) == 6961 and len(splits['dev']) == 512
    assert not ({r['question_sha256'] for r in splits['train']} & {r['question_sha256'] for r in splits['dev']})
    for name, expected in config['reward']['code_sha256'].items():
        assert sha(experiment / 'reward' / name) == expected
    sys.path.insert(0, str(experiment / 'reward'))
    from text_answer_reward import score_answer
    from transformers import AutoTokenizer, GenerationConfig
    import torch
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(config['models']['student']['local_snapshot'], local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_pretrained(config['models']['student']['local_snapshot'], local_files_only=True)
    eos = set(generation.eos_token_id if isinstance(generation.eos_token_id, list) else [generation.eos_token_id])
    initial = Path(protocol['initial_adapter'])
    assert sha(initial / 'adapter_model.safetensors') == protocol['initial_adapter_file_sha256']
    initial_sha = adapter_sha(initial)
    result = {'status': 'pass', 'audit_script_sha256': sha(Path(__file__)), 'gpu_operations': 0,
              'protocol_sha256': sha(experiment / 'verl/protocol.json'),
              'scope': 'Genuine verl engine with a custom synchronous HF controller; Ray and vLLM are not used for this pilot.',
              'baseline': audit_evaluation(project, 'verl_dev_before_01', initial, config, protocol, splits['dev'][:64], tokenizer, score_answer, eos),
              'methods': {}, 'interpretation': 'Engineering validation; no equal-budget winner claim.'}
    if args.validation_run:
        assert Path(args.validation_run).name == args.validation_run
        result['bridge_validation'] = audit_validation(project, args.validation_run, protocol, initial_sha)
    for method in ('rl', 'opd', 'joint'):
        training = audit_training(project, method, config, protocol, splits['train'][:32], tokenizer, score_answer, eos, initial_sha)
        evaluation = audit_evaluation(project, f'verl_dev_{method}_after_01', experiment / 'runs' / f'verl_pilot_{method}_01/adapter_final',
                                      config, protocol, splits['dev'][:64], tokenizer, score_answer, eos)
        assert evaluation['adapter_state_sha256'] == training['final_adapter_sha256']
        assert evaluation['versions'] == result['baseline']['versions']
        result['methods'][method] = {'training': training, 'evaluation': evaluation}
    assert len({result['methods'][method]['training']['base_sha256'] for method in result['methods']}) == 1
    assert result['methods']['opd']['training']['teacher_sha256'] == result['methods']['joint']['training']['teacher_sha256']
    result['first_round_cross_method_comparison'] = compare_first_round(project, initial_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
