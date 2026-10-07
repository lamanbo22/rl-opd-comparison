"""Independent CPU audit of one completed, eligible formal verl training run.

Only --output is written. Training modules and their scheduling/validation
functions are never imported. CUDA and inference are never used. Native tensor
payloads are loaded on CPU only after checking their committed file hashes.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from audit_verl_pilot import (adapter_sha, audit_answer, audit_framework, canonical_sha,
    close, contained, finite, prompt_ids, read, read_rows, rng_sha, sha, state_sha, tree_sha)


def cursor_state(rows, seed, seen):
    assert seen >= 0
    epoch, offset = divmod(seen, len(rows))
    order = list(range(len(rows)))
    random.Random(f'rl-opd-study:{seed}:{epoch}').shuffle(order)
    state = dict(seed=seed, epoch=epoch, offset=offset, questions_seen=seen,
                 dataset_size=len(rows),
                 order_sha256=hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest(),
                 shuffle_rule="Python random.Random('rl-opd-study:{seed}:{epoch}').shuffle(range(N))")
    return state, order


def reconstruct_schedules(initial, metrics, require_complete=True):
    """Independent replay of the frozen guarded clock, including zero updates."""
    value = dict(initial)
    assert value['mode'] == 'budgeted' and value['endpoint_policy'] == 'guarded_ceiling'
    assert value['budget_seconds'] == 7200 and value['reserve_seconds'] == 480
    assert value['overrun_tolerance_fraction'] == 0
    assert all(value[k] == 0 for k in ('charged_seconds', 'completed_rounds', 'actual_updates',
                                     'phase1_seconds', 'phase2_seconds'))
    assert value['switch_round'] is None and value['switch_at_seconds'] is None
    phases = value['method'].split('_then_')
    assert phases in (['rl'], ['opd'], ['joint'], ['rl', 'opd'], ['opd', 'rl'])
    states = [dict(value)]
    for index, metric in enumerate(metrics):
        before = value['charged_seconds']
        assert before < 7200 and 7200 - before >= 480, 'Batch began after guard stop'
        phase = int(len(phases) == 2 and before >= 3600)
        duration = metric['charged_iteration_seconds']
        assert math.isfinite(duration) and duration > 0
        assert metric['optimizer_steps'] in (0, 1)
        value['charged_seconds'] += duration
        value['completed_rounds'] += 1
        value['actual_updates'] += metric['optimizer_steps']
        value[f'phase{phase+1}_seconds'] += duration
        after = value['charged_seconds']
        next_phase = int(len(phases) == 2 and after >= 3600)
        switched = phase != next_phase
        if switched:
            assert value['switch_round'] is None
            value['switch_round'], value['switch_at_seconds'] = index + 1, after
        expected = dict(charged_iteration_seconds=duration, charged_seconds_before=before,
            charged_seconds_after=after, active_method=phases[phase], next_active_method=phases[next_phase],
            phase_index=phase, phase_switch_at_boundary=switched,
            crossed_midpoint=before < 3600 <= after, crossed_budget=before < 7200 <= after,
            primary_endpoint_eligible=after <= 7200, overrun_seconds=max(0., after-7200))
        assert all(metric[k] == v for k, v in expected.items()), ('Clock mismatch', index+1)
        assert after <= 7200, 'Unexpected overrun is ineligible for formal evaluation'
        states.append(dict(value))
    if require_complete:
        assert metrics and 6720 < value['charged_seconds'] <= 7200, 'Incomplete guarded endpoint'
        if len(phases) == 2:
            assert value['phase1_seconds'] > 0 and value['phase2_seconds'] > 0
    return states


def expected_retention(states):
    terminal = len(states) - 1
    assert terminal >= 1
    crossings = {}
    for percent in (25, 50, 75):
        candidates = [i for i, s in enumerate(states) if percent/100*7200 <= s['charged_seconds'] <= 7200]
        assert candidates, f'Missing {percent}% crossing'
        crossings[f'milestone_{percent:03d}'] = candidates[0]
    return {terminal-1, terminal, *crossings.values()}, {'primary': terminal, **crossings}


def framework_for_seed(framework, project, reference, seed):
    # The established framework checker pins the original pilot's seed42.
    # Verify the formal seed separately, then reuse its unchanged engine checks.
    assert framework['effective_engine_config']['seed'] == seed
    normalized = copy.deepcopy(framework)
    normalized['effective_engine_config']['seed'] = 42
    return audit_framework(normalized, project, reference)


def audit_rollouts(run, metadata, metrics, rows, tokenizer, score_answer, eos):
    import numpy as np
    spec, initial = metadata['spec'], metadata['initial_adapter_sha256']
    updates, current_hash, vocabulary_size = 0, initial, len(tokenizer)
    evidence_directories = set()
    for index, metric in enumerate(metrics):
        active = metric['active_method']
        assert active in ('rl', 'opd', 'joint')
        assert metric['iteration'] == index+1 and metric['question_offset'] == index*4
        assert metric['questions'] == 4 and metric['responses'] == 16
        assert metric['student_update_owner'] == 'verl engine'
        assert metric['data_cursor_before'] == cursor_state(rows, spec['seed'], index*4)[0]
        assert metric['data_cursor_after'] == cursor_state(rows, spec['seed'], (index+1)*4)[0]
        assert metric['policy_adapter_sha256_before'] == current_hash
        if index:
            assert metric['optimizer_scheduler_sha256_before_update'] == metrics[index-1]['optimizer_scheduler_sha256_after_update']
        directory = contained(run, metric['evidence_directory'])
        assert directory.parent == run/'iterations' and directory.name not in evidence_directories
        evidence_directories.add(directory.name)
        assert read(directory/'metrics.json') == metric
        wanted = {str((directory/name).relative_to(run)) for name in ('rollouts.jsonl', 'token_scores.jsonl')}
        assert set(metric['evidence_sha256']) == wanted
        for relative, expected in metric['evidence_sha256'].items():
            assert sha(contained(run, relative)) == expected
        groups, scores = read_rows(directory/'rollouts.jsonl'), read_rows(directory/'token_scores.jsonl')
        assert len(groups) == 4 and len(scores) == 16
        answers, mixed, zero_ratio_loss, absolute_signal, any_signal = [], 0, 0., 0., False
        inner_generation = 0.
        for within, group in enumerate(groups):
            offset = index*4+within
            cursor, order = cursor_state(rows, spec['seed'], offset)
            training_index = order[cursor['offset']]
            source = {**rows[training_index], 'study_training_index': training_index,
                      'study_epoch': cursor['epoch'], 'study_epoch_offset': cursor['offset'],
                      'study_question_index': offset}
            assert all(group[k] == v for k, v in source.items())
            assert group['prompt_token_ids'] == prompt_ids(source, spec['config'], tokenizer)
            assert group['iteration'] == index and group['question_offset'] == offset
            assert group['policy_step'] == updates and group['batch_seed'] == spec['seed']+offset
            assert group['policy_adapter_sha256'] == current_hash and len(group['answers']) == 4
            rewards = [a['reward'] for a in group['answers']]
            mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
            mixed += 0 < sum(rewards) < 4
            assert group['generation_seconds'] >= 0
            inner_generation += group['generation_seconds']
            for ai, answer in enumerate(group['answers']):
                audit_answer(answer, source['gold'], tokenizer, score_answer, eos, vocabulary_size)
                close(answer['advantage'], (answer['reward']-mean)/(std+1e-8), 1e-10)
                score = scores[within*4+ai]
                assert score['source_row'] == source['source_row'] and score['response'] == ai
                assert score['iteration'] == index and score['policy_step'] == updates
                assert score['policy_adapter_sha256'] == current_hash and score['token_ids'] == answer['token_ids']
                assert score['sequence_advantage'] == answer['advantage']
                old = np.asarray(score['old_student_logprobs'], dtype=np.float32)
                signal = np.asarray(score['effective_token_advantages'], dtype=np.float32)
                assert old.shape == signal.shape == (len(answer['token_ids']),)
                assert np.isfinite(old).all() and np.isfinite(signal).all() and (old <= 1e-6).all()
                if active == 'rl':
                    assert score['teacher_logprobs'] is None
                    expected = np.full_like(old, answer['advantage'])
                else:
                    teacher = np.asarray(score['teacher_logprobs'], dtype=np.float32)
                    assert teacher.shape == old.shape and np.isfinite(teacher).all() and (teacher <= 1e-6).all()
                    expected = teacher-old
                    if active == 'joint':
                        expected = expected + np.float32(answer['advantage'])
                assert np.array_equal(signal, expected), ('Token signal mismatch', index+1, within, ai)
                zero_ratio_loss -= float(signal.astype(np.float64).mean())/16
                absolute_signal += float(np.abs(signal.astype(np.float64)).mean())/16
                any_signal |= bool(np.count_nonzero(signal))
                answers.append(answer)
        assert inner_generation <= metric['generation_wall_seconds']+1e-6
        expected_counts = dict(generated_tokens=sum(len(a['token_ids']) for a in answers),
            correct=sum(a['reward'] for a in answers), invalid_answers=sum(not a['answer_valid'] for a in answers),
            truncated_answers=sum(a['truncated'] for a in answers), mixed_reward_groups=mixed)
        assert all(metric[k] == v for k, v in expected_counts.items())
        assert metric['optimizer_steps'] == int(any_signal) and bool(metric['changed_adapter_tensors']) == any_signal
        assert (metric['policy_adapter_sha256_after'] != current_hash) == any_signal
        assert (metric['optimizer_scheduler_sha256_before_update'] != metric['optimizer_scheduler_sha256_after_update']) == any_signal
        drift = metric['max_preupdate_logp_difference']
        assert 0 <= drift <= 1e-3
        close(metric['loss_before'], zero_ratio_loss, math.expm1(drift)*absolute_signal+2e-5)
        assert metric['backward_responses'] == (16 if any_signal else 0)
        if any_signal:
            assert metric['gradient_norm_before_clipping'] > 0 and metric['skip_reason'] is None
        else:
            assert metric['gradient_norm_before_clipping'] == 0 and metric['skip_reason'] == 'all_effective_signals_zero'
        if active == 'rl':
            assert metric['teacher_scoring_seconds'] == metric['teacher_scoring_peak_allocated_gib'] == 0
        else:
            assert metric['teacher_scoring_seconds'] > 0
        for key in ('generation_wall_seconds', 'old_policy_scoring_seconds', 'teacher_scoring_seconds',
                    'backward_and_optimizer_seconds', 'iteration_audit_seconds', 'checkpoint_payload_write_seconds'):
            assert metric[key] >= 0
        close(metric['training_core_seconds'], sum(metric[k] for k in ('generation_wall_seconds',
            'old_policy_scoring_seconds', 'teacher_scoring_seconds', 'backward_and_optimizer_seconds')))
        assert metric['charged_iteration_seconds'] >= metric['iteration_precheckpoint_wall_seconds'] >= metric['training_core_seconds']
        updates += metric['optimizer_steps']
        assert metric['actual_updates_after'] == updates
        current_hash = metric['policy_adapter_sha256_after']
    assert {p.name for p in (run/'iterations').iterdir()} == evidence_directories, 'Uncommitted iteration evidence'
    return updates, current_hash


def audit_sessions(sessions, metrics, states, metadata, summary, project, reference, export_count):
    assert sessions and sessions[-1]['status'] == 'complete'
    previous_end = 0
    seed = metadata['spec']['seed']
    for index, session in enumerate(sessions):
        assert session['status'] == ('complete' if index == len(sessions)-1 else 'paused_at_boundary')
        start, end = session['start_iteration'], session['end_iteration']
        assert start == previous_end and start <= end <= len(metrics)
        assert session['resume'] == (index > 0)
        assert not session['unaccounted_prior_failure_detected']
        assert session['native_optimizer_resume_verified'] == session['controller_rng_resume_verified'] == session['resume']
        assert session['schedule_after'] == states[end]
        assert session['base_sha256_after'] == metadata['base_sha256_before']
        assert session['teacher_sha256_after'] == metadata['teacher_sha256_before']
        for name in ('framework_at_start', 'framework_at_end'):
            framework_for_seed(session[name], project, reference, seed)
        resume, fresh = int(session['resume']), int(not session['resume'])
        assert session['framework_at_start']['call_counts_this_process'] == dict(
            train_batch=0, infer_batch=0, generate=0, save_checkpoint=0, load_checkpoint=resume)
        for iteration in range(start, end):
            completed = iteration-start+1
            expected = dict(train_batch=sum(m['optimizer_steps'] for m in metrics[start:iteration+1]),
                            infer_batch=completed, generate=4*completed,
                            save_checkpoint=completed-1+fresh, load_checkpoint=resume)
            assert metrics[iteration]['framework_calls_this_process'] == expected
        expected_end = dict(train_batch=sum(m['optimizer_steps'] for m in metrics[start:end]),
            infer_batch=end-start, generate=4*(end-start), save_checkpoint=end-start+fresh,
            load_checkpoint=resume+(export_count+1 if session['status'] == 'complete' else 0))
        assert session['framework_at_end']['call_counts_this_process'] == expected_end
        for key in ('student_load_seconds', 'teacher_load_seconds', 'pretraining_audit_seconds',
                    'final_audit_seconds', 'pause_audit_seconds', 'session_wall_seconds'):
            assert session[key] >= 0
        assert session['session_wall_seconds'] >= sum(m['charged_iteration_seconds'] for m in metrics[start:end])
        previous_end = end
    assert previous_end == len(metrics)
    assert summary['framework'] == sessions[-1]['framework_at_end']


def audit_native(run, metadata, metrics, summary, rows, states):
    import torch
    expected_boundaries, expected_selection = expected_retention(states)
    observed, directories, markers = [], {}, {}
    assert not list((run/'checkpoints').glob('.pending_*')), 'Uncommitted native checkpoint'
    for path in sorted((run/'checkpoints').glob('boundary_*/complete.json')):
        marker = read(path)
        boundary = marker['boundary']
        assert boundary in expected_boundaries and boundary not in markers
        assert marker['identity'] == metadata['identity']
        markers[boundary], directories[boundary] = marker, path.parent
        claimed = marker['checkpoint_files_sha256']
        actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file() and p != path}
        assert actual == set(claimed) and 'controller.pt' in claimed
        for name, expected in claimed.items():
            assert sha(contained(path.parent, name)) == expected, ('Native file digest', name)
        expected_evidence = {'metadata.json': sha(run/'metadata.json'), **metrics[boundary-1]['evidence_sha256']}
        assert marker['evidence_sha256'] == expected_evidence
        state = torch.load(path.parent/'controller.pt', map_location='cpu', weights_only=False)
        assert state['identity'] == metadata['identity']
        assert state['checkpoint_owner'] == 'verl engine' and state['controller_owner'] == 'train_verl_study.py'
        assert 'adapter' not in state and 'optimizer' not in state
        assert state['next_iteration'] == boundary and state['next_question_offset'] == boundary*4
        assert state['prompts_per_iteration'] == 4
        assert state['data_cursor'] == cursor_state(rows, metadata['spec']['seed'], boundary*4)[0]
        assert state['actual_updates'] == marker['actual_updates'] == states[boundary]['actual_updates']
        assert state['schedule_before'] == states[boundary-1] and marker['schedule_after'] == states[boundary]
        assert all(metrics[boundary-1][k] == v for k, v in marker['budget_accounting'].items())
        assert marker['checkpoint_payload_write_seconds'] == metrics[boundary-1]['checkpoint_payload_write_seconds']
        assert len(state['history']) == boundary
        for previous, current in zip(state['history'], metrics[:boundary]):
            assert all(current[k] == v for k, v in previous.items()), 'Committed history mismatch'
        assert state['history'][:-1] == metrics[:boundary-1]
        expected_adapter = metrics[boundary-1]['policy_adapter_sha256_after']
        assert state['adapter_sha256'] == marker['adapter_sha256'] == expected_adapter
        assert len(state['rng']['torch_cuda']) == 1 and rng_sha(state['rng']) == state['rng_sha256']
        native_model = torch.load(path.parent/'engine/model_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
        assert native_model and all('lora_A.' in key or 'lora_B.' in key for key in native_model)
        normalized = {key.replace('.default.', '.'): tensor for key, tensor in native_model.items()}
        assert len(normalized) == len(native_model) and state_sha(normalized) == expected_adapter
        del native_model, normalized
        optimizer = torch.load(path.parent/'engine/optim_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
        extra = torch.load(path.parent/'engine/extra_state_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
        optimizer_sha = tree_sha({'optimizer': optimizer, 'scheduler': extra['lr_scheduler']})
        assert optimizer_sha == state['optimizer_scheduler_sha256'] == metrics[boundary-1]['optimizer_scheduler_sha256_after_update']
        assert extra['lr_scheduler']['last_epoch'] == state['actual_updates']
        if boundary == len(metrics):
            assert optimizer_sha == summary['final_optimizer_scheduler_sha256']
        else:
            assert optimizer_sha == metrics[boundary]['optimizer_scheduler_sha256_before_update']
        assert bool(optimizer['state']) == bool(state['actual_updates'])
        for group in optimizer['param_groups']:
            assert group['lr'] == 1e-5 and group['weight_decay'] == 0 and group['eps'] == 1e-8
            assert tuple(group['betas']) == (.9, .999)
        for item in optimizer['state'].values():
            assert float(item['step']) == state['actual_updates']
            assert torch.isfinite(item['exp_avg']).all() and torch.isfinite(item['exp_avg_sq']).all()
        observed.append(dict(boundary=boundary, adapter_sha256=expected_adapter,
            optimizer_scheduler_sha256=optimizer_sha, rng_sha256=state['rng_sha256'],
            files_hashed=len(claimed), checkpoint_bytes=sum(contained(path.parent,p).stat().st_size for p in claimed),
            optimizer_parameter_states=len(optimizer['state']), actual_updates=state['actual_updates'],
            complete_marker_sha256=sha(path), evidence_files_sha256={
                **{str((path.parent/name).relative_to(run)): digest for name,digest in claimed.items()},
                str(path.relative_to(run)): sha(path)}))
        del optimizer, extra, state
    assert set(markers) == expected_boundaries, 'First-crossing or latest-two checkpoint missing'
    assert {p.name for p in (run/'checkpoints').glob('boundary_*')} == {p.name for p in directories.values()}
    latest, primary = read(run/'checkpoints/latest.json'), read(run/'checkpoints/primary.json')
    terminal = len(metrics)
    assert latest == {'directory': directories[terminal].name, **markers[terminal]}
    assert primary == dict(directory=directories[terminal].name, boundary=terminal,
        charged_seconds=states[terminal]['charged_seconds'], adapter_sha256=markers[terminal]['adapter_sha256'],
        actual_updates=states[terminal]['actual_updates'])
    selections = read(run/'checkpoint_selections.json')
    assert set(selections) == set(expected_selection)
    assert selections['primary'] == summary['selected_primary']
    output = {}
    for label, boundary in expected_selection.items():
        selected = selections[label]
        assert selected['boundary'] == boundary and selected['selection_purpose'] == 'predeclared_time_endpoint'
        assert selected['eligible_for_formal_evaluation'] is True and selected['diagnostic_only'] is False
        expected_directory = run/'adapter_final' if label == 'primary' else run/'adapter_milestones'/label
        assert Path(selected['adapter_directory']).resolve() == expected_directory.resolve()
        assert Path(selected['checkpoint_directory']).resolve() == directories[boundary].resolve()
        assert selected['adapter_sha256'] == markers[boundary]['adapter_sha256'] == adapter_sha(expected_directory)
        assert selected['charged_seconds'] == states[boundary]['charged_seconds']
        assert selected['actual_updates'] == states[boundary]['actual_updates'] and selected['schedule'] == states[boundary]
        output[label] = {**selected, 'adapter_files_sha256': {
            str(p.relative_to(expected_directory)): sha(p) for p in sorted(expected_directory.rglob('*')) if p.is_file()}}
    return observed, output


def audit_run(project, run_name):
    from transformers import AutoTokenizer, GenerationConfig
    assert Path(run_name).name == run_name and run_name not in ('.', '..')
    exp, run = project/'experiments', project/'experiments/runs'/run_name
    config, protocol, reference = read(exp/'configs/pilot.json'), read(exp/'verl_study/protocol.json'), read(exp/'verl/protocol.json')
    metadata, summary = read(run/'metadata.json'), read(run/'summary.json')
    metrics, sessions = read_rows(run/'metrics.jsonl'), read_rows(run/'sessions.jsonl')
    spec = metadata['spec']
    finite([summary, metrics, sessions, spec])
    assert metadata['identity'] == summary['identity'] == canonical_sha(spec)
    assert protocol['status'] == 'frozen' and spec['protocol'] == protocol
    helper_evidence = [r for r in protocol['evidence_provenance']['auditors']
                       if Path(r['path']).name == 'audit_verl_pilot.py']
    assert len(helper_evidence) == 1
    assert sha(Path(__file__).with_name('audit_verl_pilot.py')) == helper_evidence[0]['sha256']
    assert spec['protocol_sha256'] == sha(exp/'verl_study/protocol.json')
    assert spec['config'] == config and spec['config_sha256'] == protocol['source_config_sha256'] == sha(exp/'configs/pilot.json')
    assert protocol['source_verl_protocol_sha256'] == sha(exp/'verl/protocol.json')
    assert spec['reference_protocol_sha256'] == sha(exp/'configs/repeated_pilot_v1.json')
    for mapping in (spec['source_sha256'], protocol['frozen_source_sha256']):
        for relative, expected in mapping.items():
            assert sha(contained(exp, relative)) == expected, ('Source drift', relative)
    assert spec['verl_commit'] == reference['verl_commit'] == protocol['frozen_verl_commit']
    assert subprocess.check_output(['git','-C',str(project/'external/verl'),'rev-parse','HEAD'], text=True).strip() == spec['verl_commit']
    diff = subprocess.check_output(['git','-C',str(project/'external/verl'),'diff','--binary','HEAD'])
    assert hashlib.sha256(diff).hexdigest() == spec['verl_tracked_diff_sha256'] == protocol['frozen_verl_tracked_diff_sha256']
    for package, expected in protocol['frozen_versions'].items():
        assert importlib.metadata.version(package) == expected, ('Environment drift', package)
    assert all(importlib.metadata.version(p) == v for p, v in spec['versions'].items())
    assert spec['python_executable'] == protocol['frozen_python_executable'] == sys.executable
    method, seed = spec['method'], spec['seed']
    assert method in protocol['methods'] and seed in protocol['seeds']
    planned = [r for r in protocol['run_order']['runs'] if r['name'] == run_name]
    assert len(planned) == 1 and planned[0]['method'] == method and planned[0]['seed'] == seed
    assert spec['mode'] == summary['mode'] == 'budgeted' and summary['method'] == method
    assert spec['training_dataset_size'] == 6961 and spec['initial_adapter_seed'] == 42
    assert spec['prompts_per_iteration'] == spec['group_size'] == 4
    for key, expected in dict(backend='fsdp', coefficient=1., clip_epsilon=.2, gradient_clip_max_norm=1.,
        learning_rate=1e-5, weight_decay=0., reference_kl_beta=0., eos_in_loss=True, prompt_in_loss=False,
        microbatch_responses=1, optimizer_epochs_per_batch=1, scheduler_nominal_total_training_steps=8).items():
        assert spec[key] == expected
    assert spec['gpu'] == 'NVIDIA GeForce RTX 5090'
    assert summary['versions'] == spec['versions']
    splits = {}
    for split, count in (('train',6961),('dev',512)):
        path = contained(exp, config['data'][split+'_file'])
        assert sha(path) == config['data'][split+'_sha256']
        splits[split] = read_rows(path)
        assert len(splits[split]) == len({r['question_sha256'] for r in splits[split]}) == count
    assert not ({r['question_sha256'] for r in splits['train']} & {r['question_sha256'] for r in splits['dev']})
    for name, expected in config['reward']['code_sha256'].items():
        assert sha(contained(exp/'reward', name)) == expected
    sys.path.insert(0, str(exp/'reward'))
    from text_answer_reward import score_answer
    tokenizer = AutoTokenizer.from_pretrained(config['models']['student']['local_snapshot'], local_files_only=True, trust_remote_code=False)
    generation = GenerationConfig.from_pretrained(config['models']['student']['local_snapshot'], local_files_only=True)
    eos = set(generation.eos_token_id if isinstance(generation.eos_token_id, list) else [generation.eos_token_id])
    assert spec['initial_cursor'] == cursor_state(splits['train'], seed, 0)[0]
    assert metadata['verified_sampling'] == dict(do_sample=True, temperature=1., top_p=1., top_k=0, min_p=0., max_new_tokens=1024)
    assert spec['initial_adapter_file_sha256'] == reference['initial_adapter_file_sha256'] == sha(Path(reference['initial_adapter'])/'adapter_model.safetensors')
    initial = adapter_sha(Path(reference['initial_adapter']))
    assert metadata['initial_adapter_sha256'] == summary['initial_adapter_sha256'] == adapter_sha(run/'adapter_initial') == initial
    # Bind runtime frozen-base hashes to a pre-study, independently audited run.
    reference_path = exp/'runs/verl_study_rl_then_opd_check_01/metadata.json'
    frozen = [r for r in protocol['evidence_provenance']['run_evidence'] if Path(r['path']) == reference_path]
    assert len(frozen) == 1 and sha(reference_path) == frozen[0]['sha256']
    fixed = read(reference_path)
    assert metadata['base_sha256_before'] == fixed['base_sha256_before']
    assert metadata['teacher_sha256_before'] == (None if method == 'rl' else fixed['teacher_sha256_before'])
    assert summary['base_sha256_before'] == summary['base_sha256_after'] == metadata['base_sha256_before']
    assert summary['teacher_sha256_before'] == summary['teacher_sha256_after'] == metadata['teacher_sha256_before']
    for key in ('base_weights_unchanged', 'adapter_save_reload_exact', 'native_adapter_reload_exact',
                'native_optimizer_scheduler_reload_exact', 'controller_rng_restore_exact', 'budget_accounting_valid',
                'both_sequential_phases_executed'):
        assert summary[key] is True
    assert summary['teacher_weights_unchanged'] == summary['teacher_gradients_absent'] == (method != 'rl')
    assert summary['unaccounted_failed_compute'] is False and summary['status'] == 'complete'
    states = reconstruct_schedules(spec['schedule_initial'], metrics)
    updates, terminal_hash = audit_rollouts(run, metadata, metrics, splits['train'], tokenizer, score_answer, eos)
    assert summary['iterations'] == len(metrics) and summary['actual_optimizer_steps'] == updates
    assert summary['skipped_iterations'] == len(metrics)-updates
    assert summary['final_adapter_sha256'] == summary['terminal_adapter_sha256'] == terminal_hash
    final = states[-1]
    assert all(summary['budget'][k] == v for k, v in final.items())
    budget = summary['budget']
    assert budget['budget_status'] == 'within_allowance' and budget['budgeted_run'] is True and budget['within_budget'] is True
    assert budget['overrun_seconds'] == budget['overrun_fraction'] == 0 and budget['overrun_exceeds_tolerance'] is False
    assert budget['unused_allowance_seconds'] == 7200-final['charged_seconds']
    close(budget['realized_phase1_fraction'], final['phase1_seconds']/final['charged_seconds'])
    sequential = '_then_' in method
    assert budget['both_sequential_phases_executed'] == (True if sequential else None)
    assert budget['midpoint_overshoot_seconds'] == (final['switch_at_seconds']-3600 if sequential else None)
    assert budget['active_method_next'] == method.split('_then_')[-1]
    assert summary['data_cursor'] == cursor_state(splits['train'], seed, len(metrics)*4)[0]
    close(summary['charged_training_seconds'], sum(m['charged_iteration_seconds'] for m in metrics))
    for key in ('questions','responses','generated_tokens','correct','mixed_reward_groups','invalid_answers',
        'truncated_answers','generation_wall_seconds','old_policy_scoring_seconds','teacher_scoring_seconds',
        'backward_and_optimizer_seconds','training_core_seconds','iteration_audit_seconds','checkpoint_payload_write_seconds'):
        close(summary[key], sum(m[key] for m in metrics))
    assert summary['peak_iteration_torch_allocated_gib'] == max(m['peak_torch_allocated_gib'] for m in metrics)
    tails = read_rows(run/'commit_tails.jsonl')
    assert [t['iteration'] for t in tails] == list(range(1,len(metrics)+1))
    assert all(t['uncharged_commit_tail_seconds'] >= 0 for t in tails)
    close(summary['uncharged_commit_tail_seconds'], sum(t['uncharged_commit_tail_seconds'] for t in tails))
    native, selected = audit_native(run, metadata, metrics, summary, splits['train'], states)
    audit_sessions(sessions, metrics, states, metadata, summary, project, reference, len(selected))
    framework_for_seed(metadata['framework'], project, reference, seed)
    assert all(v == 0 for v in metadata['framework']['call_counts_this_process'].values())
    progress = read(run/'progress.json')
    assert progress['status'] == 'complete' and progress['completed_iterations'] == len(metrics)
    assert progress['actual_optimizer_steps'] == updates and progress['identity'] == metadata['identity']
    assert progress['schedule'] == budget and progress['data_cursor'] == summary['data_cursor']
    assert progress['stop_reason'] == 'budget_or_validation_complete'
    evidence_files = {name: sha(run/name) for name in ('metadata.json','summary.json','metrics.jsonl',
        'sessions.jsonl','checkpoint_selections.json','commit_tails.jsonl','progress.json',
        'checkpoints/latest.json','checkpoints/primary.json')}
    for metric in metrics:
        evidence_files.update(metric['evidence_sha256'])
        relative = str(Path(metric['evidence_directory'])/'metrics.json')
        evidence_files[relative] = sha(run/relative)
    for boundary in native:
        evidence_files.update(boundary['evidence_files_sha256'])
    for selection in selected.values():
        base = Path(selection['adapter_directory']).relative_to(run)
        evidence_files.update({str(base/name): digest for name,digest in selection['adapter_files_sha256'].items()})
    for path in (run/'adapter_initial').rglob('*'):
        if path.is_file():
            evidence_files[str(path.relative_to(run))] = sha(path)
    return dict(status='pass', gpu_operations=0, run_name=run_name, run_directory=str(run), method=method, seed=seed,
        identity=metadata['identity'], eligible_for_formal_evaluation=True,
        protocol_sha256=spec['protocol_sha256'], summary_sha256=sha(run/'summary.json'),
        metadata_sha256=sha(run/'metadata.json'), metrics_sha256=sha(run/'metrics.jsonl'),
        sessions_sha256=sha(run/'sessions.jsonl'), checkpoint_selections_sha256=sha(run/'checkpoint_selections.json'),
        selected_primary=selected['primary'], milestone_050=selected['milestone_050'], all_selected_checkpoints=selected,
        evidence_files_sha256=evidence_files,
        rounds=len(metrics), actual_updates=updates, skipped_rounds=len(metrics)-updates,
        charged_seconds=final['charged_seconds'], unused_allowance_seconds=7200-final['charged_seconds'],
        uncharged_commit_tail_seconds=summary['uncharged_commit_tail_seconds'],
        all_session_wall_seconds=sum(s['session_wall_seconds'] for s in sessions),
        phase1_seconds=final['phase1_seconds'], phase2_seconds=final['phase2_seconds'],
        switch_after_round=final['switch_round'], native_checkpoints=native,
        all_tokens_rewards_signals_and_cursors_verified=True, adapter_optimizer_chain_verified=True,
        frozen_weight_evidence='Runtime full base/teacher hashes equal the frozen pre-study reference and session-end hashes; native adapter/optimizer/RNG bytes independently checked on CPU.',
        limitations='Log probabilities are validated as finite recorded evidence and objective signals are recomputed; inference is not rerun by this CPU audit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        result = audit_run(args.project_root.resolve(), args.run_name)
    except Exception as error:
        result = dict(status='fail', gpu_operations=0, run_name=args.run_name,
                      error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    result.update(audit_seconds=time.perf_counter()-started, audit_script_sha256=sha(Path(__file__)),
                  hash_helper_sha256=sha(Path(__file__).with_name('audit_verl_pilot.py')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+'.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
