"""Independent CPU audit of full-dev calibration and sequential switch/resume checks.

Only the requested JSON output is written. No training source/evidence is edited;
no inference, optimizer action, CUDA tensor allocation, or GPU API is called.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

from audit_verl_pilot import (adapter_sha, audit_answer, audit_framework, canonical_sha,
    close, contained, finite, prompt_ids, read, read_rows, rng_sha, sha, state_sha, tree_sha)


def cursor_state(rows, seed, offset):
    assert 0 <= offset < len(rows)
    order = list(range(len(rows)))
    random.Random(f'rl-opd-study:{seed}:0').shuffle(order)
    state = {'seed': seed, 'epoch': 0, 'offset': offset, 'questions_seen': offset,
             'dataset_size': len(rows),
             'order_sha256': hashlib.sha256(json.dumps(order, separators=(',', ':')).encode()).hexdigest(),
             'shuffle_rule': "Python random.Random('rl-opd-study:{seed}:{epoch}').shuffle(range(N))"}
    return state, order


def expected_schedule(initial, metrics):
    """Derive this two-round forced schedule without importing scheduling code."""
    value = dict(initial)
    assert value['mode'] == 'validation' and value['validation_rounds'] == 2
    assert value['validation_switch_after'] == 1 and value['completed_rounds'] == 0
    assert value['charged_seconds'] == value['actual_updates'] == value['phase1_seconds'] == value['phase2_seconds'] == 0
    assert value['switch_round'] is None and value['switch_at_seconds'] is None
    phases = value['method'].split('_then_')
    assert phases in (['opd', 'rl'], ['rl', 'opd'])
    for i, metric in enumerate(metrics):
        duration = metric['charged_iteration_seconds']
        assert math.isfinite(duration) and duration > 0
        assert metric['active_method'] == phases[i] and metric['phase_index'] == i
        close(metric['charged_seconds_before'], value['charged_seconds'])
        value['charged_seconds'] += duration
        close(metric['charged_seconds_after'], value['charged_seconds'])
        value['completed_rounds'] += 1
        value['actual_updates'] += metric['optimizer_steps']
        value[f'phase{i+1}_seconds'] += duration
        assert metric['phase_switch_at_boundary'] == (i == 0)
        assert metric['next_active_method'] == phases[1]
        assert metric['primary_endpoint_eligible'] is True and metric['overrun_seconds'] == 0
        before = metric['charged_seconds_before']
        assert metric['crossed_midpoint'] == (before < value['budget_seconds']/2 <= value['charged_seconds'])
        assert metric['crossed_budget'] == (before < value['budget_seconds'] <= value['charged_seconds'])
        if i == 0:
            value['switch_round'] = 1
            value['switch_at_seconds'] = value['charged_seconds']
    return value


def audit_native(run, metadata, metrics, summary, rows):
    import torch
    boundaries = []
    for path in sorted((run/'checkpoints').glob('boundary_*/complete.json')):
        marker = read(path)
        boundary = marker['boundary']
        assert boundary in (1, 2) and marker['identity'] == metadata['identity']
        claimed = marker['checkpoint_files_sha256']
        actual = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file() and p != path}
        assert actual == set(claimed) and 'controller.pt' in claimed
        for name, expected in claimed.items():
            assert sha(contained(path.parent, name)) == expected
        for name, expected in marker['evidence_sha256'].items():
            assert sha(contained(run, name)) == expected
        # These are locally generated, checksummed checkpoints, loaded on CPU only.
        state = torch.load(path.parent/'controller.pt', map_location='cpu', weights_only=False)
        assert state['identity'] == metadata['identity']
        assert state['checkpoint_owner'] == 'verl engine' and state['controller_owner'] == 'train_verl_study.py'
        assert 'adapter' not in state and 'optimizer' not in state
        assert state['next_iteration'] == boundary and state['next_question_offset'] == boundary*4
        assert state['prompts_per_iteration'] == 4
        assert state['data_cursor'] == cursor_state(rows, metadata['spec']['seed'], boundary*4)[0]
        assert state['actual_updates'] == marker['actual_updates'] == sum(m['optimizer_steps'] for m in metrics[:boundary])
        assert state['schedule_before'] == expected_schedule(metadata['spec']['schedule_initial'], metrics[:boundary-1])
        assert marker['schedule_after'] == expected_schedule(metadata['spec']['schedule_initial'], metrics[:boundary])
        assert all(metrics[boundary-1][key] == value for key, value in marker['budget_accounting'].items())
        assert marker['checkpoint_payload_write_seconds'] == metrics[boundary-1]['checkpoint_payload_write_seconds']
        assert len(state['history']) == boundary
        for previous, current in zip(state['history'], metrics[:boundary]):
            assert all(current[key] == value for key, value in previous.items())
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
        if boundary == 1:
            assert optimizer_sha == metrics[1]['optimizer_scheduler_sha256_before_update']
            assert expected_adapter == metrics[1]['policy_adapter_sha256_before']
        else:
            assert optimizer_sha == summary['final_optimizer_scheduler_sha256']
        assert bool(optimizer['state']) == bool(state['actual_updates'])
        for group in optimizer['param_groups']:
            assert group['lr'] == 1e-5 and group['weight_decay'] == 0 and group['eps'] == 1e-8
            assert tuple(group['betas']) == (.9, .999)
        for item in optimizer['state'].values():
            assert float(item['step']) == state['actual_updates']
            assert torch.isfinite(item['exp_avg']).all() and torch.isfinite(item['exp_avg_sq']).all()
        boundaries.append({'boundary': boundary, 'adapter_sha256': expected_adapter,
            'optimizer_scheduler_sha256': optimizer_sha, 'rng_sha256': state['rng_sha256'],
            'files_hashed': len(claimed), 'checkpoint_bytes': sum(contained(path.parent, p).stat().st_size for p in claimed),
            'optimizer_parameter_states': len(optimizer['state'])})
        del optimizer, extra, state
    assert sorted(b['boundary'] for b in boundaries) == [1, 2]
    latest, primary = read(run/'checkpoints/latest.json'), read(run/'checkpoints/primary.json')
    assert latest['boundary'] == primary['boundary'] == 2 and latest['identity'] == metadata['identity']
    for pointer in (latest, primary):
        marker = read(contained(run/'checkpoints', pointer['directory'])/'complete.json')
        assert marker['boundary'] == pointer['boundary'] and marker['adapter_sha256'] == pointer['adapter_sha256']
    selections = read(run/'checkpoint_selections.json')
    assert set(selections) == {'primary'} and selections['primary'] == summary['selected_primary']
    selected = selections['primary']
    assert selected['boundary'] == 2 and selected['selection_purpose'] == 'engineering_validation'
    assert selected['eligible_for_formal_evaluation'] is False and selected['diagnostic_only'] is False
    assert selected['adapter_sha256'] == summary['final_adapter_sha256'] == adapter_sha(Path(selected['adapter_directory']))
    assert Path(selected['adapter_directory']).resolve() == (run/'adapter_final').resolve()
    return boundaries


def audit_sequential_run(project, method, config, source_protocol, rows, tokenizer, score_answer, eos):
    import numpy as np
    exp = project/'experiments'
    run = exp/'runs'/f'verl_study_{method}_check_01'
    metadata, summary = read(run/'metadata.json'), read(run/'summary.json')
    metrics, sessions = read_rows(run/'metrics.jsonl'), read_rows(run/'sessions.jsonl')
    spec = metadata['spec']
    finite([summary, metrics, sessions])
    assert metadata['identity'] == summary['identity'] == canonical_sha(spec)
    assert spec['config'] == config and spec['config_sha256'] == sha(exp/'configs/pilot.json')
    protocol_candidates = [p for p in (exp/'verl_study/protocol_validation.json', exp/'verl_study/protocol.json')
                           if p.is_file() and sha(p) == spec['protocol_sha256']]
    assert protocol_candidates and all(read(p) == spec['protocol'] for p in protocol_candidates)
    assert spec['protocol']['source_verl_protocol_sha256'] == sha(exp/'verl/protocol.json')
    assert spec['reference_protocol_sha256'] == sha(exp/'configs/repeated_pilot_v1.json')
    for relative, expected in spec['source_sha256'].items():
        assert sha(contained(exp, relative)) == expected
    assert spec['verl_commit'] == source_protocol['verl_commit']
    assert subprocess.check_output(['git','-C',str(project/'external/verl'),'rev-parse','HEAD'], text=True).strip() == spec['verl_commit']
    diff = subprocess.check_output(['git','-C',str(project/'external/verl'),'diff','--binary','HEAD'])
    assert hashlib.sha256(diff).hexdigest() == spec['verl_tracked_diff_sha256']
    assert all(importlib.metadata.version(p) == v for p, v in spec['versions'].items())
    assert spec['method'] == summary['method'] == method
    assert spec['mode'] == summary['mode'] == 'validation' and spec['seed'] == 42
    assert spec['training_dataset_size'] == 6961 and spec['initial_adapter_seed'] == 42
    assert spec['prompts_per_iteration'] == spec['group_size'] == 4
    for key, expected in dict(backend='fsdp', coefficient=1., clip_epsilon=.2,
            gradient_clip_max_norm=1., learning_rate=1e-5, weight_decay=0., reference_kl_beta=0.,
            eos_in_loss=True, prompt_in_loss=False, microbatch_responses=1, optimizer_epochs_per_batch=1).items():
        assert spec[key] == expected
    assert spec['gpu'] == 'NVIDIA GeForce RTX 5090'
    assert spec['initial_cursor'] == cursor_state(rows, spec['seed'], 0)[0]
    assert metadata['verified_sampling'] == dict(do_sample=True, temperature=1., top_p=1., top_k=0, min_p=0., max_new_tokens=1024)
    initial = adapter_sha(Path(source_protocol['initial_adapter']))
    assert metadata['initial_adapter_sha256'] == summary['initial_adapter_sha256'] == adapter_sha(run/'adapter_initial') == initial
    assert spec['initial_adapter_file_sha256'] == source_protocol['initial_adapter_file_sha256']
    assert summary['status'] == 'complete' and len(metrics) == 2
    assert summary['base_sha256_before'] == summary['base_sha256_after'] == metadata['base_sha256_before']
    assert summary['teacher_sha256_before'] == summary['teacher_sha256_after'] == metadata['teacher_sha256_before']
    assert summary['teacher_sha256_before'] is not None
    for key in ('base_weights_unchanged','teacher_weights_unchanged','teacher_gradients_absent',
                'adapter_save_reload_exact','native_adapter_reload_exact','native_optimizer_scheduler_reload_exact',
                'controller_rng_restore_exact','both_sequential_phases_executed'):
        assert summary[key] is True
    assert summary['budget_accounting_valid'] is False and summary['unaccounted_failed_compute'] is False
    assert len(sessions) == 2
    assert [s['status'] for s in sessions] == ['paused_at_boundary', 'complete']
    assert [(s['start_iteration'], s['end_iteration'], s['resume']) for s in sessions] == [(0,1,False),(1,2,True)]
    for session in sessions:
        assert session['base_sha256_after'] == metadata['base_sha256_before']
        assert session['teacher_sha256_after'] == metadata['teacher_sha256_before']
        assert session['native_optimizer_resume_verified'] == session['controller_rng_resume_verified'] == session['resume']
        assert not session['unaccounted_prior_failure_detected']
        assert session['schedule_after'] == expected_schedule(spec['schedule_initial'], metrics[:session['end_iteration']])
        for field in ('framework_at_start','framework_at_end'):
            audit_framework(session[field], project, source_protocol)
        counts = session['framework_at_end']['call_counts_this_process']
        metric = metrics[session['start_iteration']]
        assert counts['generate'] == 4 and counts['infer_batch'] == 1
        assert counts['train_batch'] == metric['optimizer_steps']
        assert counts['save_checkpoint'] == 1 + int(not session['resume'])
        assert counts['load_checkpoint'] == int(session['resume']) + 2*int(session['status'] == 'complete')
    audit_framework(metadata['framework'], project, source_protocol)
    audit_framework(summary['framework'], project, source_protocol)
    vocabulary_size = len(tokenizer)
    _, order = cursor_state(rows, spec['seed'], 0)
    updates, current_hash, first_groups = 0, initial, None
    for index, metric in enumerate(metrics):
        assert metric['active_method'] == method.split('_then_')[index]
        active = metric['active_method']
        assert metric['iteration'] == index+1 and metric['question_offset'] == index*4
        assert metric['questions'] == 4 and metric['responses'] == 16 and metric['student_update_owner'] == 'verl engine'
        assert metric['data_cursor_before'] == cursor_state(rows, spec['seed'], index*4)[0]
        assert metric['data_cursor_after'] == cursor_state(rows, spec['seed'], (index+1)*4)[0]
        assert metric['policy_adapter_sha256_before'] == current_hash
        if index:
            assert metric['optimizer_scheduler_sha256_before_update'] == metrics[index-1]['optimizer_scheduler_sha256_after_update']
        calls = metric['framework_calls_this_process']
        assert calls == dict(train_batch=metric['optimizer_steps'], infer_batch=1, generate=4,
                             save_checkpoint=int(index == 0), load_checkpoint=int(index == 1))
        directory = contained(run, metric['evidence_directory'])
        assert read(directory/'metrics.json') == metric
        for relative, expected in metric['evidence_sha256'].items():
            assert sha(contained(run, relative)) == expected
        groups, scores = read_rows(directory/'rollouts.jsonl'), read_rows(directory/'token_scores.jsonl')
        assert len(groups) == 4 and len(scores) == 16
        if index == 0:
            first_groups = groups
        answers, mixed, loss_zero_ratio, abs_signal, any_signal = [], 0, 0., 0., False
        inner_generation = 0.
        for within, group in enumerate(groups):
            offset = index*4+within
            training_index = order[offset]
            source = {**rows[training_index], 'study_training_index': training_index,
                      'study_epoch': 0, 'study_epoch_offset': offset, 'study_question_index': offset}
            assert all(group[key] == value for key, value in source.items())
            assert group['prompt_token_ids'] == prompt_ids(source, config, tokenizer)
            assert group['iteration'] == index and group['question_offset'] == offset
            assert group['policy_step'] == updates and group['batch_seed'] == spec['seed']+offset
            assert group['policy_adapter_sha256'] == current_hash and len(group['answers']) == 4
            rewards = [a['reward'] for a in group['answers']]
            mean, std = statistics.mean(rewards), statistics.pstdev(rewards)
            mixed += 0 < sum(rewards) < 4
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
                assert np.array_equal(signal, expected)
                loss_zero_ratio -= float(signal.astype(np.float64).mean())/16
                abs_signal += float(np.abs(signal.astype(np.float64)).mean())/16
                any_signal |= bool(np.count_nonzero(signal))
                answers.append(answer)
        assert inner_generation <= metric['generation_wall_seconds']+1e-6
        expected_counts = dict(generated_tokens=sum(len(a['token_ids']) for a in answers), correct=sum(a['reward'] for a in answers),
            invalid_answers=sum(not a['answer_valid'] for a in answers), truncated_answers=sum(a['truncated'] for a in answers), mixed_reward_groups=mixed)
        assert all(metric[k] == v for k,v in expected_counts.items())
        assert metric['optimizer_steps'] == int(any_signal) and bool(metric['changed_adapter_tensors']) == any_signal
        assert (metric['policy_adapter_sha256_after'] != current_hash) == any_signal
        assert (metric['optimizer_scheduler_sha256_before_update'] != metric['optimizer_scheduler_sha256_after_update']) == any_signal
        drift = metric['max_preupdate_logp_difference']
        assert 0 <= drift <= 1e-3
        close(metric['loss_before'], loss_zero_ratio, math.expm1(drift)*abs_signal+2e-5)
        assert metric['backward_responses'] == (16 if any_signal else 0)
        if any_signal:
            assert metric['gradient_norm_before_clipping'] > 0 and metric['skip_reason'] is None
        else:
            assert metric['gradient_norm_before_clipping'] == 0 and metric['skip_reason'] == 'all_effective_signals_zero'
        if active == 'rl':
            assert metric['teacher_scoring_seconds'] == metric['teacher_scoring_peak_allocated_gib'] == 0
        else:
            assert metric['teacher_scoring_seconds'] > 0
        close(metric['training_core_seconds'], sum(metric[k] for k in ('generation_wall_seconds',
              'old_policy_scoring_seconds','teacher_scoring_seconds','backward_and_optimizer_seconds')))
        assert metric['charged_iteration_seconds'] >= metric['iteration_precheckpoint_wall_seconds'] >= metric['training_core_seconds']
        updates += metric['optimizer_steps']
        assert metric['actual_updates_after'] == updates
        current_hash = metric['policy_adapter_sha256_after']
    assert summary['iterations'] == 2 and summary['questions'] == 8 and summary['responses'] == 32
    assert summary['actual_optimizer_steps'] == updates and summary['skipped_iterations'] == 2-updates
    assert summary['final_adapter_sha256'] == summary['terminal_adapter_sha256'] == current_hash == adapter_sha(run/'adapter_final')
    final_schedule = expected_schedule(spec['schedule_initial'], metrics)
    assert all(summary['budget'][k] == v for k,v in final_schedule.items())
    assert summary['budget']['budget_status'] == 'validation_only' and summary['budget']['budgeted_run'] is False
    assert summary['data_cursor'] == cursor_state(rows, spec['seed'], 8)[0]
    close(summary['charged_training_seconds'], sum(m['charged_iteration_seconds'] for m in metrics))
    for key in ('questions','responses','generated_tokens','correct','mixed_reward_groups','invalid_answers',
        'truncated_answers','generation_wall_seconds','old_policy_scoring_seconds','teacher_scoring_seconds',
        'backward_and_optimizer_seconds','training_core_seconds','iteration_audit_seconds','checkpoint_payload_write_seconds'):
        close(summary[key], sum(m[key] for m in metrics))
    tails = read_rows(run/'commit_tails.jsonl')
    assert [t['iteration'] for t in tails] == [1,2]
    close(summary['uncharged_commit_tail_seconds'], sum(t['uncharged_commit_tail_seconds'] for t in tails))
    native = audit_native(run, metadata, metrics, summary, rows)
    result = {'status': 'pass', 'run': run.name, 'method': method, 'phase_order': [m['active_method'] for m in metrics],
        'seed': 42, 'rounds': 2, 'actual_updates': updates, 'skipped_rounds': 2-updates,
        'initial_adapter_sha256': initial, 'final_adapter_sha256': current_hash,
        'switch_after_round': 1, 'process_resume_at_switch': True,
        'adapter_and_optimizer_chain_across_switch_verified': True,
        'rl_phase_teacher_scoring_seconds': sum(m['teacher_scoring_seconds'] for m in metrics if m['active_method']=='rl'),
        'data_cursor_seeds_prompts_rewards_token_signals_verified': True,
        'native_checkpoints': native, 'charged_seconds': summary['charged_training_seconds'],
        'training_source_and_environment_verified': True,
        'frozen_weight_evidence': 'Source-verified runtime full base/teacher hashes before/after both sessions; native adapter, optimizer and controller RNG fingerprints independently recomputed on CPU.',
        'metadata_sha256': sha(run/'metadata.json'), 'summary_sha256': sha(run/'summary.json'),
        'metrics_sha256': sha(run/'metrics.jsonl'), 'sessions_sha256': sha(run/'sessions.jsonl'),
        'checkpoint_selections_sha256': sha(run/'checkpoint_selections.json')}
    return result, first_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--scope', choices=('baselines','sequential','all'), default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    result = {'status': 'pass', 'scope': args.scope, 'gpu_operations': 0,
              'audit_script_sha256': sha(Path(__file__)),
              'hash_helper_sha256': sha(Path(__file__).with_name('audit_verl_pilot.py'))}
    if args.scope in ('baselines','all'):
        from audit_verl_study_baselines import audit_baselines
        result['baselines'] = audit_baselines(project)
    if args.scope in ('sequential','all'):
        from transformers import AutoTokenizer, GenerationConfig
        exp = project/'experiments'
        config, protocol = read(exp/'configs/pilot.json'), read(exp/'verl/protocol.json')
        splits = {}
        for split, count in (('train',6961),('dev',512)):
            path = exp/config['data'][split+'_file']
            assert sha(path) == config['data'][split+'_sha256']
            splits[split] = read_rows(path)
            assert len(splits[split]) == len({r['question_sha256'] for r in splits[split]}) == count
        assert not ({r['question_sha256'] for r in splits['train']} & {r['question_sha256'] for r in splits['dev']})
        for name, expected in config['reward']['code_sha256'].items():
            assert sha(exp/'reward'/name) == expected
        sys.path.insert(0, str(exp/'reward'))
        from text_answer_reward import score_answer
        tokenizer = AutoTokenizer.from_pretrained(config['models']['student']['local_snapshot'], local_files_only=True, trust_remote_code=False)
        generation = GenerationConfig.from_pretrained(config['models']['student']['local_snapshot'], local_files_only=True)
        eos_value = generation.eos_token_id
        eos = set(eos_value if isinstance(eos_value,list) else [eos_value])
        outputs, firsts = {}, {}
        for method in ('opd_then_rl','rl_then_opd'):
            outputs[method], firsts[method] = audit_sequential_run(project, method, config, protocol, splits['train'], tokenizer, score_answer, eos)
        pairs = list(zip(firsts['opd_then_rl'], firsts['rl_then_opd']))
        mismatch = []
        for index, (left,right) in enumerate(pairs):
            assert left['source_row'] == right['source_row'] and left['prompt_token_ids'] == right['prompt_token_ids']
            for ai,(a,b) in enumerate(zip(left['answers'],right['answers'])):
                if a['token_ids'] != b['token_ids']:
                    mismatch.append([index,ai])
        result['sequential'] = outputs
        result['first_round_cross_method'] = {'same_initial_adapter': len({r['initial_adapter_sha256'] for r in outputs.values()}) == 1,
            'all_four_prompt_token_ids_equal': True, 'all_sixteen_sampled_token_ids_equal': not mismatch,
            'mismatching_question_response_indices': mismatch}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
