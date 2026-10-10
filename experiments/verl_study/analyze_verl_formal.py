"""Audit and analyze the completed fifteen-run study on CPU.

Requires all 122 supervisor jobs and their recorded evidence. Uses NumPy
for paired bootstrap estimates and exports results without model inference.
"""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

METHODS = ['rl', 'opd', 'joint', 'opd_then_rl', 'rl_then_opd']
SEEDS = [42, 43, 44]
COUNTS = {'dev64': 64, 'dev512': 512, 'test': 1319}
CORE = ['generation_wall_seconds', 'old_policy_scoring_seconds',
        'teacher_scoring_seconds', 'backward_and_optimizer_seconds']
TRAIN_TOTALS = ['questions', 'responses', 'generated_tokens', 'correct', 'mixed_reward_groups',
                'invalid_answers', 'truncated_answers', *CORE, 'training_core_seconds',
                'iteration_audit_seconds', 'checkpoint_payload_write_seconds']
EVAL_COSTS = ['generation_seconds', 'batch_weight_audit_seconds', 'initial_weight_audit_seconds',
              'final_weight_audit_seconds', 'model_and_adapter_load_seconds',
              'known_session_wall_seconds', 'preparation_seconds', 'recovery_validation_seconds']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(value, unicode=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
            ensure_ascii=not unicode, allow_nan=False).encode()).hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), 'Path escapes evidence root: ' + str(relative))
    return path


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7),
            f'{label}: {actual} != {expected}')


class Hashes:
    """Cache reads, detect changed stat signatures, and recheck before export."""
    def __init__(self):
        self.cache = {}
        self.bytes_read = 0

    @staticmethod
    def signature(path):
        value = path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def sha(self, path):
        path = Path(path).resolve()
        signature = self.signature(path)
        if path in self.cache:
            require(self.cache[path][0] == signature, 'Evidence changed during analysis: ' + str(path))
            return self.cache[path][1]
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                h.update(block)
                self.bytes_read += len(block)
        require(self.signature(path) == signature, 'Evidence changed while hashing: ' + str(path))
        self.cache[path] = (signature, h.hexdigest())
        return h.hexdigest()

    def verify(self, path, expected):
        require(self.sha(path) == expected, 'Evidence hash mismatch: ' + str(path))

    def verify_map(self, root, mapping, label):
        require(isinstance(mapping, dict) and mapping, 'Missing evidence manifest: ' + label)
        for relative, expected in mapping.items():
            self.verify(contained(root, relative), expected)

    def unchanged(self):
        for path, (signature, _) in self.cache.items():
            require(self.signature(path) == signature, 'Evidence changed before export: ' + str(path))


def planned_jobs(protocol):
    declared = protocol['run_order']['runs']
    require(protocol['methods'] == METHODS and protocol['seeds'] == SEEDS, 'Unexpected method/seed plan')
    require(len(declared) == 15 and [r['order'] for r in declared] == list(range(1, 16)), 'Incomplete run order')
    require({(r['method'], r['seed']) for r in declared} == set(itertools.product(METHODS, SEEDS)), 'Missing seed/method')
    require(len({r['name'] for r in declared}) == 15, 'Duplicate run name')
    jobs = []
    def add(kind, name, **kw):
        require(Path(name).name == name and name not in ('.', '..'), 'Invalid run name')
        jobs.append(dict(id=f'{len(jobs)+1:03d}_{kind}_{name}', kind=kind, name=name, **kw))
    for run in declared:
        name = run['name']
        add('train', name, method=run['method'], seed=run['seed'], train_name=name)
        add('train_audit', name, train_name=name)
        for split, selection in [('dev64', 'milestone_050'), ('dev512', 'primary')]:
            for kind in ('eval', 'eval_audit'):
                add(kind, name + '_' + split, train_name=name, split=split, selection=selection)
    for kind in ('eval', 'eval_audit'):
        add(kind, 'verl_study_initial_test_01', split='test', selection='initial')
    for run in declared:
        for kind in ('eval', 'eval_audit'):
            add(kind, run['name'] + '_test', train_name=run['name'], split='test', selection='primary')
    require(len(jobs) == 122, 'Unexpected job count')
    return jobs


def validate_state(state, expected_identity):
    require(state['status'] == 'complete', 'Study is incomplete: no formal accuracy analysis is permitted')
    require(not state.get('active_child') and not state.get('current_job'), 'Supervisor still has active work')
    require(state['identity'] == expected_identity and state['identity_sha256'] == canonical(expected_identity),
            'Supervisor identity or frozen plan changed')
    require([e['job'] for e in state['jobs']] == expected_identity['jobs'], 'Supervisor job order differs')
    require(all(e['status'] == 'complete' and e.get('result') for e in state['jobs']), 'Missing completed job evidence')
    return {(e['job']['kind'], e['job']['name']): e for e in state['jobs']}


def audit_pass_in_log(path):
    """Find the top-level final audit JSON amid startup notices and exit marker."""
    text = Path(path).read_text(encoding='utf-8')
    decoder = json.JSONDecoder()
    found = []
    for offset in (i for i, ch in enumerate(text) if ch == '{' and (i == 0 or text[i-1] == '\n')):
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('status') == 'audit_pass':
            found.append(value)
    require(len(found) == 1, 'Missing unique successful evaluation audit in ' + str(path))
    require(found[0]['gpu_operations'] == 0 and 'PROCESS_EXIT_CODE: 0' in text,
            'Evaluation audit did not close successfully')
    return found[0]


def attempt_ledger(project, entry, hashes, authorized_failed_evals=None):
    attempts = entry['attempts']
    require(attempts and [a['number'] for a in attempts] == list(range(1, len(attempts)+1)), 'Missing attempt ledger')
    authorized_failed_evals = authorized_failed_evals or {}
    output = []
    for attempt in attempts:
        status = attempt['status']
        recovery = None
        if status == 'exited' and attempt.get('exit_code') == 1:
            recovery = authorized_failed_evals.get((entry['job']['id'], attempt['number']))
            require(recovery is not None and entry['job']['kind'] == 'eval' and
                    attempt['number'] < len(attempts), 'Unauthorized or final failed evaluation attempt')
            require(recovery['job'] == entry['job'] and recovery['failed_attempt'] == attempt,
                    'Authorized failed evaluation attempt was altered')
        else:
            require(status in ('complete', 'paused', 'paused_before_work', 'supervisor_recovery'), 'Unclean job attempt')
        log = contained(project, attempt['log'])
        if status == 'supervisor_recovery':
            # A completed child still needs its final audit after supervisor recovery.
            require(attempt.get('process_wall_seconds') is None and attempt.get('whole_job_wall_seconds') is None,
                    'Recovered attempt falsely claims exact cost')
            log_hash = hashes.sha(log)  # observed now; not a contemporaneous exit receipt
        else:
            require(attempt['exit_code'] == 0 or recovery is not None,
                    'A failed attempt cannot enter formal accuracy analysis')
            hashes.verify(log, attempt['log_sha256'])
            log_hash = attempt['log_sha256']
        record = {'number': attempt['number'], 'status': status, 'log': str(log),
                  'observed_log_sha256': log_hash, 'started_utc': attempt['started_utc'],
                  'ended_utc': attempt.get('ended_utc'), 'cost_accounting': attempt.get('cost_accounting')}
        if recovery is not None:
            record.update(authorized_failed_evaluation=True, excluded_from_accuracy=True,
                recovery_receipt_sha256=recovery['receipt_sha256'], archive_relative=recovery['archive_relative'],
                recovery_accounting_note='Original failed evaluation attempt costs retained once in the ordinary attempt ledger. '
                                         'Archived partial predictions are excluded from accuracy; the completed rerun is audited independently.')
        for key in ('process_wall_seconds', 'whole_job_wall_seconds', 'supervisor_preparation_seconds',
                    'postprocess_verification_seconds'):
            value = attempt.get(key)
            require(value is None or (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0),
                    'Invalid cost: ' + key)
            record[key] = value
        output.append(record)
    require(attempts[-1]['status'] in ('complete', 'supervisor_recovery'), 'Final attempt did not complete')
    if entry['job']['kind'].endswith('audit'):
        require(attempts[-1]['status'] == 'complete', 'CPU audit needs an explicit successful exit receipt')
    return output


def verify_receipt(project, entry, required_paths, hashes):
    mapping = entry['result'].get('artifacts_sha256')
    hashes.verify_map(project, mapping, entry['job']['id'])
    require({str(Path(p).resolve()) for p in required_paths} <= set(mapping),
            'Supervisor receipt omits required evidence: ' + entry['job']['id'])


def load_inputs(project, hashes):
    exp, study = project/'experiments', project/'experiments/verl_study'
    protocol, manifest = read(study/'protocol.json'), read(study/'execution_manifest.json')
    require(protocol['status'] == manifest['status'] == 'frozen', 'Protocols are not frozen')
    psha, msha = hashes.sha(study/'protocol.json'), hashes.sha(study/'execution_manifest.json')
    require(manifest['protocol_sha256'] == psha, 'Execution manifest references another protocol')
    require(protocol['budget_seconds'] == 7200 and protocol['endpoint_policy'] == 'guarded_ceiling', 'Budget changed')
    require(protocol['statistical_analysis']['bootstrap']['replicates'] == 20000 and
            protocol['statistical_analysis']['bootstrap']['seed'] == 20261003, 'Statistical plan changed')
    jobs = planned_jobs(protocol)
    state_path = study/'formal_supervisor/state.json'
    state = read(state_path)
    identity = dict(protocol_sha256=psha, execution_manifest_sha256=msha, project_root=str(project), jobs=jobs)
    indexed = validate_state(state, identity)  # fail quickly before expensive native-file hashing
    hashes.sha(state_path)
    hashes.verify_map(exp, protocol['frozen_source_sha256'], 'original frozen sources')
    hashes.verify_map(exp, manifest['source_sha256'], 'execution sources')
    required = {'verl_study/evaluate_verl_formal.py', 'verl_study/audit_verl_formal.py',
                'verl_study/audit_verl_pilot.py', 'verl_study/prepare_formal_test.py',
                'verl_study/run_verl_formal_study.py', 'verl_study/formal_child.py'}
    require(required <= manifest['source_sha256'].keys(), 'Incomplete execution-source binding')
    test_path = study/'formal_test/manifest.json'
    hashes.verify(test_path, manifest['formal_test_manifest_sha256'])
    test = read(test_path)
    require(test['status'] == 'frozen' and test['questions'] == 1319 and test['split'] == 'test', 'Wrong official test')
    require(test['frozen_study_protocol_sha256'] == psha and
            test['materializer_sha256'] == manifest['source_sha256']['verl_study/prepare_formal_test.py'], 'Test provenance differs')
    hashes.verify(contained(test_path.parent, test['source_file']), test['source_sha256'])
    hashes.verify(contained(test_path.parent, test['derived_file']), test['derived_sha256'])
    config = read(exp/'configs/pilot.json')
    splits = {}
    for split, count in [('train', 6961), ('dev', 512)]:
        path = contained(exp, config['data'][split+'_file'])
        hashes.verify(path, config['data'][split+'_sha256'])
        splits[split] = rows(path)
        require(len(splits[split]) == count, 'Wrong data count')
    splits['test'] = rows(contained(test_path.parent, test['derived_file']))
    require(len(splits['test']) == 1319, 'Official test count differs')
    for split, values in splits.items():
        ids = [r['question_sha256'] for r in values]
        require(len(ids) == len(set(ids)), 'Duplicate questions: ' + split)
        require(all(hashlib.sha256(' '.join(r['question'].split()).casefold().encode()).hexdigest() ==
                    r['question_sha256'] for r in values), 'Question hash differs')
    for a, b in itertools.combinations(splits, 2):
        require(not ({r['question_sha256'] for r in splits[a]} & {r['question_sha256'] for r in splits[b]}), 'Split overlap')
    require(test['source_question_sha256'] == [r['question_sha256'] for r in splits['test']], 'Test order differs')
    require(config['reward']['version'] == 3, 'Wrong reward version')
    hashes.verify_map(exp/'reward', config['reward']['code_sha256'], 'reward code')
    sys.path.insert(0, str(exp/'reward'))
    spec = importlib.util.spec_from_file_location('_formal_report_reward', exp/'reward/text_answer_reward.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(project=project, exp=exp, study=study, protocol=protocol, manifest=manifest,
        protocol_sha=psha, manifest_sha=msha, test=test, config=config, splits=splits,
        state=state, indexed=indexed, score_answer=module.score_answer)


def training_result(ctx, declared, hashes):
    project, exp, index = ctx['project'], ctx['exp'], ctx['indexed']
    name = declared['name']
    run = exp/'runs'/name
    entry, audited = index[('train', name)], index[('train_audit', name)]
    required = [run/f for f in ('summary.json','metadata.json','metrics.jsonl','sessions.jsonl','checkpoint_selections.json')]
    verify_receipt(project, entry, required, hashes)
    audit_path = contained(project, audited['result']['audit_output'])
    verify_receipt(project, audited, [audit_path, *required], hashes)
    report = read(audit_path)
    require(report['status'] == 'pass' and report['gpu_operations'] == 0 and
            report['eligible_for_formal_evaluation'] is True, 'Failed or ineligible training audit')
    require(report['run_name'] == name and report['method'] == declared['method'] and report['seed'] == declared['seed'],
            'Training audit belongs to another run')
    require(report['protocol_sha256'] == ctx['protocol_sha'], 'Training audit protocol differs')
    for key, source in [('audit_script_sha256','audit_verl_formal.py'), ('hash_helper_sha256','audit_verl_pilot.py')]:
        require(report[key] == ctx['manifest']['source_sha256']['verl_study/'+source], 'Audit executable was not frozen')
    for key, filename in [('summary_sha256','summary.json'),('metadata_sha256','metadata.json'),
            ('metrics_sha256','metrics.jsonl'),('sessions_sha256','sessions.jsonl'),
            ('checkpoint_selections_sha256','checkpoint_selections.json')]:
        hashes.verify(run/filename, report[key])
    require(report['all_tokens_rewards_signals_and_cursors_verified'] and report['adapter_optimizer_chain_verified'],
            'Training audit omitted required checks')
    evidence = report['evidence_files_sha256']
    hashes.verify_map(run, evidence, 'training native/evidence files')
    require(any(p.startswith('checkpoints/') for p in evidence), 'No native checkpoint evidence')
    require(all(audited['result']['artifacts_sha256'].get(str(contained(run,p))) == h for p,h in evidence.items()),
            'Supervisor did not bind all audited training evidence')
    summary, metadata, selections = read(run/'summary.json'), read(run/'metadata.json'), read(run/'checkpoint_selections.json')
    metrics, sessions = rows(run/'metrics.jsonl'), rows(run/'sessions.jsonl')
    require(summary['status'] == 'complete' and summary['mode'] == 'budgeted' and summary['budget_accounting_valid'] and
            not summary['unaccounted_failed_compute'] and summary['both_sequential_phases_executed'], 'Ineligible training')
    require(metadata['spec']['method'] == declared['method'] and metadata['spec']['seed'] == declared['seed'] and
            metadata['spec']['protocol_sha256'] == ctx['protocol_sha'], 'Training spec mismatch')
    require(metadata['identity'] == summary['identity'] == report['identity'], 'Training identity mismatch')
    budget = summary['budget']
    require(budget['budget_status'] == 'within_allowance' and 6720 < budget['charged_seconds'] <= 7200, 'Invalid time endpoint')
    close(summary['charged_training_seconds'], report['charged_seconds'], 'Audit charged time')
    close(report['charged_seconds'], sum(m['charged_iteration_seconds'] for m in metrics), 'Charged rounds')
    close(report['unused_allowance_seconds'], 7200-report['charged_seconds'], 'Unused allowance')
    require(summary['iterations'] == report['rounds'] == len(metrics) and summary['actual_optimizer_steps'] == report['actual_updates'],
            'Training counts differ from audit')
    for key in TRAIN_TOTALS:
        close(summary[key], sum(m[key] for m in metrics), 'Training total '+key)
    close(summary['training_core_seconds'], sum(summary[k] for k in CORE), 'Training core components')
    require(all(s['status'] in ('paused_at_boundary','complete') and not s.get('unaccounted_prior_failure_detected')
                for s in sessions) and sessions[-1]['status'] == 'complete', 'Unclean training session')
    for label, field in [('primary','selected_primary'),('milestone_050','milestone_050')]:
        selected, audited_selection = selections[label], report[field]
        require(selected['eligible_for_formal_evaluation'] and not selected['diagnostic_only'], 'Diagnostic adapter selected')
        require(all(audited_selection[k] == v for k,v in selected.items()), 'Audited adapter selection differs')
        directory = contained(run, selected['adapter_directory'])
        hashes.verify_map(directory, audited_selection['adapter_files_sha256'], 'selected adapter files')
    require(selections['primary'] == summary['selected_primary'] == entry['result']['selected_primary'], 'Wrong primary endpoint')
    phase_total = report['phase1_seconds']+report['phase2_seconds']
    close(phase_total, report['charged_seconds'], 'Phase clock partition')
    phases = declared['method'].split('_then_')
    phase_costs = {phases[0]: report['phase1_seconds']}
    if len(phases) == 2:
        require(report['phase1_seconds'] > 0 and report['phase2_seconds'] > 0, 'Missing sequential phase')
        phase_costs[phases[1]] = report['phase2_seconds']
    else:
        require(report['phase2_seconds'] == 0, 'Single method has a second phase')
    session_keys = ['student_load_seconds','teacher_load_seconds','pretraining_audit_seconds',
                    'initial_checkpoint_payload_write_seconds','final_audit_seconds','pause_audit_seconds','session_wall_seconds']
    session_costs = {k: sum(s.get(k,0) for s in sessions) for k in session_keys}
    close(session_costs['session_wall_seconds'], report['all_session_wall_seconds'], 'Training session wall')
    return dict(name=name, method=declared['method'], seed=declared['seed'], audit_report_sha256=hashes.sha(audit_path),
        summary=summary, selections=selections, audited_selections={k:report[k] for k in ('selected_primary','milestone_050')},
        costs=dict(charged_seconds=report['charged_seconds'], unused_allowance_seconds=report['unused_allowance_seconds'],
            training_core_seconds=summary['training_core_seconds'], core_components={k:summary[k] for k in CORE},
            within_charge_audit_seconds=summary['iteration_audit_seconds'],
            within_charge_checkpoint_payload_seconds=summary['checkpoint_payload_write_seconds'],
            uncharged_commit_tail_seconds=summary['uncharged_commit_tail_seconds'], session_costs=session_costs,
            independent_audit_internal_seconds=report['audit_seconds'], phase_seconds=phase_costs,
            phase_fractions={k:v/report['charged_seconds'] for k,v in phase_costs.items()},
            switch_after_round=report['switch_after_round'], midpoint_overshoot_seconds=budget['midpoint_overshoot_seconds']),
        counts={k:summary[k] for k in ['iterations','actual_optimizer_steps','skipped_iterations',*TRAIN_TOTALS[:7]]})


def evaluation_result(ctx, job, training, hashes):
    exp, project, index = ctx['exp'], ctx['project'], ctx['indexed']
    name, split = job['name'], job['split']
    run = exp/'runs'/name
    required = [run/f for f in ('metadata.json','initial_weights.json','responses.jsonl','summary.json')]
    required += sorted((run/'batches').glob('*.json')) + sorted((run/'sessions').glob('*.json'))
    entry, audited = index[('eval', name)], index[('eval_audit', name)]
    for receipt in (entry,audited):
        verify_receipt(project, receipt, required, hashes)
        require(receipt['result']['questions'] == COUNTS[split], 'Receipt question count differs')
    require(entry['result']['artifacts_sha256'] == audited['result']['artifacts_sha256'], 'Evaluation audit bound different files')
    metadata, summary = read(run/'metadata.json'), read(run/'summary.json')
    identity = metadata['identity']
    isha = canonical(identity, unicode=True)
    require(metadata['identity_sha256'] == summary['identity_sha256'] == isha, 'Evaluation identity mismatch')
    for value in (identity,summary):
        require(value['split'] == split and value['questions'] == COUNTS[split] and value['role'] == 'student' and
                value['formal_evaluation'] is True and value['eligible_for_main_analysis'] is True and
                value['engineering_smoke'] is False, 'Wrong evaluation role/split/eligibility')
    require(summary['status'] == 'complete' and summary['optimizer_steps'] == identity['optimizer_steps'] == 0, 'Not frozen complete inference')
    require(identity['formal_protocol_sha256'] == ctx['protocol_sha'] and
            identity['execution_manifest_sha256'] == ctx['manifest_sha'] and
            identity['script_sha256'] == ctx['manifest']['source_sha256']['verl_study/evaluate_verl_formal.py'], 'Evaluation source mismatch')
    require(identity['batch_size'] == 4 and identity['seed'] == 42 and identity['enable_thinking'] is False, 'Evaluation options changed')
    for k,v in dict(do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True).items():
        require(identity['generation'][k] == v, 'Greedy generation differs')
    require(identity['config_sha256'] == hashes.sha(exp/'configs/pilot.json') and
            identity['source_protocol_sha256'] == hashes.sha(exp/'verl/protocol.json'), 'Evaluation base config changed')
    for key in ('reward','data','prompt_instruction'):
        require(identity[key] == ctx['config'][key], 'Evaluation data/reward/prompt mismatch')
    model = ctx['protocol']['models_and_initialization']
    require(identity['model'] == model['models']['student'] and
            identity['model_files_sha256'] == model['audited_baseline_model_file_sha256']['student'], 'Model provenance changed')
    # Cache files may be symlinks to shared HF blobs, so use their pinned snapshot root here.
    for relative, expected in identity['model_files_sha256'].items():
        require(Path(relative).name == relative, 'Unsafe model filename')
        hashes.verify(Path(identity['model']['local_snapshot'])/relative, expected)
    origin = identity['adapter_origin']
    adapter = Path(identity['adapter_path']).resolve()
    hashes.verify_map(adapter, identity['adapter_files_sha256'], 'evaluation adapter')
    if job['selection'] == 'initial':
        require(origin == {'kind':'frozen_initial_student','adapter_file_sha256':model['initial_adapter_file_sha256']}, 'Wrong baseline adapter')
        require(adapter == Path(model['initial_adapter_path']).resolve(), 'Initial adapter path differs')
    else:
        train = training[job['train_name']]
        selection = train['selections'][job['selection']]
        aud_selection = train['audited_selections']['selected_primary' if job['selection']=='primary' else 'milestone_050']
        # The native export also has README.md; the evaluator fingerprints only model/config files.
        evaluated_files={k:v for k,v in aud_selection['adapter_files_sha256'].items()
                         if Path(k).name == k and Path(k).suffix in ('.json','.safetensors')}
        require(adapter == Path(selection['adapter_directory']).resolve() and
                identity['adapter_files_sha256'] == evaluated_files, 'Evaluation uses another selected adapter')
        # An unchanged time-selected adapter can legitimately have initialization bytes.
        if origin['kind'] == 'frozen_initial_student':
            require(origin['adapter_file_sha256'] == model['initial_adapter_file_sha256'] ==
                    aud_selection['adapter_files_sha256']['adapter_model.safetensors'], 'False initialization shortcut')
        else:
            require(origin['kind'] == 'time_selected_formal_adapter' and origin['training_run'] == job['train_name'] and
                    origin['selection_label'] == job['selection'] and origin['selection'] == selection and
                    origin['adapter_state_sha256'] == selection['adapter_sha256'], 'Time selection mismatch')
            for field, filename in [('training_metadata_sha256','metadata.json'),('training_summary_sha256','summary.json'),
                                    ('checkpoint_selections_sha256','checkpoint_selections.json')]:
                hashes.verify(exp/'runs'/job['train_name']/filename, origin[field])
    expected = ctx['splits']['test'] if split == 'test' else ctx['splits']['dev'][:COUNTS[split]]
    if split == 'test':
        for field,key in [('formal_test_manifest_sha256',None),('formal_test_derived_sha256','derived_sha256'),
                          ('formal_test_source_sha256','source_sha256')]:
            require(identity[field] == (ctx['manifest']['formal_test_manifest_sha256'] if key is None else ctx['test'][key]), 'Test binding changed')
    records = rows(run/'responses.jsonl')
    require(len(records) == len(expected), 'Missing evaluation questions')
    require(identity['question_sha256'] == [r['question_sha256'] for r in expected] and
            identity['source_rows'] == [r['source_row'] for r in expected], 'Evaluation order changed')
    for record, source in zip(records,expected):
        require(all(record[k] == source[k] for k in ('question','gold','question_sha256','source_row')), 'Response references another question')
        grade = ctx['score_answer'](record['completion'], source['gold'])
        require(all(record[k] == v for k,v in grade.items()), 'Recomputed reward-v3 answer differs')
        require(type(record['correct']) is bool and type(record['answer_valid']) is bool and
                type(record['truncated']) is bool and 0 < len(record['token_ids']) <= 1024, 'Invalid response flags/tokens')
    require(identity['prompt_token_ids_sha256'] == canonical([r['prompt_token_ids'] for r in records],unicode=True), 'Prompt tokens changed')
    expected_totals = dict(correct=sum(r['correct'] for r in records), invalid_answers=sum(not r['answer_valid'] for r in records),
                           truncated_answers=sum(r['truncated'] for r in records), generated_tokens=sum(len(r['token_ids']) for r in records))
    require(all(summary[k] == v for k,v in expected_totals.items()), 'Evaluation summary counts differ')
    close(summary['accuracy'], expected_totals['correct']/len(records), 'Evaluation accuracy')
    for field, path in [('responses_sha256',run/'responses.jsonl'),('initial_weights_file_sha256',run/'initial_weights.json')]:
        hashes.verify(path, summary[field])
    frozen = read(run/'initial_weights.json')
    require(frozen['identity_sha256'] == isha and summary['all_parameter_hash_before'] == summary['all_parameter_hash_after'] ==
            frozen['all_parameter_sha256'], 'Evaluation changed weights')
    batches, sessions = [], []
    for folder, output in [('batches',batches),('sessions',sessions)]:
        files = sorted((run/folder).glob('*.json'))
        mapping = summary['batch_files_sha256' if folder=='batches' else 'session_files_sha256']
        require(set(mapping) == {p.name for p in files}, 'Batch/session file set changed')
        hashes.verify_map(run/folder, mapping, folder)
        output.extend(read(p) for p in files)
    require(len(batches) == math.ceil(len(records)/4) and sessions and sessions[-1]['status'] == 'complete', 'Incomplete evaluation batches/sessions')
    flat = []
    for i,batch in enumerate(batches):
        require(batch['identity_sha256'] == isha and batch['begin'] == i*4 and batch['end'] == min((i+1)*4,len(records)), 'Batch cursor differs')
        require(batch['records_sha256'] == canonical(batch['records'],unicode=True) and
                batch['session_initial_parameter_hash'] == frozen['all_parameter_sha256'] and
                batch['parameter_versions_before'] == batch['parameter_versions_after'] and batch['optimizer_steps'] == 0, 'Batch evidence mismatch')
        flat.extend(batch['records'])
    require(flat == records, 'Response export differs from batches')
    cursor = 0
    for session in sessions:
        require(session['identity_sha256'] == isha and session['status'] in ('paused','complete') and session['optimizer_steps'] == 0 and
                session['all_parameter_hash_before'] == session['all_parameter_hash_after'] == frozen['all_parameter_sha256'], 'Unclosed or changed evaluation weights')
        require(session['start_completed_batches'] == cursor, 'Resume cursor gap')
        end = session['end_completed_batches']
        require(cursor <= end <= len(batches) and all(b['session_id'] == session['session_id'] for b in batches[cursor:end]), 'Session coverage mismatch')
        require(session['last_committed_questions'] == min(end*4,len(records)), 'Session question cursor mismatch')
        cursor = end
    require(cursor == len(batches) and summary['sessions'] == len(sessions) and
            summary['session_statuses'] == [s['status'] for s in sessions] and
            summary['all_committed_batches_have_closing_full_hash'] is True and not summary['batch_sessions_without_closing_full_hash'], 'Incomplete closing-hash coverage')
    for key in EVAL_COSTS:
        if key == 'generation_seconds':
            expected_cost = sum(b[key] for b in batches)
        elif key == 'batch_weight_audit_seconds':
            expected_cost = sum(b['weight_audit_seconds'] for b in batches)
        else:
            expected_cost = sum(s['session_wall_seconds' if key=='known_session_wall_seconds' else key] for s in sessions)
        close(summary[key], expected_cost, key)
    audit_receipt = audit_pass_in_log(contained(project, audited['attempts'][-1]['log']))
    require(audit_receipt['questions'] == len(records) and audit_receipt['correct'] == summary['correct'] and
            audit_receipt['batches'] == len(batches), 'Evaluation audit log disagrees')
    return dict(name=name,split=split,train_name=job.get('train_name'),selection=job['selection'],
        summary_sha256=hashes.sha(run/'summary.json'),identity_sha256=isha,
        predictions=np.array([r['correct'] for r in records],dtype=np.int8),
        question_ids=[r['question_sha256'] for r in records],
        prompt_ids_sha256=identity['prompt_token_ids_sha256'],generation=identity['generation'],versions=identity['versions'],
        correct=summary['correct'],questions=len(records),accuracy_percent=100*summary['accuracy'],
        invalid_answers=summary['invalid_answers'],truncated_answers=summary['truncated_answers'],generated_tokens=summary['generated_tokens'],
        costs={k:summary[k] for k in EVAL_COSTS},peak_torch_allocated_gib=summary['peak_torch_allocated_gib'],
        generation_tokens_per_second=summary['generated_tokens']/summary['generation_seconds'])


def crossed_bootstrap(predictions, baseline, replicates=20000, seed=20261003, chunk_size=128):
    """Independent seed/question draws, paired across every method and baseline."""
    data, base = np.asarray(predictions), np.asarray(baseline)
    require(data.ndim == 3 and base.shape == (data.shape[2],) and data.shape[1] >= 2, 'Bad prediction shape')
    require(np.isin(data,[0,1]).all() and np.isin(base,[0,1]).all(), 'Accuracy requires binary outcomes')
    data, base = data.astype(np.int8), base.astype(np.int8)
    require(replicates > 0 and chunk_size > 0, 'Invalid bootstrap size')
    m,s,n = data.shape
    rng = np.random.default_rng(seed)
    seed_draws = rng.integers(0,s,size=(replicates,s))
    scores, baselines = np.empty((replicates,m)), np.empty(replicates)
    for start in range(0,replicates,chunk_size):
        end = min(start+chunk_size,replicates)
        q = rng.integers(0,n,size=(end-start,n))
        sampled = data[:,seed_draws[start:end,:,None],q[:,None,:]]
        scores[start:end] = sampled.mean(axis=(2,3)).T * 100
        baselines[start:end] = base[q].mean(axis=1)*100
    return scores, baselines


def statistical_results(predictions, baseline, methods=METHODS, seeds=SEEDS, replicates=20000, bootstrap_seed=20261003):
    data = np.asarray(predictions)
    base = np.asarray(baseline)
    require(data.shape[:2] == (len(methods),len(seeds)), 'Unmatched method/seed matrix')
    scores, bootbase = crossed_bootstrap(data,base,replicates,bootstrap_seed)
    n = data.shape[2]
    counts = data.sum(axis=2)
    accuracy = counts/n*100
    means = counts.sum(axis=1)/(len(seeds)*n)*100
    base_accuracy = float(base.sum()/n*100)
    methods_result = []
    for i,method in enumerate(methods):
        per_seed=[]
        for j,seed in enumerate(seeds):
            per_seed.append(dict(seed=seed,correct=int(counts[i,j]),questions=n,accuracy_percent=float(accuracy[i,j]),
                paired_vs_initial=dict(gains=int(np.sum((data[i,j]==1)&(base==0))),
                    losses=int(np.sum((data[i,j]==0)&(base==1))),both_correct=int(np.sum((data[i,j]==1)&(base==1))),
                    both_wrong=int(np.sum((data[i,j]==0)&(base==0))))))
        methods_result.append(dict(method=method,mean_accuracy_percent=float(means[i]),
            sample_sd_percentage_points=float(accuracy[i].std(ddof=1)),minimum_accuracy_percent=float(accuracy[i].min()),
            maximum_accuracy_percent=float(accuracy[i].max()),seeds=per_seed))
    contrasts=[]
    for i,j in itertools.combinations(range(len(methods)),2):
        delta=scores[:,i]-scores[:,j]
        contrasts.append(dict(left=methods[i],right=methods[j],difference_percentage_points=float(means[i]-means[j]),
            descriptive_unadjusted_95_percent_interval=np.quantile(delta,[.025,.975],method='linear').tolist(),
            seed_differences_percentage_points=(accuracy[i]-accuracy[j]).tolist()))
    for i,method in enumerate(methods):
        delta=scores[:,i]-bootbase
        contrasts.append(dict(left=method,right='initial_student',difference_percentage_points=float(means[i]-base_accuracy),
            descriptive_unadjusted_95_percent_interval=np.quantile(delta,[.025,.975],method='linear').tolist(),
            seed_differences_percentage_points=(accuracy[i]-base_accuracy).tolist()))
    ordering=sorted(range(len(methods)),key=lambda i:(-int(counts[i].sum()),i))
    ranks=[dict(rank=1+sum(int(x.sum())>int(counts[i].sum()) for x in counts),method=methods[i],
                mean_accuracy_percent=float(means[i])) for i in ordering]
    top=ordering[0]
    separated=[]
    for j in ordering[1:]:
        interval=np.quantile(scores[:,top]-scores[:,j],[.025,.975],method='linear')
        separated.append(dict(other_method=methods[j],interval_strictly_positive=bool(interval[0]>0),
                              top_higher_in_every_seed=bool(np.all(counts[top]>counts[j]))))
    descriptive_separation=all(r['interval_strictly_positive'] and r['top_higher_in_every_seed'] for r in separated)
    return dict(questions=n,baseline=dict(correct=int(base.sum()),questions=n,accuracy_percent=base_accuracy),
        methods=methods_result,contrasts=contrasts,descriptive_ranking=ranks,
        interpretation=dict(unique_confirmed_winner=False,highest_mean_method=methods[top],
            mean_tied_methods=[methods[i] for i in ordering if counts[i].sum()==counts[top].sum()],
            exploratory_separation_checks=separated,
            result='descriptively_separated_only' if descriptive_separation else 'inconclusive_unique_winner',
            note='The largest mean alone does not establish a winner. Separation checks are descriptive, unadjusted, and use only three seeds.'),
        bootstrap=dict(replicates=replicates,seed=bootstrap_seed,numpy_version=np.__version__,rng='numpy.random.default_rng/PCG64',
            draw_order='Draw all seed indices first, then question indices in replicate order; same draws across all methods and initial baseline.',
            quantile_method='linear',score_array_sha256=hashlib.sha256(scores.astype('<f8').tobytes()+bootbase.astype('<f8').tobytes()).hexdigest(),
            uncertainty_scope='Crossed question and training-seed resampling, conditional on these five configurations and shared initialization; descriptive unadjusted intervals, no nominal coverage guarantee or confirmatory multiplicity correction.'))


def cost_totals(jobs, sessions):
    fields=['process_wall_seconds','whole_job_wall_seconds','supervisor_preparation_seconds','postprocess_verification_seconds']
    groups={}
    unknown=[]
    for job in jobs:
        kind=job['kind']
        group=groups.setdefault(kind,{k:0.0 for k in fields})
        for attempt in job['attempts']:
            for key in fields:
                value=attempt[key]
                if value is None:
                    unknown.append(dict(job_id=job['job_id'],attempt=attempt['number'],field=key,
                                        reason=attempt.get('cost_accounting') or 'Field was not recorded'))
                else:
                    group[key]+=value
    total={k:sum(g[k] for g in groups.values()) for k in fields}
    session_unknown=[i for i,s in enumerate(sessions) if s.get('known_wall_seconds') is None or not s.get('ended_utc')]
    wall=sum(s.get('known_wall_seconds',0) or 0 for s in sessions)
    complete=not any(r['field']=='whole_job_wall_seconds' for r in unknown) and not session_unknown
    return dict(known_attempt_totals_by_kind=groups,known_attempt_totals=total,unknown_attempt_costs=unknown,
        supervisor_sessions=sessions,known_supervisor_session_wall_seconds=wall,
        supervisor_sessions_without_full_wall_record=session_unknown,all_whole_job_costs_known=complete,
        supervisor_overhead_outside_jobs_seconds=wall-total['whole_job_wall_seconds'] if complete else None)


def failed_eval_recovery(project, operation_path, operation, manifest_sha, hashes):
    """Verify the approved failed-evaluation archive and retain its cost only."""
    require(project is not None, 'Failed evaluation recovery needs the project evidence root')
    project = Path(project).resolve()
    parent = operation_path.parent
    evidence = operation['evidence_sha256']
    relative = operation['evaluation_recovery_receipt_relative']
    expected_sha = operation['evaluation_recovery_receipt_sha256']
    require(relative in evidence and evidence[relative] == expected_sha,
            'Failed evaluation recovery receipt is not pinned as evidence')
    receipt_path = contained(parent, relative)
    hashes.verify(receipt_path, expected_sha)
    receipt = read(receipt_path)
    require(receipt.get('schema_version') == 1 and
            receipt.get('operation_kind') == 'authorized_failed_eval_recovery' and
            receipt.get('status') == 'applied_no_process_launched', 'Wrong failed evaluation recovery receipt')
    require(Path(receipt['project_root']).resolve() == project and
            receipt['execution_manifest_sha256'] == manifest_sha, 'Recovery belongs to another project or execution')
    bindings = {
        'authorization.json': receipt['authorization_sha256'],
        'queue-state.failed.json': receipt['state_before_sha256'],
        'queue-state.repaired.json': receipt['state_after_sha256'],
        'recovery_plan.json': receipt['plan_sha256'],
    }
    for name, expected in bindings.items():
        require(evidence.get(name) == expected, 'Recovery evidence binding differs: ' + name)
        hashes.verify(contained(parent, name), expected)
    require(operation.get('authorization_relative') == 'authorization.json' and
            operation.get('state_before_relative') == 'queue-state.failed.json' and
            operation.get('state_after_relative') == 'queue-state.repaired.json' and
            operation.get('recovery_plan_relative') == 'recovery_plan.json' and
            operation.get('authorized_queue_state_sha256') == receipt['state_after_sha256'],
            'Host operation recovery file or repaired-state binding differs')
    require(evidence.get(operation.get('recovery_source_relative')) == receipt['source_sha256'],
            'Recovery implementation is not pinned as evidence')
    flags = ('failed_attempt_costs_retained', 'no_training_change', 'no_frozen_source_change',
             'retry_from_question_zero', 'archive_verified', 'failed_attempt_unchanged',
             'completed_prefix_unchanged', 'supervisor_session_history_unchanged', 'failed_accuracy_excluded')
    require(all(receipt.get(key) is True for key in flags), 'Missing failed evaluation recovery guarantee')
    authorization = read(parent / 'authorization.json')
    require(authorization.get('authorized_evaluation_retry') is True and
            authorization.get('failed_job_id') == receipt['job_id'], 'Wrong explicit evaluation-retry authorization')
    plan = read(parent / 'recovery_plan.json')
    require(all(key == 'prepared_utc' or receipt.get(key) == value for key, value in plan.items()),
            'Applied recovery differs from its prepared plan')
    before, after = read(parent / 'queue-state.failed.json'), read(parent / 'queue-state.repaired.json')
    require(before['identity'] == after['identity'] and
            before['identity_sha256'] == after['identity_sha256'] == canonical(before['identity']),
            'Recovery changed the frozen queue identity')
    require(before['identity']['execution_manifest_sha256'] == manifest_sha and
            before['identity']['protocol_sha256'] == receipt['protocol_sha256'], 'Recovery protocol identity differs')
    hashes.verify(project / 'experiments/verl_study/protocol.json', receipt['protocol_sha256'])
    hashes.verify(project / 'experiments/verl_study/execution_manifest.json', manifest_sha)
    require(before['status'] == 'failed' and before.get('error') and before.get('active_child') is None,
            'Recovery did not start from a stopped failed suite')
    require(len(before['jobs']) == 122 and [entry['job'] for entry in before['jobs']] == before['identity']['jobs'],
            'Recovery queue plan differs')
    target_index = 106
    entry = before['jobs'][target_index]
    job = entry['job']
    require(job['id'] == receipt['job_id'] == '107_eval_verl_study_rl_seed44_01_test' and
            job['kind'] == 'eval' and job['name'] == 'verl_study_rl_seed44_01_test' and
            job['split'] == 'test' and job['selection'] == 'primary', 'Recovery targets an unapproved job')
    require(before['current_job'] == job and entry['status'] == 'running' and not entry.get('result'),
            'Recovery original evaluation state differs')
    require(all(e['status'] == 'complete' for e in before['jobs'][:target_index]) and
            all(e['status'] == 'pending' and not e.get('attempts') for e in before['jobs'][target_index + 1:]),
            'Recovery completed prefix or future queue differs')
    require(sum(e['job']['kind'] == 'train' and e['status'] == 'complete' for e in before['jobs']) == 15 and
            sum(e['job']['kind'] == 'train_audit' and e['status'] == 'complete' for e in before['jobs']) == 15,
            'Recovery was not after all fifteen audited training runs')
    require(len(entry['attempts']) == 1 and receipt['attempt_number'] == 1, 'Unexpected failed attempt number')
    attempt = entry['attempts'][0]
    require(attempt == receipt['failed_attempt'] and canonical(attempt) == receipt['failed_attempt_canonical_sha256'] and
            attempt['number'] == 1 and attempt['status'] == 'exited' and attempt['exit_code'] == 1,
            'Recovery altered the failed attempt')
    expected = copy.deepcopy(before)
    expected.update(status='paused', error=None, current_job=None, active_child=None)
    expected['jobs'][target_index]['status'] = 'pending'
    require(after == expected, 'Recovery state patch exceeds the exact allowlist')
    require(receipt['changes'] == ['status:failed->paused', 'error:original->null', 'current_job:original->null',
            'active_child:null->null', 'jobs[106].status:running->pending'], 'Recovery declared changes differ')
    log = contained(project, receipt['failed_log_relative'])
    require(log == contained(project, attempt['log']) and receipt['failed_log_sha256'] == attempt['log_sha256'],
            'Recovery failed-log binding differs')
    hashes.verify(log, receipt['failed_log_sha256'])
    require('PROCESS_EXIT_CODE: 1' in log.read_text(encoding='utf-8'), 'Missing recorded failed child exit')
    require(receipt['failed_run_relative'] == 'experiments/runs/' + job['name'], 'Recovery source run differs')
    archive = contained(project, receipt['archive_relative'])
    live_run = contained(project, receipt['failed_run_relative'])
    require(archive != live_run and archive.is_dir() and not archive.is_symlink(), 'Invalid failed evaluation archive')
    actual_files = set()
    for archived_path in archive.rglob('*'):
        require(not archived_path.is_symlink(), 'Symlink inside failed evaluation archive')
        if archived_path.is_file():
            actual_files.add(archived_path.relative_to(archive).as_posix())
    mapping = receipt['failed_evaluation_tree_sha256']
    require(actual_files == set(mapping) and 'summary.json' not in mapping, 'Failed archive inventory differs')
    hashes.verify_map(archive, mapping, 'failed evaluation archive')
    metadata = read(archive / 'metadata.json')
    fresh_metadata_path = live_run / 'metadata.json'
    hashes.sha(fresh_metadata_path)
    fresh_metadata = read(fresh_metadata_path)
    require(metadata['identity'] == fresh_metadata['identity'] and
            metadata['identity_sha256'] == fresh_metadata['identity_sha256'] == canonical(metadata['identity'], unicode=True),
            'Rerun changes the failed evaluation model, adapter, data, settings, or frozen code')
    require(metadata['identity']['optimizer_steps'] == 0, 'Archived failure was not evaluation-only')
    session = read(contained(archive, receipt['failed_session_relative']))
    require(session['status'] == 'failed' and not session.get('all_parameter_hash_after') and
            session.get('last_committed_questions') == receipt['committed_questions_excluded_from_accuracy'] == 676,
            'Failed session or excluded partial count differs')
    return dict(job_id=job['id'], job=job, attempt_number=1, failed_attempt=attempt,
        receipt_path=str(receipt_path), receipt_sha256=expected_sha, archive_relative=receipt['archive_relative'],
        committed_questions_excluded_from_accuracy=676,
        accounting_note='Original failed attempt costs appear once in the job ledger. Archive costs are nested diagnostics and are not added again.')


def host_operations(path, manifest_sha, hashes, project=None):
    if path is None:
        return dict(status='not_supplied',host_overhead_seconds=None,
                    limitation='Windows host helper/watcher overhead is outside the WSL ledger and was not measured.')
    path=Path(path)
    record=read(path)
    require(record['execution_manifest_sha256']==manifest_sha,'Host watcher belongs to another execution manifest')
    watcher=path.parent/'source/watch_verl_formal_host.ps1'
    hashes.verify(watcher,record['watcher_sha256'])
    result=dict(status='manifest_verified',manifest_sha256=hashes.sha(path),manifest=record,
        host_overhead_seconds=None,
        limitation='Host helper/watcher CPU and recovery wall overhead were not timed. These do not change the independently recorded WSL charged-training clock.')
    seen_operations=set()
    def verify_status(status):
        require(status['execution_manifest_sha256']==manifest_sha,'Host status execution manifest differs')
        if status.get('schema_version',1)==2:
            # Verify each operations-only resume back to the original immutable watcher.
            operation_path=contained(path.parent,status['operations_manifest_relative'])
            require(operation_path not in seen_operations,'Cycle in host operations history')
            require(len(seen_operations)<64,'Host operations history exceeds supported depth')
            seen_operations.add(operation_path)
            hashes.verify(operation_path,status['host_operations_manifest_sha256'])
            operation=read(operation_path)
            require(operation.get('schema_version')==2 and operation['execution_manifest_sha256']==manifest_sha and
                    operation['original_host_operations_manifest_sha256']==result['manifest_sha256'],
                    'Recovery operation belongs to different frozen evidence')
            require(status['helper_sha256']==operation['helper_sha256'] and
                    status['watcher_sha256']==operation['helper_sha256'],'Recovery host helper versions differ')
            hashes.verify(contained(operation_path.parent,operation['helper_relative_path']),operation['helper_sha256'])
            evidence=operation['evidence_sha256']
            hashes.verify_map(operation_path.parent,evidence,'recovery operation')
            prior_relative=operation['prior_host_status_relative']
            require(prior_relative in evidence and evidence[prior_relative]==operation['prior_host_status_sha256'] and
                    status['prior_host_status_sha256']==operation['prior_host_status_sha256'],
                    'Recovery prior host status hash differs')
            prior=read(contained(operation_path.parent,prior_relative))
            history=verify_status(prior)
            incident=None
            gap=None
            evaluation_recovery=None
            operation_kind=operation.get('operation_kind')
            if operation_kind == 'authorized_failed_eval_recovery':
                evaluation_recovery=failed_eval_recovery(project,operation_path,operation,manifest_sha,hashes)
            incident_relative=operation.get('incident_relative')
            if incident_relative is not None:
                require(incident_relative in evidence,'Recovery incident is not pinned as evidence')
                incident=read(contained(operation_path.parent,incident_relative))
                if evaluation_recovery is None:
                    gap=incident['difference_utc_minus_recorded_seconds']
                    require(math.isfinite(gap) and gap>=0,'Recovery UTC/process timing difference is invalid')
                    close(gap,incident['attempt_utc_elapsed_seconds']-incident['attempt_recorded_monotonic_seconds'],
                          'Recovery UTC/process timing difference')
            else:
                require(operation_kind=='user_requested_resume' or evaluation_recovery is not None,
                        'Only a routine resume or verified failed-evaluation recovery may omit incident evidence')
            history.append(dict(
                manifest_relative=status['operations_manifest_relative'],manifest_sha256=hashes.sha(operation_path),
                manifest=operation,prior_host_status=prior,incident=incident,
                failed_eval_recovery=evaluation_recovery,
                utc_minus_recorded_process_seconds=gap,added_to_charged_training_seconds=False,
                accounting_note=('Failed evaluation cost remains once in its original attempt ledger; archived predictions are excluded. '
                                 'GPU repair/helper overhead is unmeasured and no UTC/process gap is invented.' if evaluation_recovery else
                                 'Report the UTC/process timer difference separately. It is not a measured GPU compute cost or Windows helper overhead, and does not alter the preserved charged-training ledger.')))
            return history
        else:
            require(status.get('schema_version',1)==1 and status['watcher_sha256']==record['watcher_sha256'] and
                    status['host_operations_manifest_sha256']==result['manifest_sha256'],'Host status helper versions differ')
            return []
    status_path=path.parent/'host-status.json'
    if status_path.is_file():
        # Status is a live presentation file, not formal evidence; snapshot its exact bytes once.
        contents=status_path.read_bytes()
        status=json.loads(contents)
        history=verify_status(status)
        if history:
            # Count each incident once across the resume history.
            incidents=[dict(operation_manifest_relative=item['manifest_relative'],incident=item['incident'],
                utc_minus_recorded_process_seconds=item['utc_minus_recorded_process_seconds'])
                for item in history if item['incident'] is not None]
            result.update(status='original_and_recovery_manifests_verified',recovery_operation=history[-1],
                recovery_operations=history,operational_incidents=incidents,
                recorded_incident_utc_minus_process_seconds=sum(item['utc_minus_recorded_process_seconds'] for item in incidents
                                                               if item['utc_minus_recorded_process_seconds'] is not None),
                incident_differences_added_to_charged_training_seconds=False)
            recoveries=[item['failed_eval_recovery'] for item in history if item['failed_eval_recovery'] is not None]
            require(len({(r['job_id'],r['attempt_number']) for r in recoveries})==len(recoveries),
                    'Duplicate authorization for the same failed evaluation attempt')
            result['authorized_failed_eval_recoveries']=recoveries
        result.update(observed_status=status,observed_status_sha256=hashlib.sha256(contents).hexdigest())
    return result


def analyze(project,host_operations_manifest=None):
    started=time.perf_counter()
    hashes=Hashes()
    ctx=load_inputs(project.resolve(),hashes)
    print('PASS: complete frozen job plan. Checking all receipts and evidence hashes.',flush=True)
    host=host_operations(host_operations_manifest,ctx['manifest_sha'],hashes,ctx['project'])
    authorized_failed_evals={(r['job_id'],r['attempt_number']):r
                            for r in host.get('authorized_failed_eval_recoveries',[])}
    job_costs=[]
    for entry in ctx['state']['jobs']:
        attempts=attempt_ledger(ctx['project'],entry,hashes,authorized_failed_evals)
        job=entry['job']
        job_costs.append(dict(job_id=job['id'],kind=job['kind'],name=job['name'],split=job.get('split'),
            train_name=job.get('train_name'),attempts=attempts))
    used_recoveries={(job['job_id'],attempt['number']) for job in job_costs for attempt in job['attempts']
                     if attempt.get('authorized_failed_evaluation')}
    require(used_recoveries==set(authorized_failed_evals),'Authorized failed evaluation receipt was not consumed exactly once')
    training={}
    for declared in ctx['protocol']['run_order']['runs']:
        training[declared['name']]=training_result(ctx,declared,hashes)
        print('PASS: audited training '+declared['name'],flush=True)
    evaluations={}
    for entry in ctx['state']['jobs']:
        if entry['job']['kind']=='eval':
            result=evaluation_result(ctx,entry['job'],training,hashes)
            evaluations[result['name']]=result
    require(len(training)==15 and len(evaluations)==46,'Missing formal study results')
    for split in COUNTS:
        values=[r for r in evaluations.values() if r['split']==split]
        reference=values[0]
        for item in values:
            require(all(item[k]==reference[k] for k in ('question_ids','prompt_ids_sha256','generation','versions')),
                    'Evaluation pairing or generation/environment mismatch within '+split)
    names={(r['method'],r['seed']):r['name'] for r in training.values()}
    baseline=evaluations['verl_study_initial_test_01']['predictions']
    predictions=np.stack([[evaluations[names[(m,s)]+'_test']['predictions'] for s in SEEDS] for m in METHODS])
    require(predictions.shape==(5,3,1319), 'Incomplete primary matrix')
    print('PASS: all 15 train audits and 46 evaluation audits. Computing 20,000 paired crossed resamples.',flush=True)
    stats=statistical_results(predictions,baseline)
    secondary={}
    for split in ('dev64','dev512'):
        records=[]
        for method in METHODS:
            vals=[evaluations[names[(method,s)]+'_'+split] for s in SEEDS]
            accuracies=np.array([v['accuracy_percent'] for v in vals])
            records.append(dict(method=method,mean_accuracy_percent=float(accuracies.mean()),
                sample_sd_percentage_points=float(accuracies.std(ddof=1)),minimum_accuracy_percent=float(accuracies.min()),
                maximum_accuracy_percent=float(accuracies.max()),seeds=[dict(seed=s,correct=v['correct'],questions=v['questions'],
                    accuracy_percent=v['accuracy_percent']) for s,v in zip(SEEDS,vals)]))
        secondary[split]=records
    totals=cost_totals(job_costs,ctx['state'].get('supervisor_sessions',[]))
    totals['charged_training_seconds']=sum(r['costs']['charged_seconds'] for r in training.values())
    totals['unused_training_allowance_seconds']=sum(r['costs']['unused_allowance_seconds'] for r in training.values())
    totals['training_session_costs']={k:sum(r['costs']['session_costs'][k] for r in training.values())
                                     for k in next(iter(training.values()))['costs']['session_costs']}
    totals['training_core_components']={k:sum(r['costs']['core_components'][k] for r in training.values()) for k in CORE}
    totals['uncharged_commit_tail_seconds']=sum(r['costs']['uncharged_commit_tail_seconds'] for r in training.values())
    totals['training_independent_audit_internal_seconds']=sum(r['costs']['independent_audit_internal_seconds'] for r in training.values())
    totals['evaluation_by_split']={split:{k:sum(r['costs'][k] for r in evaluations.values() if r['split']==split) for k in EVAL_COSTS} for split in COUNTS}
    totals['within_charged_training_audit_seconds']=sum(r['costs']['within_charge_audit_seconds'] for r in training.values())
    totals['within_charged_checkpoint_payload_seconds']=sum(r['costs']['within_charge_checkpoint_payload_seconds'] for r in training.values())
    method_costs=[]
    for method in METHODS:
        rr=[r for r in training.values() if r['method']==method]
        ev=[e for e in evaluations.values() if e['train_name'] in {r['name'] for r in rr}]
        method_costs.append(dict(method=method,charged_training_seconds=sum(r['costs']['charged_seconds'] for r in rr),
            unused_training_allowance_seconds=sum(r['costs']['unused_allowance_seconds'] for r in rr),
            evaluation_generation_seconds=sum(e['costs']['generation_seconds'] for e in ev),
            evaluation_known_session_wall_seconds=sum(e['costs']['known_session_wall_seconds'] for e in ev),
            phase_fraction_by_seed=[dict(seed=r['seed'],**r['costs']['phase_fractions']) for r in sorted(rr,key=lambda x:x['seed'])],
            actual_updates=sum(r['counts']['actual_optimizer_steps'] for r in rr),
            generated_training_tokens=sum(r['counts']['generated_tokens'] for r in rr)))
    # Export outcome and question-order hashes.
    for result in evaluations.values():
        result['correct_mask_sha256']=hashlib.sha256(result.pop('predictions').tobytes()).hexdigest()
        result['ordered_question_ids_sha256']=canonical(result.pop('question_ids'),unicode=True)
        result.pop('generation')
    hashes.unchanged()
    return dict(schema_version=1,status='complete_verified',gpu_operations=0,created_utc=datetime.now(timezone.utc).isoformat(),
        provenance=dict(project_root=str(project.resolve()),protocol_sha256=ctx['protocol_sha'],execution_manifest_sha256=ctx['manifest_sha'],
            supervisor_state_sha256=hashes.sha(ctx['study']/'formal_supervisor/state.json'),
            formal_test_manifest_sha256=ctx['manifest']['formal_test_manifest_sha256'],analysis_script_sha256=hashes.sha(__file__),
            verified_files=len(hashes.cache),bytes_read_for_hashing=hashes.bytes_read,completed_training_runs=15,
            completed_evaluations=46,completed_supervisor_jobs=122),
        primary=stats,secondary=secondary,frozen_statistical_plan=ctx['protocol']['statistical_analysis'],
        training=list(training.values()),evaluations=list(evaluations.values()),
        costs=dict(totals=totals,by_method=method_costs,jobs=job_costs,host_operations=host,
            accounting_rules=[
                'Charged training time is the common upper-allowance clock; actual charged times may differ within the guard.',
                'Core, iteration-audit, and checkpoint timers are components within charged rounds. They are not added again to charged time.',
                'Session setup, restart, closing verification/export, and commit tails are outside charged rounds but inside training session/process wall.',
                'Evaluation preparation/recovery are outside its known session wall; loading, generation and weight audits are inside it.',
                'Supervisor whole-job wall contains child process, preparation, verification, and orchestration. Do not add nested training/evaluation/audit timers to this total.',
                'The explicitly authorized failed evaluation attempt remains in the job ledger once, with its original status and timers. Archived partial predictions never enter accuracy; archive diagnostic timers are not added again.',
                'CPU audit wall is included as its own job kind. Independent-auditor internal seconds overlap that audit process wall.',
                'Known subtotals exclude missing timers; missing costs are listed explicitly and never treated as measured zeros.'],
            omissions=[
                'Pre-study setup, model/dataset downloads, engineering pilots, and protocol calibration are outside this formal execution ledger.',
                'This post-study analysis and report production are outside the supervisor study ledger; analysis runtime is reported separately.',
                'Powered-on time outside active supervisor sessions, energy, FLOPs, GPU-busy time, and device-wide memory peaks were not measured.',
                'Windows host helper/watcher overhead and recovery wall time were not measured; a host-helper failure does not itself imply missing WSL training compute.',
                'Peak torch allocation is framework allocation, not total GPU memory; the recorded evaluation peak is cumulative within a process.',
                'Only existing evidence was audited; model inference was not rerun by this analysis.']),
        analysis_wall_seconds=time.perf_counter()-started)


def write_tables(result,directory):
    directory.mkdir(parents=True,exist_ok=True)
    method_rows=[]
    for row in result['primary']['methods']:
        record={k:v for k,v in row.items() if k!='seeds'}
        for seed in row['seeds']:
            record[f"seed{seed['seed']}_correct"]=seed['correct']
            record[f"seed{seed['seed']}_accuracy_percent"]=seed['accuracy_percent']
        method_rows.append(record)
    contrast_rows=[]
    for row in result['primary']['contrasts']:
        lo,hi=row['descriptive_unadjusted_95_percent_interval']
        contrast_rows.append(dict(left=row['left'],right=row['right'],difference_pp=row['difference_percentage_points'],ci_low_pp=lo,ci_high_pp=hi))
    cost_rows=[]
    for row in result['training']:
        cost=row['costs']
        cost_rows.append(dict(method=row['method'],seed=row['seed'],charged_seconds=cost['charged_seconds'],
            unused_seconds=cost['unused_allowance_seconds'],session_wall_seconds=cost['session_costs']['session_wall_seconds'],
            updates=row['counts']['actual_optimizer_steps'],rounds=row['counts']['iterations'],
            rl_fraction=cost['phase_fractions'].get('rl',0),opd_fraction=cost['phase_fractions'].get('opd',0),
            joint_fraction=cost['phase_fractions'].get('joint',0)))
    def csv_file(name,table):
        with (directory/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]))
            writer.writeheader();writer.writerows(table)
    csv_file('method_results.csv',method_rows)
    csv_file('paired_contrasts.csv',contrast_rows)
    csv_file('training_costs.csv',cost_rows)
    csv_file('evaluation_results.csv',[{k:r[k] for k in ('name','split','correct','questions','accuracy_percent','invalid_answers',
        'truncated_answers','generated_tokens','generation_tokens_per_second')} for r in result['evaluations']])
    def latex_escape(value):
        return str(value).replace('_',r'\_')
    lines=[r'\begin{tabular}{lrrrr}',r'\hline',r'Method & Mean (\%) & SD (pp) & Min (\%) & Max (\%) \\',r'\hline']
    for row in method_rows:
        lines.append(f"{latex_escape(row['method'])} & {row['mean_accuracy_percent']:.3f} & {row['sample_sd_percentage_points']:.3f} & {row['minimum_accuracy_percent']:.3f} & {row['maximum_accuracy_percent']:.3f} "+r'\\')
    lines += [r'\hline',r'\end{tabular}']
    (directory/'method_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{tabular}{llrr}',r'\hline',r'Left & Right & Difference (pp) & Descriptive 95\% interval (pp) \\',r'\hline']
    for row in contrast_rows:
        lines.append(f"{latex_escape(row['left'])} & {latex_escape(row['right'])} & {row['difference_pp']:.3f} & [{row['ci_low_pp']:.3f}, {row['ci_high_pp']:.3f}] "+r'\\')
    lines += [r'\hline',r'\end{tabular}']
    (directory/'paired_contrasts.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tables-dir',type=Path)
    parser.add_argument('--host-operations-manifest',type=Path,help='Optional delivered host watcher manifest; reads sibling host-status.json as an operational snapshot.')
    args=parser.parse_args()
    try:
        host=args.host_operations_manifest
        inferred=Path(__file__).resolve().parent.parent/'host_operations_manifest.json'
        if host is None and inferred.is_file():
            host=inferred
        result=analyze(args.project_root,host)
    except Exception as error:
        print(json.dumps(dict(status='rejected',gpu_operations=0,error_type=type(error).__name__,error=str(error))),file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_name(args.output.name+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,args.output)
    if args.tables_dir:
        write_tables(result,args.tables_dir)
    print(json.dumps(dict(status=result['status'],output=str(args.output),primary=result['primary']),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
