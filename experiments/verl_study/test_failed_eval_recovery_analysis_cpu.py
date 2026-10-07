"""Synthetic CPU-only checks for the analysis-only failed-evaluation exception."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

import analyze_verl_formal as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reseal(op, operation, receipt):
    """Rebind synthetic hashes so semantic-rejection tests reach their target."""
    receipt['state_before_sha256'] = sha(op/'queue-state.failed.json')
    receipt['state_after_sha256'] = sha(op/'queue-state.repaired.json')
    receipt['authorization_sha256'] = sha(op/'authorization.json')
    plan = {key: value for key, value in receipt.items() if key not in (
        'status', 'applied_utc', 'plan_sha256', 'archive_verified', 'failed_attempt_unchanged',
        'completed_prefix_unchanged', 'supervisor_session_history_unchanged', 'failed_accuracy_excluded')}
    write(op/'recovery_plan.json', dict(prepared_utc='synthetic', **plan))
    receipt['plan_sha256'] = sha(op/'recovery_plan.json')
    write(op/'recovery_receipt.json', receipt)
    operation['authorized_queue_state_sha256'] = receipt['state_after_sha256']
    operation['evaluation_recovery_receipt_sha256'] = sha(op/'recovery_receipt.json')
    operation['evidence_sha256'] = {p.name: sha(p) for p in op.iterdir() if p.is_file() and p.name != 'operations_manifest.json'}
    write(op/'operations_manifest.json', operation)
    write(op.parent.parent/'host-status.json', dict(schema_version=2, status='complete',
        execution_manifest_sha256=operation['execution_manifest_sha256'],
        operations_manifest_relative=op.relative_to(op.parent.parent).as_posix()+'/operations_manifest.json',
        host_operations_manifest_sha256=sha(op/'operations_manifest.json'),
        helper_sha256=operation['helper_sha256'], watcher_sha256=operation['helper_sha256'],
        prior_host_status_sha256=operation['prior_host_status_sha256']))


def fixture(root):
    study = root/'experiments/verl_study'
    write(study/'protocol.json', {'status': 'frozen', 'synthetic': True})
    write(study/'execution_manifest.json', {'status': 'frozen', 'protocol_sha256': sha(study/'protocol.json')})
    manifest_sha = sha(study/'execution_manifest.json')
    host = root/'host'
    write(host/'source/watch_verl_formal_host.ps1', {'synthetic': 'original helper'})
    original = host/'host_operations_manifest.json'
    write(original, dict(execution_manifest_sha256=manifest_sha, watcher_sha256=sha(host/'source/watch_verl_formal_host.ps1')))
    prior = dict(execution_manifest_sha256=manifest_sha, watcher_sha256=sha(host/'source/watch_verl_formal_host.ps1'),
                 host_operations_manifest_sha256=sha(original), status='failed')
    op = host/'operations/recovery'
    write(op/'prior-host-status.json', prior)
    write(op/'resume.ps1', {'synthetic': 'new helper'})
    write(op/'recover_failed_eval.py', {'synthetic': 'recovery program'})
    job = dict(id='107_eval_verl_study_rl_seed44_01_test', kind='eval', name='verl_study_rl_seed44_01_test',
               split='test', selection='primary', train_name='verl_study_rl_seed44_01')
    log = root/'experiments/run_logs/failed.log'
    log.parent.mkdir(parents=True)
    log.write_text('CUDA error: unknown error\nPROCESS_EXIT_CODE: 1\n')
    attempt = dict(number=1, status='exited', exit_code=1, log=str(log), log_sha256=sha(log),
                   started_utc='synthetic-start', ended_utc='synthetic-end', process_wall_seconds=10.,
                   whole_job_wall_seconds=12., supervisor_preparation_seconds=1.)
    jobs = [dict(job=dict(id=f'{i+1:03d}_synthetic', kind='eval', name=f'synthetic_{i}'),
                 status='complete' if i < 106 else 'pending', attempts=[]) for i in range(122)]
    for i in range(30):
        jobs[i]['job']['kind'] = 'train' if i % 2 == 0 else 'train_audit'
    jobs[106] = dict(job=job, status='running', attempts=[attempt])
    identity = dict(jobs=[entry['job'] for entry in jobs], execution_manifest_sha256=manifest_sha,
                    protocol_sha256=sha(study/'protocol.json'), project_root=str(root))
    before = dict(status='failed', error='CUDA failure', active_child=None, current_job=job,
                  identity=identity, identity_sha256=a.canonical(identity), jobs=jobs, supervisor_sessions=[{'unchanged': True}])
    after = copy.deepcopy(before)
    after.update(status='paused', error=None, current_job=None, active_child=None)
    after['jobs'][106]['status'] = 'pending'
    write(op/'queue-state.failed.json', before)
    write(op/'queue-state.repaired.json', after)
    write(op/'authorization.json', dict(authorized_evaluation_retry=True, failed_job_id=job['id']))
    archive_relative = 'experiments/verl_study/formal_supervisor/recovery_archives/synthetic/failed_eval_attempt001'
    archive = root/archive_relative
    eval_identity = dict(optimizer_steps=0, adapter='same adapter', generation={'do_sample': False})
    metadata = dict(identity=eval_identity, identity_sha256=a.canonical(eval_identity, unicode=True))
    write(archive/'metadata.json', metadata)
    write(archive/'sessions/failed.json', dict(status='failed', last_committed_questions=676, all_parameter_hash_after=None))
    write(archive/'batches/0000.json', {'synthetic': 'preserved partial outcomes, excluded from accuracy'})
    write(root/'experiments/runs'/job['name']/'metadata.json', metadata)
    receipt = dict(schema_version=1, operation_kind='authorized_failed_eval_recovery', status='applied_no_process_launched',
        applied_utc='synthetic', job_id=job['id'], attempt_number=1, project_root=str(root),
        execution_manifest_sha256=manifest_sha, protocol_sha256=sha(study/'protocol.json'),
        source_sha256=sha(op/'recover_failed_eval.py'), failed_run_relative='experiments/runs/'+job['name'],
        archive_relative=archive_relative, failed_attempt=attempt, failed_attempt_canonical_sha256=a.canonical(attempt),
        failed_log_relative=log.relative_to(root).as_posix(), failed_log_sha256=sha(log),
        failed_evaluation_tree_sha256={p.relative_to(archive).as_posix(): sha(p) for p in archive.rglob('*') if p.is_file()},
        failed_session_relative='sessions/failed.json', committed_questions_excluded_from_accuracy=676,
        changes=['status:failed->paused','error:original->null','current_job:original->null','active_child:null->null','jobs[106].status:running->pending'])
    for flag in ('failed_attempt_costs_retained', 'no_training_change', 'no_frozen_source_change', 'retry_from_question_zero',
                 'archive_verified', 'failed_attempt_unchanged', 'completed_prefix_unchanged',
                 'supervisor_session_history_unchanged', 'failed_accuracy_excluded'):
        receipt[flag] = True
    operation = dict(schema_version=2, operation_kind='authorized_failed_eval_recovery', execution_manifest_sha256=manifest_sha,
        original_host_operations_manifest_sha256=sha(original), helper_relative_path='resume.ps1', helper_sha256=sha(op/'resume.ps1'),
        prior_host_status_relative='prior-host-status.json', prior_host_status_sha256=sha(op/'prior-host-status.json'),
        evaluation_recovery_receipt_relative='recovery_receipt.json', recovery_source_relative='recover_failed_eval.py',
        authorization_relative='authorization.json', state_before_relative='queue-state.failed.json',
        state_after_relative='queue-state.repaired.json', recovery_plan_relative='recovery_plan.json')
    reseal(op, operation, receipt)
    completed_log = root/'experiments/run_logs/completed.log'
    completed_log.write_text('PROCESS_EXIT_CODE: 0\n')
    final_attempt = dict(attempt, number=2, status='complete', exit_code=0, log=str(completed_log),
                         log_sha256=sha(completed_log), postprocess_verification_seconds=1.)
    entry = dict(job=job, status='complete', attempts=[attempt, final_attempt], result={'synthetic': True})
    return original, op, operation, receipt, entry


class RecoveryAnalysisTests(unittest.TestCase):
    def test_authorized_failed_cost_retained_once_and_unknown_stays_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, op, operation, receipt, entry = fixture(root)
            host = a.host_operations(original, operation['execution_manifest_sha256'], a.Hashes(), root)
            recovered = host['authorized_failed_eval_recoveries']
            self.assertEqual(len(recovered), 1)
            auth = {(r['job_id'], r['attempt_number']): r for r in recovered}
            ledger = a.attempt_ledger(root, entry, a.Hashes(), auth)
            self.assertEqual([r['status'] for r in ledger], ['exited', 'complete'])
            self.assertTrue(ledger[0]['excluded_from_accuracy'])
            self.assertIsNone(ledger[0]['postprocess_verification_seconds'])
            totals = a.cost_totals([dict(job_id=entry['job']['id'], kind='eval', attempts=ledger)],
                                   [dict(known_wall_seconds=30., ended_utc='end')])
            self.assertEqual(totals['known_attempt_totals']['whole_job_wall_seconds'], 24.)
            self.assertEqual(totals['known_attempt_totals']['process_wall_seconds'], 20.)
            self.assertEqual(len(totals['unknown_attempt_costs']), 1)
            self.assertEqual(host['recorded_incident_utc_minus_process_seconds'], 0)

    def test_unapproved_failed_attempt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            *_, entry = fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'Unauthorized'):
                a.attempt_ledger(root, entry, a.Hashes())

    def test_final_failed_attempt_training_and_changed_costs_rejected(self):
        for change in ('final_failed', 'training', 'changed_cost'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                original, op, operation, receipt, entry = fixture(root)
                recovered = a.host_operations(original, operation['execution_manifest_sha256'], a.Hashes(), root)['authorized_failed_eval_recoveries'][0]
                auth = {(recovered['job_id'], 1): recovered}
                if change == 'final_failed': entry['attempts'] = entry['attempts'][:1]
                elif change == 'training': entry['job'] = dict(entry['job'], kind='train')
                else: entry['attempts'][0] = dict(entry['attempts'][0], whole_job_wall_seconds=0.)
                with self.assertRaises(RuntimeError): a.attempt_ledger(root, entry, a.Hashes(), auth)

    def test_archive_changed_missing_or_extra_file_rejected(self):
        for change in ('changed', 'missing', 'extra'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                original, op, operation, receipt, entry = fixture(root)
                archive = root/receipt['archive_relative']
                target = archive/'batches/0000.json'
                if change == 'changed': target.write_bytes(target.read_bytes()+b' ')
                elif change == 'missing': target.unlink()
                else: write(archive/'extra.json', {'extra': True})
                with self.assertRaises(RuntimeError): a.host_operations(original, operation['execution_manifest_sha256'], a.Hashes(), root)

    def test_hash_resealed_state_edit_beyond_allowlist_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, op, operation, receipt, entry = fixture(root)
            after = a.read(op/'queue-state.repaired.json')
            after['jobs'][0]['status'] = 'pending'
            write(op/'queue-state.repaired.json', after)
            reseal(op, operation, receipt)
            with self.assertRaisesRegex(RuntimeError, 'exact allowlist'):
                a.host_operations(original, operation['execution_manifest_sha256'], a.Hashes(), root)

    def test_fresh_model_or_adapter_identity_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, op, operation, receipt, entry = fixture(root)
            target = root/receipt['failed_run_relative']/'metadata.json'
            metadata = a.read(target)
            metadata['identity']['adapter'] = 'another adapter'
            metadata['identity_sha256'] = a.canonical(metadata['identity'], unicode=True)
            write(target, metadata)
            with self.assertRaisesRegex(RuntimeError, 'Rerun changes'):
                a.host_operations(original, operation['execution_manifest_sha256'], a.Hashes(), root)

    def test_wrong_authorization_and_source_binding_rejected(self):
        for change in ('authorization', 'source', 'repaired_state'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                original, op, operation, receipt, entry = fixture(root)
                if change == 'authorization':
                    write(op/'authorization.json', dict(authorized_evaluation_retry=True, failed_job_id='wrong'))
                    reseal(op, operation, receipt)
                elif change == 'source':
                    operation['recovery_source_relative'] = 'authorization.json'
                    reseal(op, operation, receipt)
                else:
                    operation['authorized_queue_state_sha256'] = '0'*64
                    write(op/'operations_manifest.json', operation)
                    status = a.read(op.parent.parent/'host-status.json')
                    status['host_operations_manifest_sha256'] = sha(op/'operations_manifest.json')
                    write(op.parent.parent/'host-status.json', status)
                with self.assertRaises(RuntimeError): a.host_operations(original, operation['execution_manifest_sha256'], a.Hashes(), root)

    def test_gpu_incident_does_not_invent_sleep_timer_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, op, operation, receipt, entry = fixture(root)
            write(op/'gpu-incident.json', {'kind': 'gpu_channel_loss', 'root_cause': 'unknown'})
            operation['incident_relative'] = 'gpu-incident.json'
            reseal(op, operation, receipt)
            host = a.host_operations(original, operation['execution_manifest_sha256'], a.Hashes(), root)
            self.assertIsNone(host['operational_incidents'][0]['utc_minus_recorded_process_seconds'])
            self.assertEqual(host['recorded_incident_utc_minus_process_seconds'], 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryAnalysisTests))
    report = dict(status='pass' if result.wasSuccessful() else 'fail', gpu_operations=0,
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        elapsed_seconds=time.perf_counter()-started, analysis_source_sha256=sha(Path(a.__file__)),
        test_source_sha256=sha(Path(__file__)), scope='Synthetic authorized failed-evaluation recovery, exact state allowlist, archive/identity/authorization tamper rejection, single-count costs. No GPU or official result statistics.')
    write(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
