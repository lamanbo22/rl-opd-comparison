"""CPU tests for study analysis using synthetic evaluation records."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze_verl_formal.py'))
a=importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(project,n=1319):
    """A complete, artificial two-session evaluation, including the final 3-record batch."""
    exp=project/'experiments'
    run=exp/'runs/synthetic_test'
    modeldir=project/'model'
    adapter=exp/'runs/initial/adapter_initial'
    write(modeldir/'config.json',{'synthetic':True})
    write(adapter/'adapter_config.json',{'synthetic':True})
    (adapter/'adapter_model.safetensors').write_bytes(b'synthetic adapter; not a model')
    modelfiles={'config.json':sha(modeldir/'config.json')}
    adapterfiles={p.name:sha(p) for p in adapter.iterdir()}
    config=dict(data={},reward={'version':3},prompt_instruction='Answer: ')
    write(exp/'configs/pilot.json',config)
    write(exp/'verl/protocol.json',{'synthetic':True})
    model=dict(id='synthetic',revision='fixed',local_snapshot=str(modeldir))
    sources=[]
    for i in range(n):
        question=f'Synthetic question {i} with unicode \u2028 separator'
        sources.append(dict(source_row=i,question=question,gold='1',
            question_sha256=hashlib.sha256(' '.join(question.split()).casefold().encode()).hexdigest()))
    records=[]
    for i,row in enumerate(sources):
        correct=i%3!=0
        records.append(dict(row,completion='1' if correct else '0',correct=correct,answer_valid=True,reward=int(correct),
                            truncated=False,eos_terminated=True,token_ids=[int(correct),9],prompt_token_ids=[10,i]))
    modelinit=dict(models={'student':model},audited_baseline_model_file_sha256={'student':modelfiles},
        initial_adapter_file_sha256=adapterfiles['adapter_model.safetensors'],initial_adapter_path=str(adapter))
    scriptsha='a'*64
    identity=dict(role='student',split='test',questions=n,formal_evaluation=True,eligible_for_main_analysis=True,
        engineering_smoke=False,optimizer_steps=0,formal_protocol_sha256='p'*64,execution_manifest_sha256='m'*64,
        script_sha256=scriptsha,batch_size=4,seed=42,enable_thinking=False,
        generation=dict(do_sample=False,num_beams=1,max_new_tokens=1024,use_cache=True),
        config_sha256=sha(exp/'configs/pilot.json'),source_protocol_sha256=sha(exp/'verl/protocol.json'),
        **config,model=model,model_files_sha256=modelfiles,adapter_path=str(adapter),adapter_files_sha256=adapterfiles,
        adapter_origin=dict(kind='frozen_initial_student',adapter_file_sha256=adapterfiles['adapter_model.safetensors']),
        formal_test_manifest_sha256='t'*64,formal_test_derived_sha256='d'*64,formal_test_source_sha256='s'*64,
        question_sha256=[r['question_sha256'] for r in sources],source_rows=list(range(n)),
        prompt_token_ids_sha256=a.canonical([r['prompt_token_ids'] for r in records],unicode=True),versions={'synthetic':'1'})
    isha=a.canonical(identity,unicode=True)
    frozen='f'*64
    write(run/'metadata.json',dict(identity=identity,identity_sha256=isha))
    write(run/'initial_weights.json',dict(identity_sha256=isha,all_parameter_sha256=frozen))
    batches=[]
    for i,begin in enumerate(range(0,n,4)):
        rr=records[begin:begin+4]
        batch=dict(identity_sha256=isha,begin=begin,end=begin+len(rr),records=rr,records_sha256=a.canonical(rr,unicode=True),
            session_id='s1' if i<2 else 's2',session_initial_parameter_hash=frozen,parameter_versions_before='v',
            parameter_versions_after='v',optimizer_steps=0,generation_seconds=.2,weight_audit_seconds=.01,peak_torch_allocated_gib=1.)
        write(run/f'batches/{i:04d}.json',batch)
        batches.append(batch)
    sessions=[]
    for sid,start,end,status in [('s1',0,2,'paused'),('s2',2,len(batches),'complete')]:
        session=dict(identity_sha256=isha,session_id=sid,status=status,optimizer_steps=0,
            all_parameter_hash_before=frozen,all_parameter_hash_after=frozen,start_completed_batches=start,
            end_completed_batches=end,last_committed_questions=min(end*4,n),initial_weight_audit_seconds=.3,
            final_weight_audit_seconds=.3,model_and_adapter_load_seconds=.4,preparation_seconds=.5,
            recovery_validation_seconds=.1,session_wall_seconds=2.)
        write(run/f'sessions/{sid}.json',session)
        sessions.append(session)
    with (run/'responses.jsonl').open('w',encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record,ensure_ascii=False)+'\n')
    summary=dict(status='complete',identity_sha256=isha,role='student',split='test',questions=n,
        formal_evaluation=True,eligible_for_main_analysis=True,engineering_smoke=False,optimizer_steps=0,
        correct=sum(r['correct'] for r in records),accuracy=sum(r['correct'] for r in records)/n,
        invalid_answers=0,truncated_answers=0,generated_tokens=n*2,
        responses_sha256=sha(run/'responses.jsonl'),initial_weights_file_sha256=sha(run/'initial_weights.json'),
        all_parameter_hash_before=frozen,all_parameter_hash_after=frozen,
        batch_files_sha256={p.name:sha(p) for p in (run/'batches').glob('*.json')},
        session_files_sha256={p.name:sha(p) for p in (run/'sessions').glob('*.json')},
        sessions=2,session_statuses=['paused','complete'],all_committed_batches_have_closing_full_hash=True,
        batch_sessions_without_closing_full_hash=[],generation_seconds=sum(b['generation_seconds'] for b in batches),
        batch_weight_audit_seconds=sum(b['weight_audit_seconds'] for b in batches),peak_torch_allocated_gib=1.)
    for key in a.EVAL_COSTS[2:]:
        summary[key]=sum(s['session_wall_seconds' if key=='known_session_wall_seconds' else key] for s in sessions)
    write(run/'summary.json',summary)
    log=project/'evaluation_audit.log'
    log.write_text('PASS: frozen provenance\n'+json.dumps(dict(status='audit_pass',questions=n,correct=summary['correct'],
                    batches=len(batches),gpu_operations=0),indent=2)+'\nPROCESS_EXIT_CODE: 0\n',encoding='utf-8')
    job=dict(kind='eval',name=run.name,split='test',selection='initial',id='synthetic_eval')
    artifacts={str(p.resolve()):sha(p) for p in run.rglob('*') if p.is_file()}
    def entry(kind):
        return dict(job=dict(job,kind=kind),status='complete',result=dict(questions=n,artifacts_sha256=copy.deepcopy(artifacts)),
                    attempts=[dict(number=1,status='complete',exit_code=0,log=str(log),log_sha256=sha(log))])
    ctx=dict(project=project,exp=exp,indexed={('eval',run.name):entry('eval'),('eval_audit',run.name):entry('eval_audit')},
        protocol_sha='p'*64,manifest_sha='m'*64,manifest=dict(source_sha256={'verl_study/evaluate_verl_formal.py':scriptsha},
        formal_test_manifest_sha256='t'*64),config=config,protocol={'models_and_initialization':modelinit},
        test=dict(derived_sha256='d'*64,source_sha256='s'*64),splits={'test':sources},
        score_answer=lambda text,gold:dict(correct=text==gold,answer_valid=True))
    return ctx,job,records


def reseal(ctx,path):
    for entry in ctx['indexed'].values():
        entry['result']['artifacts_sha256'][str(path.resolve())]=sha(path)


def host_fixture(root,recovery=False):
    watcher=root/'source/watch_verl_formal_host.ps1'
    watcher.parent.mkdir(parents=True)
    watcher.write_text('synthetic original watcher',encoding='utf-8')
    original=root/'host_operations_manifest.json'
    write(original,dict(execution_manifest_sha256='m'*64,watcher_sha256=sha(watcher)))
    status=dict(execution_manifest_sha256='m'*64,watcher_sha256=sha(watcher),
                host_operations_manifest_sha256=sha(original),status='exited')
    write(root/'host-status.json',status)
    if not recovery:
        return original,None
    op=root/'operations/resume'
    prior=op/'prior-host-status.json'
    write(prior,status)
    helper=op/'resume.ps1'
    helper.write_text('synthetic new helper',encoding='utf-8')
    incident=op/'incident.json'
    write(incident,dict(kind='host_sleep_and_clean_boundary_stop',attempt_utc_elapsed_seconds=5918.324976,
        attempt_recorded_monotonic_seconds=886.285212221992,difference_utc_minus_recorded_seconds=5032.039763778008,
        charged_seconds_preserved=865.4756365190115))
    extra=op/'evidence.json'
    write(extra,{'preserve':'unchanged'})
    manifest=op/'operations_manifest.json'
    write(manifest,dict(schema_version=2,execution_manifest_sha256='m'*64,
        helper_relative_path=helper.name,helper_sha256=sha(helper),
        original_host_operations_manifest_sha256=sha(original),prior_host_status_relative=prior.name,
        prior_host_status_sha256=sha(prior),incident_relative=incident.name,
        evidence_sha256={p.name:sha(p) for p in (prior,incident,extra)}))
    write(root/'host-status.json',dict(schema_version=2,status='running',execution_manifest_sha256='m'*64,
        operations_manifest_relative=manifest.relative_to(root).as_posix(),host_operations_manifest_sha256=sha(manifest),
        helper_sha256=sha(helper),watcher_sha256=sha(helper),prior_host_status_sha256=sha(prior)))
    return original,manifest


def append_host_resume(root,original):
    prior_status=a.read(root/'host-status.json')
    op=root/'operations/second-resume'
    prior=op/'prior-host-status.json'
    write(prior,prior_status)
    helper=op/'resume.ps1'
    helper.write_text('synthetic subsequent helper',encoding='utf-8')
    manifest=op/'operations_manifest.json'
    write(manifest,dict(schema_version=2,operation_kind='user_requested_resume',execution_manifest_sha256='m'*64,
        helper_relative_path=helper.name,helper_sha256=sha(helper),
        original_host_operations_manifest_sha256=sha(original),prior_host_status_relative=prior.name,
        prior_host_status_sha256=sha(prior),evidence_sha256={prior.name:sha(prior)}))
    write(root/'host-status.json',dict(schema_version=2,status='running',execution_manifest_sha256='m'*64,
        operations_manifest_relative=manifest.relative_to(root).as_posix(),host_operations_manifest_sha256=sha(manifest),
        helper_sha256=sha(helper),watcher_sha256=sha(helper),prior_host_status_sha256=sha(prior)))
    return manifest


class AnalysisTests(unittest.TestCase):
    def test_two_recovery_operations_keep_old_incident_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original,first=host_fixture(root,recovery=True)
            second=append_host_resume(root,original)
            result=a.host_operations(original,'m'*64,a.Hashes())
            self.assertEqual(len(result['recovery_operations']),2)
            self.assertEqual(result['recovery_operations'][0]['manifest_relative'],first.relative_to(root).as_posix())
            self.assertEqual(result['recovery_operations'][1]['manifest_relative'],second.relative_to(root).as_posix())
            self.assertIsNone(result['recovery_operation']['incident'])
            self.assertEqual(len(result['operational_incidents']),1)
            self.assertEqual(result['recorded_incident_utc_minus_process_seconds'],5032.039763778008)
            self.assertFalse(result['incident_differences_added_to_charged_training_seconds'])

    def test_prior_recovery_operation_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original,first=host_fixture(root,recovery=True)
            append_host_resume(root,original)
            first.write_bytes(first.read_bytes()+b' ')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                a.host_operations(original,'m'*64,a.Hashes())

    def test_recovery_operation_cycle_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original,manifest=host_fixture(root,recovery=True)
            status=a.read(root/'host-status.json')
            prior=manifest.parent/'prior-host-status.json'
            # Reject the cycle before following its self-referential digest.
            write(prior,status)
            operation=a.read(manifest)
            operation['prior_host_status_sha256']=sha(prior)
            operation['evidence_sha256'][prior.name]=sha(prior)
            write(manifest,operation)
            status['prior_host_status_sha256']=sha(prior)
            status['host_operations_manifest_sha256']=sha(manifest)
            write(root/'host-status.json',status)
            with self.assertRaisesRegex(RuntimeError,'Cycle in host operations history'):
                a.host_operations(original,'m'*64,a.Hashes())

    def test_original_host_operations_remains_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            original,_=host_fixture(Path(temp))
            result=a.host_operations(original,'m'*64,a.Hashes())
            self.assertEqual(result['status'],'manifest_verified')
            self.assertIsNone(result['host_overhead_seconds'])
            self.assertNotIn('recovery_operation',result)

    def test_recovery_host_operations_preserves_unmodified_costs(self):
        with tempfile.TemporaryDirectory() as temp:
            original,_=host_fixture(Path(temp),recovery=True)
            result=a.host_operations(original,'m'*64,a.Hashes())
            self.assertEqual(result['status'],'original_and_recovery_manifests_verified')
            self.assertIsNone(result['host_overhead_seconds'])
            recovery=result['recovery_operation']
            self.assertEqual(recovery['utc_minus_recorded_process_seconds'],5032.039763778008)
            self.assertFalse(recovery['added_to_charged_training_seconds'])
            self.assertEqual(recovery['incident']['charged_seconds_preserved'],865.4756365190115)

    def test_recovery_helper_evidence_and_manifest_tampering_rejected(self):
        for filename in ('resume.ps1','evidence.json','operations_manifest.json'):
            with self.subTest(file=filename),tempfile.TemporaryDirectory() as temp:
                original,manifest=host_fixture(Path(temp),recovery=True)
                target=manifest.parent/filename
                target.write_bytes(target.read_bytes()+b' ')
                with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                    a.host_operations(original,'m'*64,a.Hashes())

    def test_recovery_paths_must_remain_inside_evidence_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original,_=host_fixture(root,recovery=True)
            status=a.read(root/'host-status.json')
            status['operations_manifest_relative']='../outside.json'
            write(root/'host-status.json',status)
            with self.assertRaisesRegex(RuntimeError,'escapes evidence root'):
                a.host_operations(original,'m'*64,a.Hashes())

    def test_recovery_original_reference_and_unpinned_incident_rejected(self):
        for change in ('original_reference','unpinned_incident','invalid_difference','wrong_prior_watcher'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                original,manifest=host_fixture(root,recovery=True)
                operation=a.read(manifest)
                status=a.read(root/'host-status.json')
                if change=='original_reference': operation['original_host_operations_manifest_sha256']='0'*64
                elif change=='unpinned_incident': operation['evidence_sha256'].pop('incident.json')
                elif change=='invalid_difference':
                    incident_path=manifest.parent/'incident.json'
                    incident=a.read(incident_path);incident['difference_utc_minus_recorded_seconds']=1
                    write(incident_path,incident)
                    operation['evidence_sha256']['incident.json']=sha(incident_path)
                else:
                    prior_path=manifest.parent/'prior-host-status.json'
                    prior=a.read(prior_path);prior['watcher_sha256']='0'*64
                    write(prior_path,prior)
                    operation['prior_host_status_sha256']=sha(prior_path)
                    operation['evidence_sha256'][prior_path.name]=sha(prior_path)
                    status['prior_host_status_sha256']=sha(prior_path)
                write(manifest,operation)
                status['host_operations_manifest_sha256']=sha(manifest)
                write(root/'host-status.json',status)
                with self.assertRaises(RuntimeError): a.host_operations(original,'m'*64,a.Hashes())

    def test_crossed_sampler_matches_slow_reference_and_chunk_invariance(self):
        data=np.array([[[1,0,1,0],[0,1,0,1],[1,1,0,0]],[[0,1,1,0],[1,1,0,0],[0,0,1,1]]])
        base=np.array([0,1,0,1])
        actual,actualbase=a.crossed_bootstrap(data,base,replicates=31,seed=12,chunk_size=7)
        alternate,alternatebase=a.crossed_bootstrap(data,base,replicates=31,seed=12,chunk_size=19)
        np.testing.assert_array_equal(actual,alternate)
        np.testing.assert_array_equal(actualbase,alternatebase)
        rng=np.random.default_rng(12)
        seeds=rng.integers(0,3,size=(31,3))
        questions=rng.integers(0,4,size=(31,4))
        reference=[]
        for ss,qq in zip(seeds,questions):
            reference.append([sum(int(data[m,s,q]) for s in ss for q in qq)/12*100 for m in range(2)])
        np.testing.assert_array_equal(actual,np.array(reference))
        np.testing.assert_array_equal(actualbase,np.array([sum(base[q] for q in qq)/4*100 for qq in questions]))

    def test_twenty_thousand_replicates_known_constant_contrasts(self):
        data=np.zeros((5,3,13),dtype=np.int8)
        data[0]=1;data[2]=1
        result=a.statistical_results(data,np.zeros(13,dtype=np.int8))
        self.assertEqual(len(result['contrasts']),15)
        self.assertEqual(result['bootstrap']['replicates'],20000)
        self.assertEqual(result['methods'][0]['sample_sd_percentage_points'],0)
        for row in result['contrasts']:
            self.assertEqual(row['descriptive_unadjusted_95_percent_interval'],[row['difference_percentage_points']]*2)
        self.assertEqual(result['interpretation']['mean_tied_methods'],['rl','joint'])
        self.assertFalse(result['interpretation']['unique_confirmed_winner'])

    def test_sample_sd_seed_counts_pairing_and_signs(self):
        data=np.array([[[0]*4,[1,1,0,0],[1]*4],[[1]*4,[1]*4,[1]*4]])
        base=np.array([0,0,1,1])
        result=a.statistical_results(data,base,methods=['a','b'],seeds=[42,43,44],replicates=100)
        self.assertEqual(result['methods'][0]['mean_accuracy_percent'],50)
        self.assertEqual(result['methods'][0]['sample_sd_percentage_points'],50)
        self.assertEqual(result['methods'][0]['minimum_accuracy_percent'],0)
        self.assertEqual(result['methods'][0]['maximum_accuracy_percent'],100)
        self.assertEqual(result['contrasts'][0]['difference_percentage_points'],-50)
        self.assertEqual(result['methods'][0]['seeds'][1]['paired_vs_initial'],dict(gains=2,losses=2,both_correct=0,both_wrong=0))
        with self.assertRaises(RuntimeError):
            a.crossed_bootstrap(np.full((2,3,4),.5),base)

    def test_complete_plan_and_missing_invalid_audit_gates(self):
        protocol=dict(methods=a.METHODS,seeds=a.SEEDS,run_order={'runs':[
            dict(order=i+1,method=m,seed=s,name=f'{m}_{s}') for i,(m,s) in enumerate(__import__('itertools').product(a.METHODS,a.SEEDS))]})
        jobs=a.planned_jobs(protocol)
        self.assertEqual(len(jobs),122)
        self.assertEqual(sum(j['kind']=='eval' for j in jobs),46)
        identity=dict(jobs=jobs,protocol_sha256='p',execution_manifest_sha256='m',project_root='/synthetic')
        state=dict(status='complete',identity=identity,identity_sha256=a.canonical(identity),jobs=[
            dict(job=j,status='complete',result={'artifacts_sha256':{'placeholder':'x'}}) for j in jobs])
        self.assertEqual(len(a.validate_state(state,identity)),122)
        for change in ('running','missing_audit','changed_identity','missing_result'):
            bad=copy.deepcopy(state)
            if change=='running': bad['status']='running'
            elif change=='missing_audit': bad['jobs'].pop(1)
            elif change=='changed_identity': bad['identity_sha256']='0'*64
            else: bad['jobs'][1]['result']={}
            with self.assertRaises(RuntimeError,msg=change): a.validate_state(bad,identity)

    def test_full_1319_evaluation_last_three_resume_and_unicode(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,job,records=fixture(Path(temp))
            result=a.evaluation_result(ctx,job,{},a.Hashes())
            self.assertEqual(result['correct'],sum(r['correct'] for r in records))
            self.assertEqual(result['questions'],1319)
            self.assertEqual(result['generated_tokens'],2638)
            self.assertEqual(len(a.rows(ctx['exp']/'runs/synthetic_test/responses.jsonl')),1319)

    def test_native_adapter_readme_is_audited_but_not_an_evaluator_model_file(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,job,_=fixture(Path(temp))
            identity=a.read(ctx['exp']/'runs/synthetic_test/metadata.json')['identity']
            job.update(selection='primary',train_name='synthetic_train')
            selection=dict(adapter_directory=identity['adapter_path'])
            native_files=dict(identity['adapter_files_sha256'],**{'README.md':'r'*64})
            training={'synthetic_train':dict(selections={'primary':selection},
                audited_selections={'selected_primary':dict(selection,adapter_files_sha256=native_files)})}
            self.assertEqual(a.evaluation_result(ctx,job,training,a.Hashes())['questions'],1319)

    def test_stale_receipt_and_semantically_incorrect_summary_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,job,_=fixture(Path(temp))
            path=ctx['exp']/'runs/synthetic_test/summary.json'
            summary=a.read(path);summary['accuracy']=.123
            write(path,summary)
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'): a.evaluation_result(ctx,job,{},a.Hashes())
            reseal(ctx,path)
            with self.assertRaisesRegex(RuntimeError,'Evaluation accuracy'): a.evaluation_result(ctx,job,{},a.Hashes())

    def test_missing_audit_receipt_and_failed_audit_log_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,job,_=fixture(Path(temp))
            aud=ctx['indexed'][('eval_audit',job['name'])]
            original=copy.deepcopy(aud['result']['artifacts_sha256'])
            aud['result']['artifacts_sha256']={}
            with self.assertRaisesRegex(RuntimeError,'Missing evidence manifest'): a.evaluation_result(ctx,job,{},a.Hashes())
            aud['result']['artifacts_sha256']=original
            Path(aud['attempts'][0]['log']).write_text('{"status":"audit_fail"}\nPROCESS_EXIT_CODE: 1\n')
            with self.assertRaisesRegex(RuntimeError,'successful evaluation audit'): a.evaluation_result(ctx,job,{},a.Hashes())

    def test_missing_session_and_wrong_question_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx,job,_=fixture(Path(temp))
            path=ctx['exp']/'runs/synthetic_test/sessions/s2.json'
            path.unlink()
            with self.assertRaises(FileNotFoundError): a.evaluation_result(ctx,job,{},a.Hashes())
        with tempfile.TemporaryDirectory() as temp:
            ctx,job,_=fixture(Path(temp))
            ctx['splits']['test'][0]['gold']='0'
            with self.assertRaisesRegex(RuntimeError,'another question'): a.evaluation_result(ctx,job,{},a.Hashes())

    def test_unknown_costs_are_never_zero_or_double_counted(self):
        one=dict(process_wall_seconds=10.,whole_job_wall_seconds=12.,supervisor_preparation_seconds=1.,postprocess_verification_seconds=1.)
        unknown=dict(one,number=2,process_wall_seconds=None,whole_job_wall_seconds=None,cost_accounting='supervisor lost clock')
        jobs=[dict(job_id='train',kind='train',attempts=[dict(one,number=1),unknown]),
              dict(job_id='audit',kind='train_audit',attempts=[dict(one,number=1)])]
        result=a.cost_totals(jobs,[dict(known_wall_seconds=25.,ended_utc='x')])
        self.assertEqual(result['known_attempt_totals']['whole_job_wall_seconds'],24.)
        self.assertEqual(len(result['unknown_attempt_costs']),2)
        self.assertFalse(result['all_whole_job_costs_known'])
        self.assertIsNone(result['supervisor_overhead_outside_jobs_seconds'])

    def test_manifest_and_evidence_missing_fail_before_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError): a.analyze(Path(temp))
            file=Path(temp)/'evidence.json'
            write(file,{'x':1})
            entry=dict(job={'id':'audit'},result={'artifacts_sha256':{str(file):sha(file)}})
            a.verify_receipt(Path(temp),entry,[file],a.Hashes())
            write(file,{'x':2})
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'): a.verify_receipt(Path(temp),entry,[file],a.Hashes())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests))
    report=dict(status='pass' if result.wasSuccessful() else 'fail',gpu_operations=0,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-started,
        analysis_source_sha256=sha(Path(a.__file__)),test_source_sha256=sha(Path(__file__)),numpy_version=np.__version__,
        scope='Synthetic bootstrap reference, exact outcomes, evidence/identity rejection, full 1319-row two-session fixture, missing costs, original and recovered Windows host operations, containment and tamper rejection. No official model inference or formal result analysis.')
    write(args.output,report)
    print(json.dumps(report,indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
