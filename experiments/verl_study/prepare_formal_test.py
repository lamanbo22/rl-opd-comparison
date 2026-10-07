"""Freeze the official GSM8K test split from the original immutable dataset revision.

Reads cached raw bytes if present. --download-if-missing authorizes exactly one
small official test download; it never fetches a model or a floating revision.
Existing outputs must match exactly. No model is loaded and no inference runs.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def question_sha(question):
    return hashlib.sha256(' '.join(question.split()).casefold().encode()).hexdigest()


def save_identical(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Existing frozen file differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name+'.tmp-', dir=path.parent)
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Exclusive hard-link commit prevents replacing a concurrently created file.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def derive(project, source_path):
    exp = project/'experiments'
    frozen_path = exp/'verl_study/protocol.json'
    frozen, config = read(frozen_path), read(exp/'configs/pilot.json')
    assert frozen['status'] == 'frozen' and sha(exp/'configs/pilot.json') == frozen['source_config_sha256']
    original_dir = project/'work/text-validation-2026-09-30/protocol_v2'
    original_path = original_dir/'manifest.json'
    original = read(original_path)
    dataset = original['dataset']
    assert dataset['repository'] == 'https://github.com/openai/grade-school-math'
    assert dataset['revision'] == '3101c7d5072418e28b9008a6636bde82a006892c'
    assert re.fullmatch('[0-9a-f]{40}', dataset['revision'])
    source_url = f"https://raw.githubusercontent.com/openai/grade-school-math/{dataset['revision']}/grade_school_math/data/test.jsonl"
    assert dataset['source_url'] == source_url.replace('/test.jsonl','/train.jsonl')
    train_source = original_dir/'gsm8k_train_source.jsonl'
    assert sha(train_source) == dataset['sha256']
    original_rows = rows(train_source)
    assert len(original_rows) == dataset['rows'] == 7473
    splits = {}
    for split, count in (('train',6961),('dev',512)):
        path = exp/config['data'][split+'_file']
        assert sha(path) == config['data'][split+'_sha256']
        splits[split] = rows(path)
        assert len(splits[split]) == len({r['question_sha256'] for r in splits[split]}) == count
    for name, expected in config['reward']['code_sha256'].items():
        assert sha(exp/'reward'/name) == expected
    sys.path.insert(0, str(exp/'reward'))
    from text_reward import canonical, parse_gold
    # Verify that both existing splits really derive from the pinned original train bytes.
    for row in splits['train']+splits['dev']:
        original_row = original_rows[row['source_row']]
        assert row['question'] == original_row['question']
        assert row['question_sha256'] == question_sha(original_row['question'])
        assert row['gold'] == canonical(parse_gold(original_row['answer']))
    source = rows(source_path)
    assert len(source) == 1319, 'Official test row count differs'
    derived = [{'source_split':'test','source_row':i,'question_sha256':question_sha(row['question']),
                'question':row['question'],'gold':canonical(parse_gold(row['answer']))}
               for i,row in enumerate(source)]
    assert all(isinstance(row['question'],str) and row['question'] for row in source)
    assert len({r['question_sha256'] for r in derived}) == 1319, 'Duplicate official-test question IDs'
    train_ids, dev_ids = ({r['question_sha256'] for r in splits[s]} for s in ('train','dev'))
    test_ids = {r['question_sha256'] for r in derived}
    assert not train_ids & dev_ids
    overlaps = {'test_train':len(test_ids & train_ids),'test_dev':len(test_ids & dev_ids),
                'train_dev':len(train_ids & dev_ids)}
    content = ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in derived).encode()
    manifest = {'schema_version':1,'status':'frozen' if not any(overlaps.values()) else 'overlap_requires_review',
        'dataset_repository':dataset['repository'],'dataset_revision':dataset['revision'],
        'source_url':source_url,'source_file':'source_test.jsonl','source_sha256':sha(source_path),
        'source_bytes':source_path.stat().st_size,'split':'test','questions':1319,
        'derived_file':'test.jsonl','derived_sha256':hashlib.sha256(content).hexdigest(),
        'question_normalization':'Whitespace collapsed, casefolded, UTF-8 SHA256',
        'gold_derivation':'Frozen reward/text_reward.py canonical(parse_gold(original answer))',
        'rows_kept':'All 1319 official test rows in source order; none dropped or selected by score.',
        'overlap_counts':overlaps,'source_question_sha256':[r['question_sha256'] for r in derived],
        'training_and_development':config['data'],'reward':config['reward'],
        'original_dataset_manifest_sha256':sha(original_path),'original_train_source_sha256':sha(train_source),
        'frozen_study_protocol_sha256':sha(frozen_path),'frozen_study_at_utc':frozen['frozen_at_utc'],
        'materializer_sha256':sha(Path(__file__)),'model_inference_performed':False,'gpu_operations':0}
    return derived, content, manifest


def verify_frozen_test(project):
    """Read-only, CPU-only verification used by the formal evaluator."""
    directory = Path(project)/'experiments/verl_study/formal_test'
    saved = read(directory/'manifest.json')
    derived, content, actual = derive(Path(project), directory/'source_test.jsonl')
    assert saved == actual, 'Official-test manifest/source/protocol identity differs'
    assert saved['status'] == 'frozen', 'Test overlap requires explicit review before inference'
    assert (directory/'test.jsonl').read_bytes() == content, 'Derived official test changed'
    return derived, saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--download-if-missing',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    project = args.project_root.resolve()
    directory = project/'experiments/verl_study/formal_test'
    raw = directory/'source_test.jsonl'
    if args.check_only:
        data, manifest = verify_frozen_test(project)
        print(json.dumps({'status':'check_pass','questions':len(data),'manifest_sha256':sha(directory/'manifest.json'),
                          'source_sha256':manifest['source_sha256'],'overlap_counts':manifest['overlap_counts'],
                          'gpu_operations':0},indent=2),flush=True)
        return
    if not raw.exists():
        assert args.download_if_missing, 'Raw official test is absent; use --download-if-missing for the pinned official URL'
        original = read(project/'work/text-validation-2026-09-30/protocol_v2/manifest.json')['dataset']
        assert original['revision'] == '3101c7d5072418e28b9008a6636bde82a006892c'
        url = original['source_url'].replace('/train.jsonl','/test.jsonl')
        expected = f"https://raw.githubusercontent.com/openai/grade-school-math/{original['revision']}/grade_school_math/data/test.jsonl"
        assert url == expected and read(project/'experiments/verl_study/protocol.json')['status'] == 'frozen'
        with urllib.request.urlopen(url,timeout=60) as response:
            assert response.status == 200 and response.url == url
            content = response.read(5*1024*1024+1)
        assert 0 < len(content) <= 5*1024*1024
        parsed = [json.loads(line) for line in content.splitlines() if line.strip()]
        assert len(parsed) == 1319
        save_identical(raw,content)
    _, content, manifest = derive(project,raw)
    save_identical(directory/'test.jsonl',content)
    save_identical(directory/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_question_sha256'},indent=2),flush=True)
    assert manifest['status'] == 'frozen', 'Overlap detected; preserved all rows and manifest for review'


if __name__ == '__main__':
    main()
