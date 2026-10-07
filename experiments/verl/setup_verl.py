"""Reproduce the tested, isolated local verl environment from a pinned source."""
import subprocess
import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[2]
assert (root/'experiments/configs/pilot.json').is_file(), root
commit = '8718ca30a3f002f93b7c4fd99b9b2506718681bc'
source = root/'external/verl'
setup = root/'.setup-tools'
def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)
if not setup.exists():
    run([sys.executable, '-m', 'venv', setup])
run([setup/'bin/python', '-m', 'pip', 'install', 'uv==0.11.11'])
if not source.exists():
    run(['git','clone','https://github.com/verl-project/verl.git',source])
actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
if actual != commit:
    assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip(), 'Preserve local verl modifications before changing revision.'
    run(['git','-C',source,'fetch','origin',commit])
    run(['git','-C',source,'checkout','--detach',commit])
env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(root/'.venv-verl'))
run([setup/'bin/uv','sync','--frozen','--extra','fsdp','--extra','vllm','--python','/usr/bin/python3.12'],cwd=source,env=env)
# The upstream lock violates these two distributions' declared requirements.
run([setup/'bin/uv','pip','install','--python',root/'.venv-verl/bin/python','numpy==2.3.5','nvidia-cusparselt-cu13==0.8.1'])
run([setup/'bin/uv','pip','check','--python',root/'.venv-verl/bin/python'])
with (root/'experiments/verl/requirements-frozen.txt').open('w') as stream:
    run([setup/'bin/uv','pip','freeze','--python',root/'.venv-verl/bin/python'],stdout=stream)
print('Ready:',root/'.venv-verl/bin/python')
