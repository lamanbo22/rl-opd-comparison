# Reproduction

Run commands from the repository root. Statistical reproduction uses only a CPU. Training uses the original Linux/WSL environment and one RTX 5090.

## 1. Verify the reported results

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/reproduce_results.py --check
python -m unittest discover -s tests -v
```

This regenerates accuracy tables, paired intervals, and cost tables from the released outcomes and metadata. The bootstrap draws match the original analysis exactly. See [results and metrics](results.md) for the data format.

## 2. Prepare the data

```bash
python scripts/prepare_assets.py
python scripts/check_environment.py
```

Preparation downloads the pinned GSM8K training and test files, selects the original training/development rows, and checks every derived file against its recorded SHA256. Files go to `artifacts/assets/`. Existing files with different contents are rejected.

For offline preparation, pass `--train-source` and `--test-source` with the original raw files. Expected revisions and hashes are in [assets.json](../configs/assets.json).

## 3. Prepare the training assets

The model download needs roughly 12 GB, plus space for the environment, adapters, checkpoints, and generated responses. Use Python 3.12.3 on Linux/WSL with a working NVIDIA driver and one visible RTX 5090. The frozen trainer checks its original hardware and software versions.

```bash
python -m pip install huggingface-hub==1.32.0
python scripts/prepare_assets.py --models --download-adapter
python scripts/check_environment.py --models
python scripts/run_reproduction.py prepare
```

The model revisions and initial adapter are fixed. The adapter is available as a [release asset](https://github.com/lamanbo22/rl-opd-comparison/releases/tag/v0.2.0); its checksum is verified on download. An existing exact adapter can be supplied with `--adapter-dir` instead.

Preparation creates an isolated working directory at `artifacts/reproduction/`. It copies the experiment source and changes model paths and dependent configuration hashes in that copy. A `reproduction_manifest.json` records the original and runtime identities. The committed study protocol and results stay unchanged.

## 4. Install and check the training environment

Install **inside the generated workspace**, so the editable verl installation resolves to the checkout expected by the trainer:

```bash
python3.12 artifacts/reproduction/experiments/verl/setup_verl.py

artifacts/reproduction/.venv-verl/bin/python scripts/check_environment.py \
  --training --verl-source artifacts/reproduction/external/verl

artifacts/reproduction/.venv-verl/bin/python scripts/run_reproduction.py train \
  --method rl --seed 42 --name reproduction_rl42 --check-only
```

The setup script expects `/usr/bin/python3.12`. The preflight checks the exact Python and package versions, model hashes, verl commit and import path, and visible GPU. `--check-only` checks training inputs without loading a model or starting training.

## 5. Train and evaluate

Start one run by removing `--check-only`:

```bash
artifacts/reproduction/.venv-verl/bin/python scripts/run_reproduction.py train \
  --method rl --seed 42 --name reproduction_rl42
```

Methods are `rl`, `opd`, `joint`, `opd_then_rl`, and `rl_then_opd`. Seeds are 42, 43, and 44. Each run uses the original 7,200-second ceiling. For a short engineering check, use a different run name and add `--validation`. Cleanly paused runs can use `--resume` with the same method, seed, and name.

Evaluate the resulting adapter:

```bash
artifacts/reproduction/.venv-verl/bin/python scripts/evaluate_adapter.py \
  --split test \
  --adapter artifacts/reproduction/experiments/runs/reproduction_rl42/adapter_final \
  --output artifacts/evaluations/reproduction_rl42_test
```

Omit `--adapter` to evaluate the initial student. Use `--split dev --limit 8` for a small development check. Official test evaluation always uses all 1,319 questions. Replace `--output ...` with `--check-only` to verify inputs without inference.

These commands create new reproduction runs. The historical supervisor and forensic analysis scripts retain their original archive requirements. New training and evaluation wrappers have CPU tests and were checked against the original settings; the complete GPU workflow was not rerun for this release. The recorded study results are independently reproducible through step 1.

## Tests and source records

GitHub Actions runs the public command tests, the original synthetic analysis/recovery tests, the supervisor tests, checksum verification, and table/figure reproduction on a CPU.

The original source-and-results release is preserved at [commit c8e2d01](https://github.com/lamanbo22/rl-opd-comparison/tree/c8e2d018e59cce56da6e15ee1cba2b0fae50fe95). Files covered by the frozen study manifests remain byte-identical. Later analysis-comment edits and public tooling are recorded in the current release manifest.
