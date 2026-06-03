# BraTS2026

[![tests](https://github.com/Kirscher/BraTS2026/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Kirscher/BraTS2026/actions/workflows/tests.yml)

Baseline tooling and experiment scaffolding for the MICCAI BraTS 2026 cluster:
https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097

The working dataset mirror is expected at:

```bash
/mnt/CPS-RADT/datasets/MICCAI/2026/BraTS2026
```

## Quick Start

```bash
source ~/venvs/synapseclient/bin/activate
python -m pip install -e .
python -m brats2026.cli tasks
python -m brats2026.cli discover --task task1 --split train --limit 5 --output work/manifests/task1_train_sample.jsonl
```

MRI preprocessing requires optional dependencies:

```bash
python -m pip install -e '.[mri]'
python -m brats2026.cli preprocess-mri \
  --manifest work/manifests/task1_train_sample.jsonl \
  --modalities t1n,t1c,t2f,t2w \
  --output-dir work/preprocessed/task1_sample
```

See `docs/PLAN.md` for the task summary, important rules, and the baseline strategy.
