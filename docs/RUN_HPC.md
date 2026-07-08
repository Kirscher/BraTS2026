# Running a GoAT training on the HPC ("press go")

This is the exact sequence for the mentor to launch `nnUNetTrainerGoAT` on the HPC after a
`git pull`. The Python package wires everything except the two things that are **intentionally
human-gated**: the hyperparameter numbers and the train-launch approval.

## 0. Prerequisites (once)

```bash
pip install -e ".[mri,train,dev]"          # installs brats2026 + nnU-Net v2 + torch
export nnUNet_raw=/path/to/work/nnUNet_raw
export nnUNet_preprocessed=/path/to/work/nnUNet_preprocessed
export nnUNet_results=/path/to/work/nnUNet_results
```

All three roots live under a writable `work/` — the NAS is read-only.

## 1. Make nnU-Net see the custom trainer

nnU-Net only discovers trainers inside its own package tree, so register ours once:

```bash
brats2026 install-trainer
```

This drops an import shim at `nnunetv2/training/nnUNetTrainer/variants/brats2026_goat.py` so
`-tr nnUNetTrainerGoAT` resolves.

## 2. Convert + preprocess the data

The discovery manifest → nnU-Net raw (`Dataset501_BraTSGoAT`, region-based ET/TC/WT declared in
`dataset.json`), then nnU-Net's ResEnc-L planner/preprocessor:

```bash
brats2026 discover --task task3 --split train --output work/manifests/train.jsonl
# data-pipeline: apply the conversion plan (symlinks, dataset.json) into $nnUNet_raw
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncL -c 3d_fullres --verify_dataset_integrity
```

## 3. Fill the config (ai-specialist, human-gated)

`configs/train.yaml` ships as an **intentionally invalid stub** — a run refuses to start while it
is unfilled. The ai-specialist fills the provenance header (including the real
`dataset_fingerprint` from the preprocessed data) and every `# SPECIALIST:` hook. Point the
trainer at the filled file:

```bash
export BRATS_GOAT_TRAIN_CONFIG=/path/to/configs/train.yaml
```

At `initialize()` the trainer loads this file, validates it (`assert_valid_config`), and applies
`initial_lr / weight_decay / num_epochs / oversample_foreground_percent`. If the file is missing
or unfilled it raises **before** training — no silent default run.

## 4. Launch (the "go")

```bash
# one fold, or loop 0..4 for the 5-fold cross-validation
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerGoAT -p nnUNetResEncUNetLPlans
```

`brats2026.nnunet.plan.train_command()` / `kfold_train_commands()` emit these argv verbatim.

## What is and isn't wired yet

- **Wired:** config→trainer injection (env var + validation + apply), region-based ET/TC/WT
  training (from `dataset.json`), trainer discovery, the exact launch commands.
- **Still TODO in `nnUNetTrainerGoAT.initialize()`** (marked in code): the domain-balanced
  sampler (`domain_sampling_weights`), the domain-randomization transforms
  (intensity/contrast/noise), and the region loss weights. Until those land, a launch is a
  **configured region-based nnU-Net baseline**, not yet the full GoAT recipe.
- **Self-training (SSL) and MAE pre-training** are separate offline runners
  (`brats2026.ssl` — see `self_training_plan` + `brats2026 ssl-select`; and `brats2026.ssl.mae`),
  not part of a single `nnUNetv2_train`.
