# Running a **v2** (warm-started, domain-randomised) training on the HPC

This is the launch sequence for `nnUNetTrainerGoATv2`. It is the companion to
[`RUN_HPC.md`](RUN_HPC.md), which covers the v1 trainer — read that one first if you have not
run v1 yet, because v2 **warm-starts from a converged v1 checkpoint** and reuses its plans.

> **v2 does not disturb a v1 run in flight.** Different trainer class name, different shim
> filename, different env var, different results directory. You can launch v2 while v1 folds are
> still training on the same box.

## Why v2 exists

Hyperparameter tuning has saturated: submission 1 already sits at top-leaderboard parity and
successive runs move the score by ~1e-2 to 1e-3. Task 3 is scored on *generalisation* across five
cohorts, and two of them (SSA, PED) have no training data at all. The remaining lever is
augmentation strength, not optimiser tuning. So v2

1. **warm-starts** from an already-converged fold (`init_weights_from`), so a fold v1 has finished
   can be improved without paying for a from-scratch run, and
2. **inherits `nnUNetTrainerDA5`**, DKFZ's own aggressive-augmentation variant, rather than
   hand-tuning probabilities on top of the stock pipeline.

The published anchor for (2) is SynthSeg (Billot et al., *Medical Image Analysis* 2023), which
shows that randomising contrast, resolution and bias field forces contrast-agnostic features.
DA5 is the pragmatic route to the same place: it already ships
`BrightnessGradientAdditiveTransform` and `LocalGammaTransform` at p=0.3 — spatially-varying
intensity corruption — plus median filtering, sharpening, cutout and much wider contrast/low-res
ranges, and it has been exercised in DKFZ's own challenge entries. Only the warm start, the
config gate and the retuning hooks are ours.

> Because the base class changed, some `da_*` hooks may no longer bind (DA5 uses different
> transform classes for the same intent). **§4's smoke test is what tells you which bound** —
> do not skip it.

## 0. Prerequisites

Same as v1 — the package installed, and the three nnU-Net roots exported under a writable
`work/` (the NAS is read-only):

```bash
pip install -e ".[mri,train,dev]"
export nnUNet_raw=/path/to/work/nnUNet_raw
export nnUNet_preprocessed=/path/to/work/nnUNet_preprocessed
export nnUNet_results=/path/to/work/nnUNet_results
```

**v2 needs no new data work.** It reuses `Dataset501_BraTSGoAT` and the same
`nnUNetResEncUNetLPlans` / `3d_fullres` produced for v1. Do not re-run the planner: the warm
start is shape-compatible only if the plans are identical.

## 1. Register the v2 trainer

```bash
brats2026 install-trainer-v2
```

This writes `nnunetv2/training/nnUNetTrainer/variants/brats2026_goat_v2.py`, a two-line import
shim so `-tr nnUNetTrainerGoATv2` resolves. The filename is deliberately distinct from v1's
`brats2026_goat.py` — overwriting that file would swap the trainer out from under a running job.

## 2. Fill and bind the config

`configs/train_v2.yaml` carries all 18 `# SPECIALIST:` hooks. Three provenance fields
(`git_sha`, `date`, `dataset_fingerprint`) are data/environment-bound and cannot exist until the
data is preprocessed, so bind them here:

```bash
brats2026 stamp-config --config configs/train_v2.yaml \
                       --raw "$nnUNet_raw/Dataset501_BraTSGoAT"
```

`stamp-config` validates against the `kind:` the file declares, so it checks `train_v2` against
the v2 hook tuple automatically. Pass `--kind train_v2` to force it.

Until stamped, `assert_valid_config` refuses the file and the trainer will not launch — by
design, so no run happens against a config not bound to the data it was tuned on.

### Point the trainer at it

```bash
export BRATS_GOAT_V2_TRAIN_CONFIG=/abs/path/to/configs/train_v2.yaml
```

Deliberately **not** the same variable as v1's `BRATS_GOAT_TRAIN_CONFIG`, so both can live in one
shell without the v2 config hijacking a v1 run.

## 3. Enable the warm start

In `configs/train_v2.yaml`, uncomment `init_weights_from` and point it at a **converged fold of
our own GoAT run**:

```yaml
init_weights_from: "/…/nnUNet_results/Dataset501_BraTSGoAT/nnUNetTrainerGoAT__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth"
```

Leaving it unset is legal and means *train from scratch*.

**On the GoAT rule.** Warm-starting from our own GoAT-trained weights is allowed: the rule forbids
external data and weights derived from a **prior BraTS**, not resuming one of our own runs. The
hook is named `init_weights_from` rather than `pretrained_*` precisely because
`compliance.PRIOR_BRATS_TOKENS` flags those substrings — a compliant name must not trip our own
gate.

**What cannot happen silently.** Before loading, `plan_weight_transfer` compares every tensor
shape and `assert_transferable` raises if nothing matches or if any shape differs. The failure
mode that costs GPU-days — pointing at a checkpoint from different plans, getting a near-empty
load, and training from what is effectively a random init without noticing — is an exception, not
a warning. Set `init_weights_strict: true` to additionally require full coverage.

## 4. ⚠ Smoke-test the transform pipeline BEFORE committing GPU time

The pure functions in `trainer_v2.py` are unit-tested, but the glue against
`batchgeneratorsv2` runs **only** where nnU-Net is installed — never in CI, never on the dev
laptop. Verify it on the HPC first, it takes seconds:

```bash
python - <<'PY'
from brats2026.nnunet.trainer_v2 import load_goat_v2_config, nnUNetTrainerGoATv2
import os
cfg = load_goat_v2_config(os.environ["BRATS_GOAT_V2_TRAIN_CONFIG"])
nnUNetTrainerGoATv2.goat_config = cfg
t = nnUNetTrainerGoATv2.get_training_transforms(
    patch_size=cfg.patch_size, rotation_for_DA=(-0.5, 0.5),
    deep_supervision_scales=None, mirror_axes=(0, 1, 2), do_dummy_2d_data_aug=False,
)
print(t)
PY
```

You are looking for the `[GoATv2] augmentation retuned: {...}` line listing **every** `da_*` hook.
If it prints `WARNING: hooks that matched no transform: [...]`, an upstream transform has been
renamed and that override is silently inert — **fix that before training**, or the v1-vs-v2
comparison measures nothing.

## 5. Launch

```bash
nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainerGoATv2 -p nnUNetResEncUNetLPlans
```

Or build the argv from the package so the trainer name is never hand-typed:

```python
from brats2026.nnunet.plan import train_v2_command
print(" ".join(train_v2_command(fold=0)))
```

At `initialize()` the trainer resolves the config, applies `initial_lr` / `weight_decay` /
`num_epochs` / `oversample_foreground_percent` **before** `super().initialize()` (so the optimizer
and dataloaders pick them up), then performs the warm start and logs its coverage.

## 6. Where the results land

```
work/nnUNet_results/Dataset501_BraTSGoAT/
  nnUNetTrainerGoAT__nnUNetResEncUNetLPlans__3d_fullres/     # v1 — untouched
  nnUNetTrainerGoATv2__nnUNetResEncUNetLPlans__3d_fullres/   # v2
```

nnU-Net derives this directory from `<trainer>__<plans>__<configuration>`, so the distinct class
name is what keeps v2 from writing into the folds v1 is still training.

## 7. Scoring the result

v2 must be judged by `validation-metrics` on the **same fold** as its parent v1 run, or the
comparison is meaningless:

```bash
brats2026 evaluate --pred <v2 predictions> --gt <fold-0 labels> --output work/reports/v2_fold0.json
```

Compare against the v1 report for that fold. The metric that matters is **worst-cohort**, not
mean — v2's entire thesis is generalisation, so a v2 that raises the mean while lowering the
worst cohort has failed on its own terms.

## What v2 changes at inference: nothing

Same architecture, same plans, same patch size. The warm-started checkpoint is a drop-in
replacement for v1's in `configs/infer.yaml`, so the marginal inference cost against the 8h
container budget is zero. `inference-packager`'s profiling remains authoritative before adding
any ensemble member.

## Rollback

v2 is additive. To abandon it: delete the
`nnUNetTrainerGoATv2__…` results directory and
`nnunetv2/training/nnUNetTrainer/variants/brats2026_goat_v2.py`. Nothing in v1's path, config,
env var, or results is touched by any of the above.
