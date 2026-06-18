# Repository architecture — BraTS 2026 Task 3 (GoAT)

This layout maps 1:1 onto the phases of `PIPELINE_nnUNetv2_Task3.md`. Every training /
inference value stays a `# SPECIALIST:` hook — modules here build *structure*, not numbers.

```
BraTS2026/
├── src/brats2026/
│   ├── tasks.py            # task specs (label maps, data roots)            [Phase 0]
│   ├── discover.py         # manifest builder (structure/headers only)      [Phase 0/1]
│   ├── preprocess.py       # legacy npz normalize/crop helper
│   ├── domains.py          # cohort extraction from BraTS-<COHORT>- IDs     [Phase 0/1]
│   ├── compliance.py       # GoAT no-external-data / no-prior-weights gate  [Phase 0]
│   ├── provenance.py       # pixel-free ledger + dataset fingerprint (RoT)  [Phase 0/1]
│   ├── config.py           # provenance-bound config schema/validator (gate) [Phase 2]
│   ├── cli.py              # `brats2026` entry point
│   ├── nnunet/
│   │   ├── convert.py      # manifest -> Dataset501 raw + dataset.json      [Phase 1]
│   │   ├── splits.py       # domain-balanced 5-fold + leave-one-domain-out  [Phase 1/2]
│   │   ├── plan.py         # ResEnc-L plan_and_preprocess command builder   [Phase 2]
│   │   └── trainer.py      # nnUNetTrainerGoAT scaffold (SPECIALIST hooks)  [Phase 2/3]
│   ├── evaluation/
│   │   ├── metrics.py      # Dice + HD95 on ET/TC/WT, legacy + lesion-wise  [Phase 3]
│   │   ├── report.py       # per-cohort + worst-cohort + LODO tables (inf-aware) [Phase 3]
│   │   └── protocol.py     # sealed nested inner-dev/outer-LODO protocol    [Phase 3]
│   ├── ssl/
│   │   └── pseudolabel.py  # self-training / pseudo-label gating            [Phase 4]
│   ├── inference/
│   │   ├── postprocess.py  # TTA merge + small-component filter + ET-suppression [Phase 5]
│   │   └── predict.py      # nnUNetv2_predict + ensemble command builders   [Phase 5]
│   └── packaging/
│       ├── geometry.py     # exact NIfTI geometry preservation guard        [Phase 6]
│       └── budget.py       # fixed 8h inference time-budget guard           [Phase 6]
├── tests/                  # pytest mirror of the package (numpy/stdlib only)
├── configs/                # specialist config templates (hyperparameters)  [Phase 2/4/5]
│   ├── train.yaml          # UNFILLABLE stub: empty header + all hooks null  [Phase 2]
│   └── infer.yaml          # UNFILLABLE stub: empty header + all hooks null  [Phase 2]
├── docker/                 # offline submission container                   [Phase 6]
├── paper/                  # LNCS short paper                               [Phase 7]
├── scripts/                # thin CLI wrappers / launch helpers
└── work/                   # ALL derived artifacts (gitignored, NAS is read-only)
    ├── manifests/  nnUNet_raw/  nnUNet_preprocessed/  nnUNet_results/
    ├── predictions/  reports/
```

## Design rules

- **NAS is read-only.** Everything derived is written under `work/` (gitignored).
- **No invented numbers.** Hyperparameters, thresholds, TTA configs, ensemble weights live
  behind `# SPECIALIST:` markers with the nnU-Net default as a placeholder only.
- **Testable without heavy deps.** Pure logic (domains, splits, metrics, post-processing)
  is numpy/stdlib only so `pytest` runs without torch/nnU-Net installed. nnU-Net/predict
  wrappers are *command builders* validated by asserting the emitted argv.
- **Geometry is sacred.** Submission outputs must match the input NIfTI affine/shape/header
  exactly (`packaging/geometry.py`).

## Phase 2 module reference

- **`config.py`** (stdlib core; YAML behind a guarded `import yaml`) — the **provenance-bound
  config gate**, the file-level analogue of `trainer.assert_configured`. The ai-specialist may
  set hyperparameters *only* through a config that (1) carries a complete provenance header
  (`PROVENANCE_FIELDS = git_sha, dataset_fingerprint, seed, parent_run_id, date,
  config_schema_version`) and (2) fills every `# SPECIALIST:` hook. `TRAIN_HOOK_KEYS` is
  imported **verbatim** from `trainer.SPECIALIST_HOOKS` (a test asserts equality) so code and
  config can never drift; `INFER_HOOK_KEYS = (tta_axes, ensemble_members, ensemble_weights,
  et_suppression_min_voxels, cc_min_voxels)`. `make_template(kind)` emits a config **born
  unusable** (empty header + every hook `None`); `validate_config(cfg, kind) ->
  ConfigValidation(ok, missing_header, unset_hooks)` is ok only when the header is complete and
  no hook is unset; `assert_valid_config` raises `ConfigError` naming exactly what is missing.
  `check_fingerprint_binding(cfg, expected)` enforces that the config's recorded
  `dataset_fingerprint` (from `provenance.py`) equals the shipping data's — i.e. it was not
  tuned against different data. `load_config`/`dump_config` round-trip YAML and raise a clear
  ImportError (`pip install pyyaml`) when PyYAML is absent.
- **`configs/train.yaml`, `configs/infer.yaml`** — committed **stubs** generated from
  `make_template(...)`: empty provenance header + all hooks `null`, with a top-of-file comment
  noting `assert_valid_config` refuses them on purpose. The ai-specialist fills them *after*
  the dataset fingerprint exists.

## Phase 5/6 module reference

- **`inference/postprocess.py`** (numpy/stdlib) — `merge_tta(prob_maps)` averages a stack of
  mirroring-TTA probability maps (rejects empty / shape-mismatched input);
  `connected_component_filter(mask, min_voxels)` drops sub-threshold blobs (reusing
  `evaluation.metrics.connected_components`); `et_suppression(label_array, min_voxels,
  replace_with=1)` relabels tiny ET (=3) components to NCR while preserving tumor core and
  leaving NCR/ED untouched. `min_voxels` is a required `# SPECIALIST:` value from
  `configs/infer.yaml` — no baked-in default. Size `== min_voxels` is kept (boundary inclusive).
- **`inference/predict.py`** (command builder) — `predict_command(...)` assembles the
  `nnUNetv2_predict` argv (`-i -o -d -f -tr -p -c`, `--disable_tta` when requested), mirroring
  the `plan.py` constants (`RESENC_L_PLANS`, `GOAT_TRAINER`); `predict_ensemble_commands(
  model_dirs, ...)` emits one argv per ensemble member into a per-member output subdir. Ensemble
  members and TTA toggling are parameters, never invented.
- **`packaging/geometry.py`** — the exact-geometry guard: `geometry_matches(...)` requires an
  exactly-equal shape and an affine equal within `atol` (default `0.0` = bit-exact via
  `np.array_equal`); `assert_same_geometry(...)` raises `GeometryError` naming shape-vs-affine
  mismatch; `read_geometry(path)` returns `(affine, shape)` behind a guarded nibabel import,
  header-only (never `get_fdata`).
- **`packaging/budget.py`** — the fixed 8h inference budget (`BUDGET_HOURS = 8`, a challenge
  rule, not a specialist knob): `estimate_total_seconds`, `fits_budget`, `headroom_seconds`, and
  `assert_within_budget(...)` which raises `BudgetError` with the projected time + overage or
  returns a `BudgetProjection`. Boundary convention: exactly-on-budget **fits** (`<=`).
- **`docker/Dockerfile`** — offline A10G submission container *template* (won't build without
  real weights): pinned CUDA base, build-time deps, marked `BAKE WEIGHTS HERE` and entrypoint
  plug-in points, runtime reads `/input` (read-only) → writes flat `/output`, no network.
