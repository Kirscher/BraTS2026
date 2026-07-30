# BraTS-GoAT 2026 Task 3

Minimal inference and evaluation code for a conventional nnU-Net submission to
BraTS-GoAT 2026 Task 3.

## Final method

- nnU-Net v2.6.2 `3d_fullres` PlainConvUNet;
- five fixed folds trained for 1,000 epochs;
- equal-weight probability ensembling across folds 0–4;
- nnU-Net spatial mirroring at inference;
- no tuned threshold or connected-component post-processing.

The official pooled validation server returned the following global scores on
the 450 cases shared by all evaluated configurations:

| Region | DSC | NSD | HD95 |
|---|---:|---:|---:|
| Enhancing tumor | 0.7805 | 0.5495 | 41.1847 |
| Tumor core | 0.8288 | 0.5089 | 19.3524 |
| Whole tumor | 0.8854 | 0.4908 | 13.5777 |

## Contents

- `docker/`: final offline inference container;
- `src/brats2026/packaging/`: NIfTI geometry validation;
- `scripts/`: model staging, input preparation, packaging, and analysis;
- `configs/final_inference.yaml`: submitted inference configuration;
- `tests/`: unit tests for geometry safeguards.

No challenge data, model weights, predictions, score exports, manuscript
sources, credentials, or cluster-specific launch scripts are distributed.

## Installation and tests

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Install the additional scientific packages only when reproducing the analysis:

```bash
python -m pip install -e '.[analysis]'
```

## Build the inference image

Place the five trained
`nnUNetTrainer__nnUNetPlans__3d_fullres` folds under
`work/nnUNet_results/Dataset501_BraTSGoAT/`, then run:

```bash
scripts/prepare_submission_model_base.sh

IMAGE=brats-goat-2026:fivefold-tta
docker build --platform linux/amd64 \
  -f docker/Dockerfile.base -t "$IMAGE" .
```

The image reads the four modalities from read-only `/input` and writes one
geometry-checked NIfTI label map per case to `/output`.

```bash
docker run --rm --gpus all --network none --shm-size=8g \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  "$IMAGE"
```

TTA is enabled by default. Set `BRATS_ENABLE_TTA=0` at runtime only for an
ablation.

## Analysis

The repository includes the scripts used for:

- paired comparisons of official validation scores;
- source-to-pooled-validation DSC analysis;
- out-of-fold failure characterization;
- cross-fitted ET probability-threshold and component-size sweeps.

All input paths are explicit command-line arguments or repository-relative
local paths. Analysis outputs are written below ignored `work/` directories.
