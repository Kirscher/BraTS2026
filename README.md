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
- `configs/nnunet/`: exact `dataset.json` and `plans.json` used by the model;
- `results/`: aggregate validation, failure-analysis, and ET-sweep summaries;
- `tests/`: unit tests for geometry safeguards.

No challenge data, predictions, per-case score exports, manuscript sources,
credentials, or cluster-specific launch scripts are distributed.

## Checkpoints

The five epoch-1,000 checkpoints are distributed in the
[`camera-ready-v1` GitHub Release](https://github.com/Kirscher/BraTS2026/releases/tag/camera-ready-v1).
Download and extract the bundle from the repository root:

```bash
curl -fL -o brats-goat-2026-nnunet-camera-ready-v1.tar.gz \
  https://github.com/Kirscher/BraTS2026/releases/download/camera-ready-v1/brats-goat-2026-nnunet-camera-ready-v1.tar.gz
tar -xzf brats-goat-2026-nnunet-camera-ready-v1.tar.gz -C work/nnUNet_results
```

The release includes a SHA-256 manifest for all five checkpoints and the two
configuration files. The MIT licence covers the repository code; use of the
checkpoints remains subject to the applicable BraTS challenge/data terms.

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

After extracting the checkpoint release as above, run:

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

On the 451-case validation archive, five-fold mirrored inference took 1 h 55
min on one NVIDIA Quadro RTX 6000 (about 15.3 s/case, end-to-end wall time).
Peak allocated VRAM was not instrumented.

## Analysis

The repository includes the scripts used for:

- paired comparisons of official validation scores;
- source-to-pooled-validation DSC analysis;
- out-of-fold failure characterization;
- fold-stratified bootstrap intervals and the paper failure figure;
- cross-fitted ET probability-threshold and component-size sweeps.

All input paths are explicit command-line arguments or repository-relative
local paths. Analysis outputs are written below ignored `work/` directories.
The scripts that reproduce paired contrasts and failure analyses require
organizer-provided labels, OOF predictions, or official per-case score exports;
those inputs cannot be redistributed here. The release therefore supports code
inspection and rerunning with authorized inputs, while `results/` provides the
reported aggregate outputs.

The four targeted partial-correlation intervals can be regenerated from an
authorized `case_features.csv` as follows:

```bash
python scripts/render_failure_analysis.py \
  --features work/analysis/failure_atlas/case_features.csv \
  --output work/analysis/failure_atlas/failure_analysis.pdf \
  --stats-output work/analysis/failure_atlas/failure_analysis_stats.json
```

## Citation

If you use this code or the checkpoints, please cite the preprint:

```bibtex
@misc{kirscher2026assessing,
  title         = {Assessing nnU-Net Generalization across Brain Tumor Populations in BraTS-GoAT 2026},
  author        = {Kirscher, Tristan and Metzger, Vivian and Meyer, Philippe and Coubez, Xavier},
  year          = {2026},
  eprint        = {2609.15524},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.15524}
}
```
