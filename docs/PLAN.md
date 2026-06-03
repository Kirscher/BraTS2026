# BraTS 2026 Baseline Plan

Sources checked on 2026-06-03:

- Synapse project `syn74274097` via `synapseclient`, raw pages archived in `docs/synapse_wiki_raw/`.
- MICCAI registered challenge entry DOI `10.5281/zenodo.19714728`.

## Challenge Snapshot

BraTS 2026 contains five tasks:

1. Brain metastases segmentation, pre/post treatment MRI.
2. Pediatric brain tumor segmentation, pre/post treatment MRI.
3. BraTS-GoAT generalizable tumor segmentation across tumor entities.
4. Local MRI inpainting of healthy tissue in tumor regions.
5. BraTS-Path histology patch classification.

Key timeline:

- Training and validation data released May 1, 2026.
- Validation leaderboard live June 1, 2026.
- Docker queues open June 17, 2026.
- Early final package deadline July 2, 2026.
- Standard final package deadline July 23, 2026.
- MICCAI satellite events September 27 to October 1, 2026.

## Rules That Affect Engineering

- Short paper is mandatory for final ranking.
- Final submissions are Docker containers, no runtime network access.
- Hidden test input is mounted read-only at `/input`; outputs must be flat in `/output`.
- Tasks 1-3 must preserve source NIfTI spatial metadata exactly for file submissions.
- Task 4 outputs must end with `-t1n-inference.nii.gz`.
- Task 5 outputs exactly `/output/predictions.csv` with `SubjectID,Prediction`.
- External data/pretrained models are generally allowed only if public, but BraTS-GoAT is stricter: only data provided through the BraTS-GoAT sub-challenge may be used.

## Winning Strategy

The safest path is a shared data and container infrastructure plus task-specific model families:

- Tasks 1-3: MONAI/nnU-Net-style 3D segmentation ensemble with residual/SegResNet and transformer variants, deep supervision, heavy spatial/intensity augmentation, test-time mirroring, connected-component post-processing, and lesion-sensitive losses.
- Task 1: add small-lesion detection emphasis, high-resolution crops around candidate enhancing regions, and threshold sweeps optimized for lesion-wise F1/AUC.
- Task 2: train without skull stripping by default, matching validation/test distribution; use longitudinal cases as separate timepoints plus patient-grouped validation folds.
- Task 3: treat as domain generalization; use domain-balanced sampling, label-presence-aware loss, and strict data provenance checks.
- Task 4: start from the official inpainting baseline, then move to 3D U-Net/diffusion or masked autoencoder refiners, evaluated on SSIM/PSNR/MSE inside mask.
- Task 5: WebDataset pipeline with ConvNeXt/EfficientNet/ViT pathology backbones, strong color augmentation/stain normalization, class-balanced sampling, patient/slide-grouped validation, and calibrated ensembles.

## First Implemented Step

This repo now has a lightweight `brats2026` package:

- `brats2026 tasks` lists configured tasks.
- `brats2026 discover` writes manifests without requiring imaging dependencies.
- `brats2026 preprocess-mri` normalizes and foreground-crops MRI cases into compressed `.npz` files when `numpy+nibabel` are installed.

Recommended next steps:

1. Install MRI dependencies in the selected environment.
2. Generate full train/validation manifests for tasks 1-4.
3. Build grouped validation folds and dataset statistics from those manifests.
4. Add MONAI datamodules/losses for tasks 1-3.
5. Add WebDataset datamodule and slide-aware validation for task 5.
