# BraTS 2026 Task 3 (GoAT) — offline submission container.
#
# Hard challenge rules this image must satisfy:
#   - Target GPU: NVIDIA A10G, 24 GB VRAM (CUDA pinned below).
#   - 32 GiB RAM, 200 GB disk, 8h TOTAL inference budget (see packaging/budget.py).
#   - NO network at runtime: all deps + weights are baked into the image.
#   - /input is read-only, /output is written FLAT (one seg per case, no sub-folders).
#   - NIfTI geometry preserved EXACTLY (enforced by inference/entrypoint.py, atol=0).
#
# BEFORE BUILDING, stage the trained nnU-Net results tree at docker/weights/ (see COPY below).
# Those weights MUST be trained from scratch on GoAT data only — no external data, no
# prior-BraTS pretraining (GoAT disqualification rule).
#
#   docker build -f docker/Dockerfile -t brats2026-goat:latest .
#   docker run --rm --gpus all --network none \
#     -v "$PWD/testdata:/input:ro" -v "$PWD/out:/output" brats2026-goat:latest

# Pinned CUDA base for the A10G. Pin a concrete digest at submission time for reproducibility.
FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    nnUNet_raw=/opt/nnunet/raw \
    nnUNet_preprocessed=/opt/nnunet/preprocessed \
    nnUNet_results=/opt/nnunet/results

WORKDIR /opt/app

# --- Dependencies (installed at BUILD time; runtime is fully offline) ------------------------
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir nnunetv2 && \
    python -m pip install --no-cache-dir ".[mri,config]"

# --- Trained weights --------------------------------------------------------------------------
# Expects the nnU-Net results layout:
#   docker/weights/Dataset501_BraTSGoAT/<trainer>__<plans>__3d_fullres/
#     ├── plans.json
#     ├── dataset.json
#     └── fold_0/checkpoint_final.pth        (+ fold_1..4 if ensembling)
RUN mkdir -p /opt/nnunet/raw /opt/nnunet/preprocessed /opt/nnunet/results
COPY docker/weights/ /opt/nnunet/results/

# --- Inference config -------------------------------------------------------------------------
# Post-processing thresholds. Any null value SKIPS that step — the entrypoint never invents a
# threshold at submission time.
COPY configs/infer.yaml /opt/app/infer.yaml

# --- Register the custom trainers so nnU-Net resolves them at inference -----------------------
# BOTH shims are installed on purpose. nnU-Net resolves a trainer by the NAME baked into the
# results directory (<trainer>__<plans>__<configuration>), so a container shipping a v2 checkpoint
# fails at load time if only the v1 shim is present — and it fails inside the 8h scored run, with
# no network to fix it. Installing both is free and removes that whole failure mode.
RUN python -c "from brats2026.nnunet.trainer import install_goat_trainer; print(install_goat_trainer())" && \
    python -c "from brats2026.nnunet.trainer_v2 import install_goat_trainer_v2; print(install_goat_trainer_v2())"

# --- Entrypoint --------------------------------------------------------------------------------
COPY docker/run_inference.sh /opt/app/run_inference.sh
RUN chmod +x /opt/app/run_inference.sh
ENTRYPOINT ["/opt/app/run_inference.sh"]
