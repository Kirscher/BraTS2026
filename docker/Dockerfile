# BraTS 2026 Task 3 (GoAT) — offline submission container TEMPLATE.
#
# This is a TEMPLATE: it will NOT build as-is because the trained weights are a placeholder
# (see "BAKE WEIGHTS HERE"). The inference-packager fills in real weights before submission.
#
# Hard challenge rules this image must satisfy:
#   - Target GPU: NVIDIA A10G, 24 GB VRAM (CUDA pinned below).
#   - 32 GiB RAM, 200 GB disk, 8h TOTAL inference budget (see packaging/budget.py).
#   - NO network at runtime: all deps + weights are baked into the image.
#   - /input is read-only, /output is written FLAT (one seg per case).
#   - NIfTI geometry preserved EXACTLY (see packaging/geometry.py).

# Pinned CUDA base for the A10G. Pin a concrete digest/tag at submission time for reproducibility.
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Fail fast and keep the image lean.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # nnU-Net v2 roots live inside the image (no NAS, no network at runtime).
    nnUNet_raw=/opt/nnunet/raw \
    nnUNet_preprocessed=/opt/nnunet/preprocessed \
    nnUNet_results=/opt/nnunet/results

WORKDIR /opt/app

# --- Dependencies (offline at runtime: installed at BUILD time only) -------------------------
# Copy metadata first for layer caching, then install the package + its inference extras.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir .

# --- BAKE WEIGHTS HERE ----------------------------------------------------------------------
# Place the trained nnU-Net results tree (plans, checkpoints, dataset.json, folds) into
# $nnUNet_results. These weights MUST be trained from scratch on GoAT data only — no external
# data / no prior-BraTS pretraining (GoAT disqualification rule). Placeholder:
#   COPY work/nnUNet_results/ /opt/nnunet/results/
RUN mkdir -p /opt/nnunet/raw /opt/nnunet/preprocessed /opt/nnunet/results

# --- Inference entrypoint -------------------------------------------------------------------
# The entrypoint reads /input (read-only) and writes a FLAT /output. Wire the actual prediction
# (built by brats2026.inference.predict.predict_command / predict_ensemble_commands, then
# post-processed by brats2026.inference.postprocess, with the geometry guard from
# brats2026.packaging.geometry) inside run_inference.sh / a Python entrypoint here.
#   PLUG ENTRYPOINT IN HERE:
# COPY docker/run_inference.sh /opt/app/run_inference.sh
# RUN chmod +x /opt/app/run_inference.sh
# ENTRYPOINT ["/opt/app/run_inference.sh"]

# Placeholder entrypoint so the template is self-documenting; replace with the line above.
ENTRYPOINT ["python", "-c", "import sys; sys.exit('TEMPLATE: bake weights + wire the predict entrypoint before building')"]
