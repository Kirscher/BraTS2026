#!/usr/bin/env bash
# BraTS 2026 Task 3 (GoAT) — container entrypoint.
#
# Reads the read-only /input mount, writes one flat prediction per case to /output.
# All real work lives in brats2026.inference.entrypoint so it is unit-tested on a laptop;
# this script only pins the container's fixed paths and fails loudly.
#
# -e  : any failing step aborts the run rather than shipping a partial /output
# -u  : an unset variable is a bug, not an empty string
# -o pipefail : a failure anywhere in a pipeline propagates
set -euo pipefail

: "${BRATS_FOLDS:=0}"           # space-separated folds to ensemble, e.g. "0 1 2 3 4"
: "${BRATS_DATASET:=501}"
: "${BRATS_TRAINER:=nnUNetTrainerGoAT}"
: "${BRATS_PLANS:=nnUNetResEncUNetLPlans}"
: "${BRATS_CONFIGURATION:=3d_fullres}"
: "${BRATS_DISABLE_TTA:=0}"     # set to 1 to drop mirroring TTA and buy back time

TTA_FLAG=()
if [[ "${BRATS_DISABLE_TTA}" == "1" ]]; then
  TTA_FLAG=(--disable-tta)
fi

echo "[run_inference] BraTS 2026 Task 3 (GoAT) container starting"
echo "[run_inference] dataset=${BRATS_DATASET} folds=${BRATS_FOLDS} tta_disabled=${BRATS_DISABLE_TTA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "[run_inference] no GPU visible"

START=$(date +%s)

# shellcheck disable=SC2086  # BRATS_FOLDS is intentionally word-split into separate argv items
python -m brats2026.cli predict-container \
  --input /input \
  --output /output \
  --staging /tmp/brats2026_staging \
  --config /opt/app/infer.yaml \
  --dataset "${BRATS_DATASET}" \
  --folds ${BRATS_FOLDS} \
  --trainer "${BRATS_TRAINER}" \
  --plans "${BRATS_PLANS}" \
  --configuration "${BRATS_CONFIGURATION}" \
  "${TTA_FLAG[@]}"

ELAPSED=$(( $(date +%s) - START ))
echo "[run_inference] finished in ${ELAPSED}s ($(( ELAPSED / 60 )) min) of the 8h (28800s) budget"

# Guard the flat-output rule: a sub-folder in /output invalidates the submission.
if find /output -mindepth 1 -type d | grep -q .; then
  echo "[run_inference] FATAL: /output contains sub-directories; the submission requires a flat layout" >&2
  exit 1
fi

echo "[run_inference] wrote $(find /output -maxdepth 1 -name '*.nii.gz' | wc -l) prediction(s)"
