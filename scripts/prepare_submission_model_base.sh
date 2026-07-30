#!/bin/bash
# Stage only the files needed by docker/Dockerfile.base.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=nnUNetTrainer__nnUNetPlans__3d_fullres
SOURCE=$PROJECT_DIR/work/nnUNet_results/Dataset501_BraTSGoAT/$MODEL
DEST=$PROJECT_DIR/work/submission_model_base/nnUNet_results/Dataset501_BraTSGoAT/$MODEL

for metadata in dataset.json plans.json; do
    test -f "$SOURCE/$metadata" || { echo "missing $SOURCE/$metadata" >&2; exit 1; }
done
for fold in 0 1 2 3 4; do
    test -f "$SOURCE/fold_$fold/checkpoint_final.pth" || {
        echo "fold $fold is incomplete" >&2
        exit 1
    }
done

rm -rf "$DEST"
mkdir -p "$DEST"
cp "$SOURCE/dataset.json" "$SOURCE/plans.json" "$DEST/"
for fold in 0 1 2 3 4; do
    mkdir -p "$DEST/fold_$fold"
    cp "$SOURCE/fold_$fold/checkpoint_final.pth" "$DEST/fold_$fold/"
done

echo "staged stock nnU-Net model in $DEST"
du -sh "$PROJECT_DIR/work/submission_model_base"
