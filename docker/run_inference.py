#!/usr/bin/env python3
"""Offline BraTS GoAT container entrypoint: /input -> flat /output."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from brats2026.packaging.geometry import assert_same_geometry, read_geometry


MODALITIES = {"t1n": "0000", "t1c": "0001", "t2f": "0002", "t2w": "0003"}


def discover_cases(input_dir: Path) -> dict[str, dict[str, Path]]:
    cases: dict[str, dict[str, Path]] = {}
    for path in sorted(input_dir.rglob("*.nii.gz")):
        name = path.name[: -len(".nii.gz")]
        modality = None
        case = None
        for candidate in MODALITIES:
            suffix = f"-{candidate}"
            if name.endswith(suffix):
                case, modality = name[: -len(suffix)], candidate
                break
        if modality is None and len(name) > 5 and name[-5] == "_":
            channel = name[-4:]
            reverse = {value: key for key, value in MODALITIES.items()}
            if channel in reverse:
                case, modality = name[:-5], reverse[channel]
        if case is None or modality is None:
            continue
        if modality in cases.setdefault(case, {}):
            raise RuntimeError(f"duplicate {modality} image for {case}")
        cases[case][modality] = path

    if not cases:
        raise RuntimeError(f"no BraTS cases found below {input_dir}")
    for case, images in cases.items():
        missing = sorted(set(MODALITIES) - set(images))
        if missing:
            raise RuntimeError(f"{case}: missing modalities {missing}")
    return cases


def main() -> None:
    input_dir = Path(os.environ.get("BRATS_INPUT", "/input"))
    output_dir = Path(os.environ.get("BRATS_OUTPUT", "/output"))
    staging = Path("/tmp/brats_nnunet_input")
    prediction = Path("/tmp/brats_nnunet_output")
    if not input_dir.is_dir():
        raise RuntimeError(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(prediction, ignore_errors=True)
    staging.mkdir(parents=True)
    prediction.mkdir(parents=True)

    cases = discover_cases(input_dir)
    for case, images in cases.items():
        for modality, channel in MODALITIES.items():
            (staging / f"{case}_{channel}.nii.gz").symlink_to(images[modality].resolve())

    folds = os.environ.get("BRATS_NNUNET_FOLDS", "0 1 2 3 4").split()
    if not folds or any(not fold.isdigit() for fold in folds):
        raise RuntimeError(f"invalid BRATS_NNUNET_FOLDS: {folds}")
    command = [
        "nnUNetv2_predict",
        "-i", str(staging),
        "-o", str(prediction),
        "-d", os.environ.get("BRATS_NNUNET_DATASET", "501"),
        "-c", os.environ.get("BRATS_NNUNET_CONFIG", "3d_fullres"),
        "-tr", os.environ.get("BRATS_NNUNET_TRAINER", "nnUNetTrainer"),
        "-p", os.environ.get("BRATS_NNUNET_PLANS", "nnUNetPlans"),
        "-f", *folds,
        "-npp", os.environ.get("NNUNET_NPP", "3"),
        "-nps", os.environ.get("NNUNET_NPS", "3"),
        "--disable_progress_bar",
    ]
    if os.environ.get("BRATS_ENABLE_TTA", "0") != "1":
        command.append("--disable_tta")
    print(f"Predicting {len(cases)} case(s); TTA={'on' if '--disable_tta' not in command else 'off'}")
    subprocess.run(command, check=True)

    for case, images in cases.items():
        generated = prediction / f"{case}.nii.gz"
        if not generated.is_file():
            raise RuntimeError(f"missing prediction for {case}: {generated}")
        reference_geometry = read_geometry(images["t1n"])
        output_geometry = read_geometry(generated)
        assert_same_geometry(*reference_geometry, *output_geometry)
        shutil.copy2(generated, output_dir / generated.name)

    outputs = list(output_dir.glob("*.nii.gz"))
    if len(outputs) != len(cases):
        raise RuntimeError(f"expected {len(cases)} flat outputs, found {len(outputs)}")
    print(f"Wrote {len(outputs)} geometry-checked segmentations to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise
