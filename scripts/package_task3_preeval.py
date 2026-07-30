#!/usr/bin/env python3
"""Validate and package flat Task 3 predictions for Synapse preliminary evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from brats2026.packaging.geometry import assert_same_geometry, read_geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    case_dirs = sorted(path for path in args.inputs.iterdir() if path.is_dir())
    expected = {path.name for path in case_dirs}
    predictions = {path.name.removesuffix(".nii.gz"): path for path in args.predictions.glob("*.nii.gz")}
    missing = sorted(expected - predictions.keys())
    extra = sorted(predictions.keys() - expected)
    if missing or extra:
        raise SystemExit(f"prediction mismatch: missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})")

    records = []
    for case_dir in case_dirs:
        case = case_dir.name
        reference = case_dir / f"{case}-t1n.nii.gz"
        prediction = predictions[case]
        assert_same_geometry(*read_geometry(reference), *read_geometry(prediction))
        image = nib.load(str(prediction))
        labels = sorted(int(value) for value in np.unique(np.asanyarray(image.dataobj)))
        if not set(labels) <= {0, 1, 2, 3}:
            raise SystemExit(f"{case}: invalid labels {labels}")
        records.append({"case": case, "file": prediction.name, "labels": labels})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as archive:
        for case in sorted(expected):
            prediction = predictions[case]
            archive.write(prediction, arcname=prediction.name)

    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "task": "BraTS 2026 Task 3 GoAT preliminary evaluation",
                "cases": len(records),
                "archive": args.output.name,
                "sha256": sha256(args.output),
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"submission ready: {args.output} ({len(records)} flat NIfTI files)")
    print(f"sha256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
