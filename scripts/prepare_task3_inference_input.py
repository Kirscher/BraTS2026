#!/usr/bin/env python3
"""Create an idempotent nnU-Net input folder for the unlabeled GoAT cases."""

from __future__ import annotations

import argparse
from pathlib import Path


MODALITIES = {"t1n": "0000", "t1c": "0001", "t2f": "0002", "t2w": "0003"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="folder containing one directory per case")
    parser.add_argument("output", type=Path, help="flat nnU-Net input directory")
    parser.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory not found: {source}")
    output.mkdir(parents=True, exist_ok=True)

    cases = [path for path in sorted(source.iterdir()) if path.is_dir()]
    if not cases:
        raise SystemExit(f"no case directories found in {source}")

    expected: set[Path] = set()
    for case_dir in cases:
        case = case_dir.name
        for modality, channel in MODALITIES.items():
            matches = list(case_dir.glob(f"{case}-{modality}.nii.gz"))
            if len(matches) != 1:
                raise SystemExit(
                    f"{case}: expected exactly one {modality} image, found {len(matches)}"
                )
            destination = output / f"{case}_{channel}.nii.gz"
            expected.add(destination)
            source_file = matches[0].resolve()
            if destination.is_symlink() and destination.resolve() == source_file:
                continue
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            if args.copy:
                import shutil

                shutil.copy2(source_file, destination)
            else:
                destination.symlink_to(source_file)

    stale = [path for path in output.glob("*.nii.gz") if path not in expected]
    if stale:
        raise SystemExit(
            f"refusing input folder with {len(stale)} stale NIfTI file(s); clean {output} first"
        )
    print(f"prepared {len(cases)} cases ({len(expected)} modalities) in {output}")


if __name__ == "__main__":
    main()
