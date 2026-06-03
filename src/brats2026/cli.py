from __future__ import annotations

import argparse
import json
from pathlib import Path

from .discover import discover_mri_cases, discover_pathology, write_jsonl
from .preprocess import preprocess_mri_manifest
from .tasks import DEFAULT_DATA_ROOT, TASKS, get_task


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task", choices=sorted(TASKS), required=True)


def cmd_discover(args: argparse.Namespace) -> None:
    spec = get_task(args.task)
    if spec.kind == "pathology_classification":
        info = discover_pathology(args.data_root, spec, inspect_first_shard=args.inspect_first_shard)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
        print(f"Wrote pathology manifest to {args.output}")
        return

    records = discover_mri_cases(args.data_root, spec, split=args.split, max_records=args.limit)
    write_jsonl(records, args.output)
    missing_required = sum(bool(record.missing_required) for record in records)
    missing_target = sum("missing_target" in record.notes for record in records)
    print(
        f"Wrote {len(records)} records to {args.output} "
        f"({missing_required} with missing required inputs, {missing_target} with missing target)"
    )


def cmd_preprocess_mri(args: argparse.Namespace) -> None:
    modalities = args.modalities.split(",") if args.modalities else None
    preprocess_mri_manifest(
        manifest=args.manifest,
        output_dir=args.output_dir,
        modalities=modalities,
        limit=args.limit,
        crop_margin=args.crop_margin,
    )
    print(f"Wrote preprocessed cases to {args.output_dir}")


def cmd_tasks(_: argparse.Namespace) -> None:
    for key, spec in TASKS.items():
        print(f"{key}: {spec.name} [{spec.kind}]")
        print(f"  inputs: {', '.join(spec.input_suffixes)}")
        if spec.optional_suffixes:
            print(f"  optional: {', '.join(spec.optional_suffixes)}")
        print(f"  target: {spec.target_suffix}")
        print(f"  notes: {spec.notes}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brats2026")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tasks_parser = subparsers.add_parser("tasks", help="List configured challenge tasks.")
    tasks_parser.set_defaults(func=cmd_tasks)

    discover_parser = subparsers.add_parser("discover", help="Create a dataset manifest.")
    add_common_args(discover_parser)
    discover_parser.add_argument("--split", choices=("train", "validation"), default="train")
    discover_parser.add_argument("--output", type=Path, required=True)
    discover_parser.add_argument("--limit", type=int)
    discover_parser.add_argument("--inspect-first-shard", action="store_true")
    discover_parser.set_defaults(func=cmd_discover)

    prep_parser = subparsers.add_parser("preprocess-mri", help="Normalize/crop MRI cases from a manifest.")
    prep_parser.add_argument("--manifest", type=Path, required=True)
    prep_parser.add_argument("--output-dir", type=Path, required=True)
    prep_parser.add_argument("--modalities", help="Comma-separated modality suffixes, e.g. t1n,t1c,t2f,t2w.")
    prep_parser.add_argument("--limit", type=int)
    prep_parser.add_argument("--crop-margin", type=int, default=8)
    prep_parser.set_defaults(func=cmd_preprocess_mri)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
