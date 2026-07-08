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


def cmd_ssl_select(args: argparse.Namespace) -> None:
    from .ssl import load_ssl_config, select_from_stats_records

    config = load_ssl_config(args.config)
    records = json.loads(Path(args.stats_json).read_text(encoding="utf-8"))
    accepted = select_from_stats_records(records, config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(accepted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n = sum(len(v) for v in accepted.values())
    print(f"ssl-select: accepted {n} pseudo case(s) across {len(accepted)} cohort(s) -> {args.out}")


def cmd_install_trainer(_: argparse.Namespace) -> None:
    from .nnunet.trainer import GOAT_TRAIN_CONFIG_ENV, install_goat_trainer

    shim = install_goat_trainer()
    print(f"Installed nnUNetTrainerGoAT discovery shim at {shim}")
    print(f"Now set ${GOAT_TRAIN_CONFIG_ENV}=<path to a filled configs/train.yaml> before nnUNetv2_train.")


def cmd_stamp_config(args: argparse.Namespace) -> None:
    import subprocess
    from datetime import date as _date

    from .config import assert_valid_config, load_config, stamp_config_file
    from .nnunet.convert import CHANNEL_ORDER, DATASET_NAME
    from .provenance import dataset_fingerprint_from_raw

    extra = {
        "label_map": {"NCR": 1, "ED": 2, "ET": 3},
        "seed": (load_config(args.config).get("provenance") or {}).get("seed"),
        "dataset": DATASET_NAME,
        "channels": list(CHANNEL_ORDER),
    }
    fingerprint = dataset_fingerprint_from_raw(args.raw, extra, read_headers=not args.no_headers)

    git_sha = args.git_sha
    if git_sha is None:
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
            ).strip()
        except Exception:  # noqa: BLE001 - git absent / not a repo: caller can pass --git-sha
            raise SystemExit("could not read git HEAD; pass --git-sha <sha> explicitly")

    stamp_config_file(
        args.config,
        {"git_sha": git_sha, "date": _date.today().isoformat(), "dataset_fingerprint": fingerprint},
    )
    assert_valid_config(load_config(args.config), "train")  # fail loud if anything is still unset
    print(f"stamp-config: bound {args.config} -> fingerprint {fingerprint[:16]}… @ {git_sha[:10]}")
    print("Config is now valid; set $BRATS_GOAT_TRAIN_CONFIG to it before nnUNetv2_train.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .evaluation.score import evaluate_directory

    report = evaluate_directory(args.pred, args.gt, args.output)
    worst = report["worst_cohort"]["WT"]
    print(f"evaluate: scored {report['n_cases']} case(s) -> {args.output}")
    legacy = report["per_cohort"]["legacy"]["cohorts"]
    for cohort, body in legacy.items():
        if body["n_cases"]:
            wt = body["regions"]["WT"]["dice"]
            print(f"  {cohort}: n={body['n_cases']:<4} WT Dice(legacy)={wt:.4f}" if wt is not None else f"  {cohort}: n={body['n_cases']}")
    if worst["cohort"] is not None:
        print(f"  worst cohort (lesion WT Dice): {worst['cohort']} = {worst['dice']:.4f}")


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

    sel_parser = subparsers.add_parser(
        "ssl-select",
        help="Filter teacher pseudo-labels by confidence / cohort quota (self-training step).",
    )
    sel_parser.add_argument("--stats-json", type=Path, required=True,
                            help="Per-case CaseStats records (produced from the teacher softmax).")
    sel_parser.add_argument("--config", type=Path, required=True, help="configs/ssl.yaml")
    sel_parser.add_argument("--out", type=Path, required=True, help="Accepted cohort→case-ids JSON.")
    sel_parser.set_defaults(func=cmd_ssl_select)

    install_parser = subparsers.add_parser(
        "install-trainer",
        help="Register nnUNetTrainerGoAT on nnU-Net's trainer search path (run once on the HPC).",
    )
    install_parser.set_defaults(func=cmd_install_trainer)

    stamp_parser = subparsers.add_parser(
        "stamp-config",
        help="Bind a train config to the preprocessed data + commit (git_sha/date/fingerprint).",
    )
    stamp_parser.add_argument("--config", type=Path, required=True, help="configs/train.yaml to stamp in place.")
    stamp_parser.add_argument("--raw", type=Path, required=True,
                              help="nnU-Net raw dataset dir, e.g. $nnUNet_raw/Dataset501_BraTSGoAT.")
    stamp_parser.add_argument("--git-sha", help="Override commit SHA (default: git rev-parse HEAD).")
    stamp_parser.add_argument("--no-headers", action="store_true",
                              help="Fingerprint from file sizes only (skip NIfTI header reads).")
    stamp_parser.set_defaults(func=cmd_stamp_config)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Score predicted vs ground-truth segmentations into the GoAT per-cohort report.",
    )
    eval_parser.add_argument("--pred", type=Path, required=True, help="Dir of predicted <case>.nii.gz.")
    eval_parser.add_argument("--gt", type=Path, required=True, help="Dir of ground-truth <case>.nii.gz.")
    eval_parser.add_argument("--output", type=Path, required=True, help="Report JSON to write.")
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
