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


def cmd_install_trainer_v2(_: argparse.Namespace) -> None:
    from .nnunet.trainer_v2 import GOAT_V2_TRAIN_CONFIG_ENV, install_goat_trainer_v2

    shim = install_goat_trainer_v2()
    print(f"Installed nnUNetTrainerGoATv2 discovery shim at {shim}")
    print(f"Now set ${GOAT_V2_TRAIN_CONFIG_ENV}=<path to a filled configs/train_v2.yaml> before nnUNetv2_train.")
    print("This does NOT touch the v1 shim — a v1 run already in flight keeps the trainer it started with.")


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
    # Validate against the kind the file actually declares. Hardcoding "train" here would stamp a
    # train_v2 config correctly and THEN reject it against the v1 hook tuple.
    stamped = load_config(args.config)
    kind = args.kind or stamped.get("kind") or "train"
    assert_valid_config(stamped, kind)  # fail loud if anything is still unset
    env_var = "BRATS_GOAT_V2_TRAIN_CONFIG" if kind == "train_v2" else "BRATS_GOAT_TRAIN_CONFIG"
    print(f"stamp-config: bound {args.config} (kind={kind}) -> fingerprint {fingerprint[:16]}… @ {git_sha[:10]}")
    print(f"Config is now valid; set ${env_var} to it before nnUNetv2_train.")


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


def cmd_synthesize(args: argparse.Namespace) -> None:
    """Stage-3 generation: mint ``--n`` synthetic cases for ``--cohort`` into ``--out``.

    Everything that can be decided without touching a GPU happens here — the SPECIALIST hooks are
    loaded and validated, the case ids are minted and checked against the GoAT-only rule — and only
    then is each case handed to the (torch) generator. With ``--dry-run`` the ids are written out
    and no generation is attempted, which is how the plan is inspected on a laptop.
    """
    from .domains import COHORTS
    from .synthesis.generate import assert_synthetic_compliant, synthetic_case_id
    from .synthesis.masks import assert_configured as assert_mask_configured
    from .synthesis.masks import load_mask_config

    mask_config = load_mask_config(args.config)
    assert_mask_configured(mask_config)  # refuses while the ai-specialist hooks are still null

    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]
    per_cohort, remainder = divmod(args.n, len(cohorts))
    case_ids: list[str] = []
    for i, cohort in enumerate(cohorts):
        count = per_cohort + (1 if i < remainder else 0)
        case_ids.extend(synthetic_case_id(cohort, len(case_ids) + j) for j in range(count))
    assert_synthetic_compliant(case_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        planned = args.out / "planned_case_ids.json"
        planned.write_text(json.dumps(sorted(case_ids), indent=2) + "\n", encoding="utf-8")
        print(f"synthesize --dry-run: planned {len(case_ids)} case(s) across {len(cohorts)} cohort(s) -> {planned}")
        return

    from .synthesis.generate import generate_synthetic_case  # noqa: F401 - torch-only scaffold

    raise SystemExit(
        f"synthesize: {len(case_ids)} case(s) planned, but generate_synthetic_case is still a "
        "scaffold (needs the trained Stage-1 AE + Stage-2 diffusion checkpoints in a torch env). "
        "Re-run with --dry-run to emit the planned case ids."
    )


def cmd_build_synth_dataset(args: argparse.Namespace) -> None:
    """Assemble the augmented Dataset701 from real manifest records + generated cases."""
    from .preprocess import load_manifest_records
    from .synthesis.dataset import apply_synth_dataset, discover_synthetic_cases, plan_synth_dataset

    real_records = load_manifest_records(args.manifest) if args.manifest else []
    _records, incomplete = discover_synthetic_cases(args.synthetic_root)
    plan, synthetic_ids = plan_synth_dataset(
        real_records, args.synthetic_root, args.raw_root, dataset_id=args.dataset_id
    )
    sidecar = apply_synth_dataset(plan, synthetic_ids, link=not args.copy, overwrite=args.overwrite)

    n_real = plan.num_training - len(synthetic_ids)
    print(f"build-synth-dataset: {plan.dataset_dir}")
    print(f"  {plan.num_training} training case(s) = {n_real} real + {len(synthetic_ids)} synthetic")
    if incomplete:
        print(f"  {len(incomplete)} generated case(s) skipped as incomplete (missing modality/seg)")
    if plan.skipped:
        print(f"  {len(plan.skipped)} case(s) skipped by the converter")
    print(f"  synthetic provenance -> {sidecar}")


def cmd_predict_container(args: argparse.Namespace) -> None:
    """Run the submission container pipeline: /input -> predictions -> flat /output."""
    from .inference.entrypoint import run_container

    written = run_container(
        input_dir=args.input,
        output_dir=args.output,
        staging_dir=args.staging,
        infer_config=args.config,
        model_dir=args.dataset,
        folds=tuple(args.folds),
        trainer=args.trainer,
        plans=args.plans,
        configuration=args.configuration,
        disable_tta=args.disable_tta,
    )
    print(f"predict-container: wrote {written} prediction(s) to {args.output}")


def cmd_sweep_thresholds(args: argparse.Namespace) -> None:
    """Search the (probability threshold, min component size) grid for a better operating point.

    Reads nnU-Net's saved softmax (`nnUNetv2_predict --save_probabilities` writes one .npz per
    case) plus the ground-truth labels, and reports the operating point maximising each
    objective. Nothing is applied: the winning numbers go into configs/infer.yaml by hand, so a
    threshold change is a reviewed config edit and not a silent behaviour change.
    """
    import numpy as np

    from .evaluation.metrics import REGIONS, region_mask
    from .evaluation.threshold_sweep import (
        best_operating_point,
        summarise,
        sweep_operating_points,
    )

    region = REGIONS[args.region]
    channel = args.channel if args.channel is not None else list(REGIONS).index(args.region)

    cases = []
    for prob_path in sorted(Path(args.probs).glob("*.npz")):
        gt_path = Path(args.gt) / (prob_path.stem + ".nii.gz")
        if not gt_path.exists():
            print(f"  skip {prob_path.name}: no ground truth at {gt_path.name}")
            continue
        with np.load(prob_path) as handle:
            key = "probabilities" if "probabilities" in handle else list(handle.keys())[0]
            prob = np.asarray(handle[key])[channel]
        import nibabel as nib

        gt = region_mask(np.asarray(nib.load(str(gt_path)).dataobj), region)
        cases.append((prob, gt))

    if not cases:
        raise SystemExit(f"no (probability, ground-truth) pairs found under {args.probs}")

    thresholds = [float(t) for t in args.thresholds.split(",")]
    min_voxels = [int(v) for v in args.min_voxels.split(",")]
    results = sweep_operating_points(cases, thresholds, min_voxels)

    baseline = None
    if args.baseline_threshold is not None:
        baseline = next(
            (r for r in results
             if r.threshold == args.baseline_threshold and r.min_voxels == args.baseline_min_voxels),
            None,
        )
        if baseline is None:
            print("  note: the baseline point is not on the grid; reporting without deltas")

    report = summarise(results, baseline=baseline)
    report["region"] = args.region
    report["n_cases"] = len(cases)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"sweep-thresholds: {len(cases)} case(s), {len(results)} operating point(s) -> {args.out}")
    for name in ("f1", "f2", "recall", "precision"):
        best = best_operating_point(results, name)
        print(
            f"  best {name:<9} thr={best.threshold:<5} min_voxels={best.min_voxels:<5} "
            f"recall={best.recall:.3f} precision={best.precision:.3f} dice={best.mean_dice:.4f}"
        )
    print("Apply by editing configs/infer.yaml — this command never writes a threshold itself.")


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

    install_v2_parser = subparsers.add_parser(
        "install-trainer-v2",
        help="Register nnUNetTrainerGoATv2 on nnU-Net's search path (does not touch the v1 shim).",
    )
    install_v2_parser.set_defaults(func=cmd_install_trainer_v2)

    stamp_parser = subparsers.add_parser(
        "stamp-config",
        help="Bind a train config to the preprocessed data + commit (git_sha/date/fingerprint).",
    )
    stamp_parser.add_argument("--config", type=Path, required=True, help="configs/train.yaml to stamp in place.")
    stamp_parser.add_argument("--kind", choices=("train", "train_v2", "infer"),
                              help="Config kind to validate against (default: the file's own `kind:` field).")
    stamp_parser.add_argument("--raw", type=Path, required=True,
                              help="nnU-Net raw dataset dir, e.g. $nnUNet_raw/Dataset501_BraTSGoAT.")
    stamp_parser.add_argument("--git-sha", help="Override commit SHA (default: git rev-parse HEAD).")
    stamp_parser.add_argument("--no-headers", action="store_true",
                              help="Fingerprint from file sizes only (skip NIfTI header reads).")
    stamp_parser.set_defaults(func=cmd_stamp_config)

    synth_parser = subparsers.add_parser(
        "synthesize",
        help="Stage-3: generate synthetic tumour cases for a cohort (DiffTumor track).",
    )
    synth_parser.add_argument("--config", type=Path, required=True,
                              help="configs/synthesis/synthesis.yaml (mask geometry + pool hooks).")
    synth_parser.add_argument("--cohort", default="all",
                              help="GoAT cohort to generate for, or 'all' to spread across cohorts.")
    synth_parser.add_argument("--n", type=int, required=True, help="Number of cases to generate.")
    synth_parser.add_argument("--out", type=Path, required=True, help="Output root for generated cases.")
    synth_parser.add_argument("--seed", type=int, help="Override the config RNG seed.")
    synth_parser.add_argument("--dry-run", action="store_true",
                              help="Plan and write the case ids without generating volumes.")
    synth_parser.set_defaults(func=cmd_synthesize)

    build_parser = subparsers.add_parser(
        "build-synth-dataset",
        help="Assemble the augmented Dataset701 (real + synthetic) with a synthetic-provenance sidecar.",
    )
    build_parser.add_argument("--synthetic-root", type=Path, required=True,
                              help="Root of generated cases (written by `brats2026 synthesize`).")
    build_parser.add_argument("--raw-root", type=Path, required=True,
                              help="nnU-Net raw root, e.g. work/nnUNet_raw.")
    build_parser.add_argument("--manifest", type=Path,
                              help="Discovery manifest of the REAL labelled cases to merge in.")
    build_parser.add_argument("-d", "--dataset-id", type=int, default=None,
                              help="Augmented dataset id to build (default: base 501 + 200 = 701).")
    build_parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    build_parser.add_argument("--overwrite", action="store_true", help="Replace existing links.")
    build_parser.set_defaults(func=cmd_build_synth_dataset)

    container_parser = subparsers.add_parser(
        "predict-container",
        help="Submission entrypoint: read /input case folders, write flat geometry-checked /output.",
    )
    container_parser.add_argument("--input", type=Path, default=Path("/input"),
                                  help="Read-only root of test case folders (challenge mount).")
    container_parser.add_argument("--output", type=Path, default=Path("/output"),
                                  help="Flat output dir; one <case_id>.nii.gz per case.")
    container_parser.add_argument("--staging", type=Path, default=Path("/tmp/brats2026_staging"),
                                  help="Writable scratch for nnU-Net-named inputs (/input is read-only).")
    container_parser.add_argument("--config", type=Path,
                                  help="configs/infer.yaml; post-processing is SKIPPED if absent/null.")
    container_parser.add_argument("--dataset", default="501", help="nnU-Net dataset id or name.")
    container_parser.add_argument("--folds", type=int, nargs="+", default=[0],
                                  help="Folds to ensemble at prediction time.")
    container_parser.add_argument("--trainer", default="nnUNetTrainerGoAT")
    container_parser.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    container_parser.add_argument("--configuration", default="3d_fullres")
    container_parser.add_argument("--disable-tta", action="store_true",
                                  help="Turn off mirroring TTA to buy back time against the 8h budget.")
    container_parser.set_defaults(func=cmd_predict_container)

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Score predicted vs ground-truth segmentations into the GoAT per-cohort report.",
    )
    eval_parser.add_argument("--pred", type=Path, required=True, help="Dir of predicted <case>.nii.gz.")
    eval_parser.add_argument("--gt", type=Path, required=True, help="Dir of ground-truth <case>.nii.gz.")
    eval_parser.add_argument("--output", type=Path, required=True, help="Report JSON to write.")
    eval_parser.set_defaults(func=cmd_evaluate)

    sweep_parser = subparsers.add_parser(
        "sweep-thresholds",
        help="Search probability/min-component operating points on saved softmax (no retraining).",
    )
    sweep_parser.add_argument("--probs", type=Path, required=True,
                              help="Dir of nnU-Net .npz softmax (nnUNetv2_predict --save_probabilities).")
    sweep_parser.add_argument("--gt", type=Path, required=True, help="Dir of <case>.nii.gz labels.")
    sweep_parser.add_argument("--out", type=Path, required=True, help="Report JSON to write.")
    sweep_parser.add_argument("--region", choices=("ET", "TC", "WT"), default="ET",
                              help="Region to sweep (default ET — the recall bottleneck).")
    sweep_parser.add_argument("--channel", type=int,
                              help="Softmax channel index (default: the region's position in REGIONS).")
    sweep_parser.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6",
                              help="Comma-separated probability thresholds to try.")
    sweep_parser.add_argument("--min-voxels", dest="min_voxels", default="1,5,10,25,50",
                              help="Comma-separated minimum component sizes to try (1 = filter off).")
    sweep_parser.add_argument("--baseline-threshold", type=float,
                              help="Currently-shipped threshold, to report deltas against.")
    sweep_parser.add_argument("--baseline-min-voxels", type=int, default=1,
                              help="Currently-shipped min component size (pairs with --baseline-threshold).")
    sweep_parser.set_defaults(func=cmd_sweep_thresholds)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
