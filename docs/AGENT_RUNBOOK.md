# Agent runbook — how the BraTS-GoAT team works together

Six subagents live in `.claude/agents/`. **You (the lead) never run them directly** — you
type a normal request in the main Claude (Fable 5) thread and Claude delegates to the right
subagent based on its `description`. To force a specific one, name it: *"Use the
**nnunet-trainer** subagent to ..."*. Each subagent runs in its own context and reports back
to you; they do **not** talk to each other — **you are the bus** that passes one agent's output
to the next.

The phases below are the intended order. Copy a prompt, paste it into the thread, review the
agent's report, then move to the next. The human AI specialist fills in every `# SPECIALIST:`
hook between phases.

---

## Phase 0 — Sanity check (lead, no agent)

> Confirm the repo installs and tests pass: `pip install -e .[mri,train,dev]` then `pytest -q`.
> Show me `brats2026 tasks` output and whether the Task 3 data root resolves on disk.

## Phase 1 — Data → nnU-Net raw (`data-pipeline`)

> Use the **data-pipeline** subagent. Verify the BraTS-GoAT Task 3 data root (try
> `/mnt/NAS2418_RADT/datasets/MICCAI/2026/BraTS2026` as the parent of `Task3`). Run
> `brats2026 discover --task task3`; if it finds 0 cases, locate the real 2026 GoAT folders and
> patch the `task3` roots in `src/brats2026/tasks.py`. Then convert all cases into
> `work/nnUNet_raw/Dataset501_BraTSGoAT/` (channels 0=t1n,1=t1c,2=t2f,3=t2w), write
> `dataset.json` with labels NCR=1/ED=2/ET=3, and build both a domain-balanced 5-fold
> `splits_final.json` and a leave-one-domain-out split set, keyed off the `BraTS-<cohort>-`
> prefix. Report case counts per cohort and any missing-modality cases. NAS read-only; all
> output under `work/`.

**Gate before Phase 2:** the dataset id, per-cohort counts, and splits exist under `work/`.

## Phase 2 — Training scaffolding (`nnunet-trainer`)  →  specialist tunes

> Use the **nnunet-trainer** subagent. Plan and preprocess dataset 501 with the ResEnc-L
> preset for 3d_fullres (`nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity`).
> Add an `nnUNetTrainerGoAT` subclass with domain-balanced sampling and domain-randomization
> augmentation, region-based deep supervision for ET/TC/WT. Leave every hyperparameter (LR,
> schedule, epochs, batch/patch size, loss weights, aug magnitudes) as a labelled
> `# SPECIALIST:` knob with the nnU-Net default as placeholder — do not invent values. Give me
> the launch commands for the 5 folds and the leave-one-domain-out run, and flag anything that
> risks OOM on A10G 24GB.

**Hand to the human AI specialist:** they set every `# SPECIALIST:` value. Then:

> Use the **nnunet-trainer** subagent to launch fold 0 with the specialist's settings and
> monitor GPU memory and the first epochs; report the results path.

## Phase 3 — Evaluate (`validation-metrics`)

> Use the **validation-metrics** subagent. Build the eval harness: Dice + HD95 on
> ET={3}/TC={1,3}/WT={1,2,3}, implemented both legacy-overlap and lesion-wise. Score the
> trained fold(s), break results down per cohort (GLI/SSA/MEN/MET/PED) and for the
> leave-one-domain-out run, and write a comparison table to `work/reports/`. Sanity-check that
> predicted labels ⊆ {0,1,2,3} and geometry matches the reference before trusting scores.

**Loop:** feed the per-cohort table back to the specialist; they adjust knobs; re-run Phase 2/3
until the worst cohort is acceptable.

## Phase 4 — Submission container (`inference-packager`)

> Use the **inference-packager** subagent. Build the `/input → /output` predictor for the
> selected nnU-Net ensemble: read the 4 modalities per case, write segmentations flat to
> `/output`, and assert output NIfTI affine/shape/header match the input reference exactly. Add
> switchable mirroring-TTA, connected-component, and ET-suppression hooks with thresholds as
> `# SPECIALIST:` params. Write an offline Dockerfile (weights baked in, no network, fits A10G
> 24GB / 8h) and a local smoke test using a small `work/` input/output pair.

## Phase 5 — Compliance gate (`compliance-guard`) — required before any submission

> Use the **compliance-guard** subagent to review the container and training configs against
> the GoAT rules: no external data, no prior-BraTS pretrained weights, train-from-scratch,
> offline + flat-output + exact-geometry, correct label encoding, and nothing from the NAS
> committed. Return PASS/FAIL/UNSURE per item with file:line, and tell me where we stand
> against the Jun 17 / Jul 2 / Jul 23 deadlines. A single FAIL blocks submission.

Fix any FAIL with the owning agent, then re-run Phase 5. Only submit on a clean pass.

## Phase 6 — Paper (`paper-writer`, runs alongside from Phase 3 on)

> Use the **paper-writer** subagent to draft the LNCS short paper from the real method
> (data-pipeline conversion, nnUNetTrainerGoAT architecture, post-processing) and the latest
> `work/reports/` numbers — per-cohort and leave-one-domain-out Dice/HD95. Emphasize the
> cross-tumor generalization and train-from-scratch compliance story. Do not invent results;
> request any missing number. Keep it to 8–10 LNCS pages in `paper/`.

---

## Quick reference

| Phase | Agent | Produces | Blocks on |
|-------|-------|----------|-----------|
| 1 | data-pipeline | `work/nnUNet_raw` + splits | data root verified |
| 2 | nnunet-trainer | trainer + launch cmds | specialist sets knobs |
| 3 | validation-metrics | `work/reports/` tables | trained fold exists |
| 4 | inference-packager | offline Docker container | chosen ensemble |
| 5 | compliance-guard | PASS/FAIL gate | required pre-submit |
| 6 | paper-writer | LNCS paper | real results only |

Routing rule of thumb: data trouble → data-pipeline; training/architecture → nnunet-trainer;
"how good is it / per cohort" → validation-metrics; packaging/Docker/geometry →
inference-packager; "are we allowed / are we ready" → compliance-guard; writing → paper-writer.
