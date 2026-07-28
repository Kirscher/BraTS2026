# Submission runbook — BraTS 2026 Task 3 (GoAT)

**Deadline: 30 July 2026.** Three things must arrive, and the paper gates the container:
organizers only evaluate a Docker submission that is linked to a submitted short paper.

1. Containerized algorithm (Docker)
2. Short paper (Microsoft CMT, Springer LNCS, 8–10 pp excluding references)
3. Signed copyright form

Everything below is the critical path. Anything not on it — MAE, self-training, synthesis,
domain randomization, retraining — is out of scope for this deadline.

---

## Step 1 — Stage the trained weights (blocking, needs the HPC)

The container bakes weights at build time; there is no network at runtime. Copy the nnU-Net
results tree of **the model that produced the scored submission** into `docker/weights/`:

```
docker/weights/
└── Dataset501_BraTSGoAT/
    └── <trainer>__<plans>__3d_fullres/
        ├── plans.json
        ├── dataset.json
        └── fold_0/checkpoint_final.pth        # + fold_1..4 if ensembling
```

> **Recover the run's identity while you are there.** The paper must state the learning rate,
> schedule, epochs, batch/patch size and folds actually used, and the linked source code must
> reproduce them. `nnUNetTrainer` writes these into the fold's `training_log_*.txt` and
> `plans.json` — read them off rather than reconstructing from memory.

`docker/weights/` is gitignored territory in spirit; do not commit checkpoints.

## Step 2 — Decide post-processing (30 minutes, no GPU)

`configs/infer.yaml` currently has every threshold `null`, and **null means the step is
skipped** — the entrypoint never invents a threshold. Two honest options:

- **Ship raw.** Leave the nulls. Safe, matches the already-scored submission.
- **Enable suppression.** Set `et_suppression_min_voxels` and/or `cc_min_voxels`. This targets
  the measured weakness directly (`fp_et = 0.97`, `fn_et = 2.50`, `hd95_et = 40.35 mm`) at zero
  training cost — but an untuned threshold can *lose* Dice.

If the validation leaderboard is still open before the 30th, tune on it. If it is not, ship raw
rather than guessing: an unverified threshold change on the hidden test set is a coin flip.

## Step 3 — Build and validate locally (needs Linux + GPU)

```bash
docker build -f docker/Dockerfile -t brats2026-goat:latest .

# Validate on a handful of real cases laid out exactly like /input:
#   testdata/BraTS-GLI-00001-000/BraTS-GLI-00001-000-{t1n,t1c,t2f,t2w}.nii.gz
mkdir -p out
docker run --rm --gpus all --network none \
  -v "$PWD/testdata:/input:ro" \
  -v "$PWD/out:/output" \
  brats2026-goat:latest
```

The run must satisfy all of the following, and the entrypoint checks each one:

- [ ] `--network none` succeeds — no runtime downloads
- [ ] `/output` is **flat**, one `<case_id>.nii.gz` per input folder, no sub-directories
- [ ] geometry identical to input (checked at `atol=0`, aborts on mismatch)
- [ ] nothing written to `/input`
- [ ] per-case wall time × test-set size stays inside **8 h**

Time it and project. `brats2026.packaging.budget.assert_within_budget(per_case_s, n_cases)` does
the arithmetic and raises if it will not fit. If it will not, set `BRATS_DISABLE_TTA=1` (drops
mirroring TTA, roughly 8× faster inference) or reduce `BRATS_FOLDS` before touching anything else.

Environment knobs, all with safe defaults:

| Variable | Default | Purpose |
|---|---|---|
| `BRATS_FOLDS` | `0` | Space-separated folds to ensemble, e.g. `"0 1 2 3 4"` |
| `BRATS_DATASET` | `501` | nnU-Net dataset id |
| `BRATS_TRAINER` | `nnUNetTrainerGoAT` | Must match the trained model |
| `BRATS_PLANS` | `nnUNetResEncUNetLPlans` | Must match the trained model |
| `BRATS_DISABLE_TTA` | `0` | Set `1` to buy back time against the 8 h budget |

> **The trainer name must match the weights.** If the scored submission was trained with stock
> `nnUNetTrainer` rather than `nnUNetTrainerGoAT`, set `BRATS_TRAINER=nnUNetTrainer` or nnU-Net
> will not resolve the checkpoint.

## Step 4 — Write the paper (parallel with steps 1–3)

`paper/brats2026_goat.tex` is a complete LNCS skeleton with the real validation numbers already
filled in. Open a Springer LNCS project on Overleaf and paste it in.

Every `\TODO{}` is a fact only the team can supply; every `\CHECK{}` is a number to re-read off
the real run. **Resolve all of them before submitting** — the red and orange markers render
visibly, so nothing can slip through unnoticed.

The section that needs the most care is Methods: it must describe the model that was actually
submitted, and the GitHub link must point at code that reproduces it.

## Step 5 — Submit

1. Paper → Microsoft CMT, with the **Synapse team name** exactly as registered (this is how the
   paper and container get linked; a mismatch means the container is never evaluated).
2. Container → the Task 3 submission queue on Synapse.
3. Signed copyright form.

---

## Local smoke test without a GPU

The entrypoint's discovery, staging, naming and post-processing logic is unit-tested and runs on
any machine:

```bash
python -m pytest tests/inference/test_entrypoint.py -q
python -m brats2026.cli predict-container --help
```

This verifies the container's decision logic without CUDA, nnU-Net or real data. It does not
verify that the model loads — only step 3 does that.
