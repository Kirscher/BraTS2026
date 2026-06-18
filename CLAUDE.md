# BraTS 2026 — Task 3 (GoAT) project guide

This repo is the codebase for **MICCAI BraTS 2026 Task 3 = BraTS-GoAT** (Generalizability
Across Tumors): one segmentation model that must generalize across 5 cohorts (adult glioma
`GLI`, sub-Saharan `SSA`, meningioma `MEN`, metastasis `MET`, pediatric `PED`) and across
scanners/demographics. We start from an **nnU-Net v2** baseline.

## Team model

The whole project runs on **Fable 5**. All subagents use `model: inherit`, so they run on
the session's default model — keep that set to Fable 5 (`/model`) and the entire team stays
on Fable 5.

- **Lead = the human "vibe coder"**, driving the main Claude (Fable 5) thread. The lead owns
  structure and orchestration and routes work to the specialist subagents below.
- **`ai-specialist`** (an AI agent since the 2026-06-16 council rebuild — formerly a human) owns
  **hyperparameters and inference-time config**. Other agents must NOT invent training numbers
  (LR, schedules, loss weights, TTA configs, thresholds): they leave `# SPECIALIST:` hooks and
  the `ai-specialist` resolves them as versioned YAML under `configs/`. The expensive
  **train-launch still needs human approval** of the config.

## Specialist subagents (`.claude/agents/`)

| Agent | Owns | Invoke when |
|-------|------|-------------|
| `python-dev` | The `brats2026` package code + pytest (substrate the others import) | Implement/refactor/test a module |
| `data-pipeline` | nnU-Net raw conversion, `dataset.json`, domain folds, **cohort-balance QC, provenance ledger + write-once outer LODO fold** | Preparing/refreshing data |
| `ai-specialist` | **All hyperparameters + inference config → `configs/*.yaml`** (blind to outer LODO; 8h budget as input) | Resolving `# SPECIALIST:` hooks / retuning |
| `nnunet-trainer` | nnU-Net v2 plans + `nnUNetTrainerGoAT`, runs `configs/train.yaml`, invents nothing | Wiring or running training |
| `validation-metrics` | Region Dice+HD95 (ET/TC/WT), per-cohort + worst-cohort, **nested inner-dev/outer-LODO protocol** (judges, never tunes) | Scoring or comparing models |
| `inference-packager` | Prediction entrypoint + Docker, builds `configs/infer.yaml`, **8h profiling** | Building/validating the submission |
| `compliance-guard` | GoAT rule, container limits, geometry, **ledger↔container audit** (read-only gate) | Before any submission or merge |
| `paper-writer` | The mandatory LNCS short paper, grounded in real code/results (late-activated) | Drafting/updating the paper |

Typical flow: `python-dev` builds the package → `data-pipeline` → `ai-specialist` sets config →
`nnunet-trainer` (human-gated launch) → `validation-metrics` → `inference-packager` →
`compliance-guard` (gate) → submit; `paper-writer` activates once real results exist.

**Routing shortcut:** a user prompt beginning **`BraTS:`** must be answered *through* these
agents (enforced by a `UserPromptSubmit` hook in `.claude/settings.json`), not from the main thread.

## Hard constraints (every agent must respect these)

- **★ GoAT no-external-data rule:** ONLY BraTS-GoAT-provided data may be used. Any external/
  private data OR pretrained weights derived from prior BraTS → **disqualification**. Train
  from scratch. SSL/semi-supervised is allowed **only on GoAT's own data**.
- **Labels** harmonized to `NCR=1, ED=2, ET=3` (background 0). Scored on **ET={3}, TC={1,3},
  WT={1,2,3}** via **Dice + HD95** (rank-aggregated). Sens/Spec/Precision advisory only.
- **Per case:** 4 modalities `t1n, t1c, t2f, t2w` (all required) → target `seg`. Cohort is
  encoded in the case-ID prefix `BraTS-<GLI|SSA|MEN|MET|PED>-...` → use for balanced sampling
  and leave-one-domain-out validation.
- **Container limits:** A10G 24GB, 32 GiB RAM, 200 GB disk, **8h total inference**, **no
  network**, `/input` read-only, `/output` flat. **Preserve NIfTI geometry exactly.**
- **Deadlines:** Docker queues open Jun 17; **early final Jul 2**; **final Jul 23**; mandatory
  8–10pp LNCS short paper.

## Data layout (verify on disk — there is a known discrepancy)

- Code default `tasks.py:DEFAULT_DATA_ROOT = /mnt/CPS-RADT/...`, but Task 3 data is stated at
  `/mnt/NAS2418_RADT/datasets/MICCAI/2026/BraTS2026/Task3`. `--data-root` must be the **parent
  of `Task3`** (the code appends `Task3`).
- `task3` train roots in code use **2024** GoAT folder names; the **2026** layout may differ.
  `brats2026 discover` returning 0 cases means the roots need patching.
- **Treat all NAS mounts as read-only.** Write every derived artifact under `work/`
  (manifests, `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`, predictions, reports).
- The NAS holds **authorized controlled research data** under a Synapse data-use agreement:
  inspect **structure/headers/metadata only** (`nib.load().shape/.affine/.header`, no
  `get_fdata` on pixels). This authorization is established — don't re-litigate it.

## Existing package surface

`src/brats2026/`: `cli.py` (`tasks` / `discover` / `preprocess-mri`), `discover.py`,
`preprocess.py`, `tasks.py` (task specs incl. `task3` label map `NCR=1, ED=2, ET=3`).
Install: `pip install -e .[mri,train,dev]`. Tests in `tests/`. There is **no training code yet** —
that is what this team is building.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
