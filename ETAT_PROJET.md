# BraTS 2026 — État du projet

> Document d'avancement. Pour l'installation et la prise en main, voir `README.md` ; pour la stratégie, `docs/PLAN.md` et `docs/ARCHITECTURE.md`.
>
> **Dernière mise à jour : 2026-07-08** · Branche : `feat/brats2026-package-build`
>
> **Lancer un entraînement sur le HPC : voir `docs/RUN_HPC.md`** (séquence exacte « press go » côté mentor).

## Objectif

Participation à la **MICCAI BraTS 2026**, **Task 3 — BraTS-GoAT** (segmentation généralisable de tumeurs cérébrales à travers plusieurs entités tumorales). Approche : pipeline **nnU-Net v2** packagé, avec préparation des données, entraînement, inférence, évaluation et empaquetage Docker conformes aux règles du challenge.

Contraintes clés du challenge (cf. `docs/PLAN.md`) :
- Soumission finale = **conteneur Docker sans accès réseau** à l'exécution.
- **Short paper obligatoire** pour le classement final.
- Jalons : leaderboard validation 1er juin 2026 ; files Docker ouvertes 17 juin ; deadline package anticipée 2 juillet, standard 23 juillet 2026.

## État global

| Indicateur | Valeur |
|---|---|
| Phases construites | **0 à 6** (préparation → packaging) |
| Tests | **214 tests**, verts sur Linux (212 passés ; 2 échecs Windows-only : symlink `WinError 1314` + suffixe de chemin `/`) |
| CI | GitHub Actions (`.github/workflows/tests.yml`) |
| Modules source | `src/brats2026/` (dont `nnunet/`, `ssl/` avec MAE, `evaluation/`) |
| Bloqueur principal | Entraînement réel non lancé (câblé + human-gated, cf. `docs/RUN_HPC.md`) ; config `train.yaml` à remplir par l'ai-specialist |

## Ce qui est construit

| Domaine | Module(s) | Rôle | Statut |
|---|---|---|---|
| Configuration | `config.py`, `tasks.py` | Config challenge + définition des 5 tasks | ✅ |
| Découverte données | `discover.py` | Génération de manifests (JSONL) par task/split | ✅ |
| Domaines / cohortes | `domains.py` | Gestion des domaines GoAT (généralisation cross-entités) | ✅ |
| Conformité | `compliance.py` | Vérifs règles challenge (pas de réseau, données autorisées) | ✅ |
| Traçabilité | `provenance.py` | Provenance des artefacts | ✅ |
| Prétraitement | `preprocess.py` | Normalisation / crop MRI multi-modalités (t1n, t1c, t2f, t2w) | ✅ |
| nnU-Net | `nnunet/convert.py`, `plan.py`, `splits.py`, `trainer.py` | Conversion format nnU-Net, planification, splits, wrapper d'entraînement | ✅ (code) |
| SSL (Phase 4) | `ssl/__init__.py`, `ssl/mae.py` | Self-training itératif (≤2 rounds) + pré-entraînement MAE 3D (substitut FM conforme, sur données GoAT uniquement) | ✅ (code) |
| Inférence | `inference/predict.py`, `postprocess.py` | Prédiction + post-traitement | ✅ |
| Évaluation | `evaluation/metrics.py`, `protocol.py`, `report.py` | Métriques **Dice + HD95 + NSD** (surface, rank-agrégée par le leaderboard), protocole, rapport par-cohorte + LODO | ✅ |
| Empaquetage | `packaging/budget.py`, `geometry.py` | Budget ressources + géométrie pour le conteneur de soumission | ✅ |
| CLI | `cli.py` | Commandes `tasks`, `discover`, `preprocess-mri`, `ssl-select`, `install-trainer` | ✅ (5 commandes exposées) |

**Scaffolding projet** (commit `69a02ea`) : guide projet, équipe de 8 sous-agents Claude Code (`.claude/agents/`), `configs/`, `docker/`, `paper/`.

## Ce qui reste à faire

- **Phase 4 — SSL / self-training** : ✅ **implémentée** (`ssl/__init__.py` + `configs/ssl.yaml`, 33 tests). Self-training itératif borné à 2 rounds, filtrage de confiance par-cas/par-voxel, pool training-only par défaut (hook `allow_validation_pool`), gardes anti-fuite. Reste : valider sur run réel ; **confirmer sur le forum Synapse** si le validation set GoAT est utilisable comme pool non labellisé ; envisager une métrique `mean_entropy` (lot de suivi, cf. spec).
- **Entraînement réel** : câblage terminé — `nnUNetTrainerGoAT` charge/valide `configs/train.yaml` via `$BRATS_GOAT_TRAIN_CONFIG` et refuse de démarrer si non rempli. Séquence de lancement HPC dans **`docs/RUN_HPC.md`**. Restent, marqués `TODO` dans `trainer.initialize()` : sampler domain-balanced, augmentations domain-randomization, poids de loss par région. Lancer d'abord une boucle courte avant le run complet.
- **Exposer dans la CLI** les commandes inférence / évaluation / packaging (déjà exposées : `tasks`, `discover`, `preprocess-mri`, `ssl-select`, `install-trainer`).
- **Conteneur Docker** de soumission à finaliser et valider (sans réseau).
- **Short paper** (dossier `paper/`).
- **Branche Task3** dédiée à pousser (cf. TODO de fin de stage).

## Structure du dépôt

```
brats2026/
├── README.md              Prise en main (install, quickstart)
├── ETAT_PROJET.md         Ce document
├── src/brats2026/         Package (config, discover, domains, nnunet/, inference/, evaluation/, packaging/, ssl/)
├── tests/                 214 tests (miroir de src/)
├── configs/               train.yaml (stub à remplir), infer.yaml, ssl.yaml
├── docker/                Dockerfile (conteneur de soumission)
├── docs/                  RUN_HPC.md (lancement HPC), PLAN.md, ARCHITECTURE.md, AGENT_RUNBOOK.md, synapse_wiki_raw/
├── paper/                 Short paper (à rédiger)
├── .claude/agents/        Équipe de 8 sous-agents
└── .github/workflows/     CI tests
```

## Données

Miroir du dataset attendu à : `/mnt/CPS-RADT/datasets/MICCAI/2026/BraTS2026`. Données soumises aux règles du challenge (pas d'usage de données externes non autorisées — vérifié par `compliance.py`).

## Reproduire / continuer

```bash
source ~/venvs/synapseclient/bin/activate
python -m pip install -e '.[mri]'
python -m pytest -q                       # 214 tests (2 échecs Windows-only attendus)
python -m brats2026.cli tasks             # lister les tasks
python -m brats2026.cli discover --task task3 --split train --output work/manifests/task3_train.jsonl
```

## Points de vigilance

- `.venv/` est local (exclu du git et de la sync NAS) — recréer l'environnement à la prise en main.
- Les hyperparamètres critiques sont balisés `# SPECIALIST:` dans le code (à renseigner avant entraînement).
- L'entraînement réel n'a pas encore tourné : le code nnU-Net est testé unitairement mais pas validé sur un run complet.
