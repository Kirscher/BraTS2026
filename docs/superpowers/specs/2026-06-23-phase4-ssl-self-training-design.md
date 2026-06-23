# Phase 4 — SSL / self-training (pseudo-labelling) — Design

> BraTS 2026 Task 3 (GoAT). Module `src/brats2026/ssl/`. Validé par le Lead après
> consultation du conseil (`ai-specialist` + `compliance-guard`), 2026-06-23.

## Objectif

Améliorer la généralisation cross-cohorte (objectif worst-cohort sur GLI/SSA/MEN/MET/PED)
en exploitant les cas **non labellisés propres à GoAT** par self-training : un *teacher*
entraîné from-scratch sur les cas labellisés génère des pseudo-labels, filtrés par confiance,
sur lesquels un *student* est ré-entraîné. Borné à 2 rounds.

Comme le reste du package, ce module **ne lance aucun GPU** : il fournit des **fonctions
pures** + une **dataclass de config** (hooks `# SPECIALIST:` à `None`) + des **constructeurs
d'argv** (`nnUNetv2_predict` / `nnUNetv2_train`). Les valeurs numériques sont la propriété de
l'`ai-specialist` (`configs/ssl.yaml`), le run réel est gaté humainement.

## Décisions figées

- **Schéma : self-training itératif, borné à `n_rounds=2`.** `n_rounds` reste un hook
  SPECIALIST (mettable à 1). Single-round = sous-cas `n_rounds=1`. Pas de Mean-Teacher
  (réécrirait la boucle nnU-Net) — on se compose *par-dessus* `nnUNetTrainerGoAT`.
- **Filtrage de confiance par-cas ET par-voxel.** Voxels sous le seuil → `ignore-label`
  (exclus de la loss). Cas sous le seuil / quasi-vides / hallucinés → rejetés.
- **Pool non labellisé paramétrable, training-only par défaut.** Hook
  `allow_validation_pool` (défaut `False` = cas non labellisés du training set uniquement,
  subset MEN/MET sans GT). Le validation set GoAT n'entre dans le pool que si le forum
  Synapse le confirme ET si `allow_validation_pool=True`.

## Budgets (rappel)

- **Inférence conteneur (8h / A10G 24GB)** : INCHANGÉE. Le modèle livré est un
  `nnUNetTrainerGoAT` standard ; le SSL est entièrement training-time.
- **Entraînement (offline)** : chaque round = un `nnUNetv2_train` complet + une génération de
  pseudo-labels. 2 rounds ≈ 2× le wall-clock ; `n_rounds` (et les epochs côté trainer) se
  calent sur le budget réel par l'`ai-specialist`.

## Architecture

Module unique `src/brats2026/ssl/__init__.py` (miroir de `nnunet/trainer.py`), réutilisant
`inference/predict.py` (builders d'argv), `domains.cohort_from_case_id`, `compliance.*`
(provenance) et `nnunet/splits` (étanchéité LODO).

### Config

```python
SPECIALIST_HOOKS: tuple[str, ...] = (...)   # noms des champs ci-dessous

@dataclass
class GoATSelfTrainingConfig:
    n_rounds: Optional[int] = None                       # SPECIALIST: 1–2
    confidence_threshold_voxel: Optional[float] = None   # SPECIALIST: softmax max/voxel, ~0.75–0.95
    confidence_threshold_case: Optional[float] = None    # SPECIALIST: score agrégé/cas, ~0.85
    case_confidence_metric: Optional[str] = None         # SPECIALIST: mean_fg_softmax|mean_entropy|frac_confident_voxels
    min_pseudo_foreground_voxels: Optional[int] = None   # SPECIALIST: rejette cas quasi-vides, ~100–500
    max_pseudo_foreground_fraction: Optional[float] = None  # SPECIALIST: anti-hallucination, ~0.5
    labeled_unlabeled_ratio: Optional[float] = None      # SPECIALIST: 1.0–3.0
    per_cohort_pseudo_quota: Optional[bool] = None       # SPECIALIST: plafonne #pseudo/cohorte
    pseudo_label_softmax_temperature: Optional[float] = None  # SPECIALIST: défaut 1.0
    ignore_label_index: Optional[int] = None             # SPECIALIST: index ignore-label nnU-Net
    pseudo_use_tta: Optional[bool] = None                # SPECIALIST: mirroring à la génération
    pseudo_use_ensemble: Optional[bool] = None           # SPECIALIST: ensemble folds inner-dev = teacher
    teacher_folds: Optional[tuple[int, ...]] = None      # SPECIALIST: folds teacher (inner-dev only)
    recompute_pseudo_each_round: Optional[bool] = None   # SPECIALIST: cœur du self-training itératif
    allow_validation_pool: bool = False                  # NON-SPECIALIST: gate règlement; défaut sûr
    seed: Optional[int] = None                           # SPECIALIST: RNG seed (provenance)
```

`allow_validation_pool` n'est **pas** un hook SPECIALIST (c'est une règle de conformité, pas
un hyperparamètre) → il a un défaut sûr et n'apparaît pas dans `SPECIALIST_HOOKS` ni
`unset_hooks`.

### Fonctions pures

```python
def unset_hooks(config) -> list[str]            # miroir trainer.py (ignore allow_validation_pool)
def assert_configured(config) -> None           # refuse si un hook SPECIALIST est None

@dataclass
class CaseStats:                                 # agrégats par cas (pas de pixels bruts)
    case_id: str
    mean_fg_softmax: float
    n_fg_voxels: int
    fg_fraction: float
    frac_confident_voxels: float

def voxel_confidence_mask(max_softmax, threshold, temperature=1.0)  # -> masque bool des voxels confiants
def case_confidence_score(stats, metric) -> float
def accept_case(stats, config) -> bool           # seuil cas + min_fg + max_fg_fraction
def select_pseudo_cases(case_stats, cohort_of, config) -> dict[str, list[str]]  # + quota/cohorte
def labeled_unlabeled_sampling_plan(n_labeled, n_pseudo, ratio) -> dict[str, float]

def pseudo_label_predict_commands(teacher_model_dir, unlabeled_input_dir, pseudo_output_dir, config) -> list[list[str]]
def self_training_round_commands(round_index, teacher_model_dir, train_data_dir, unlabeled_input_dir, work_root, config) -> list[list[str]]
def self_training_plan(config, ...) -> list[list[list[str]]]   # n_rounds; teacher[r] = student[r-1]
```

### Gardes de conformité (intégrées, assertions fatales)

- `assert_goat_pool(case_ids, allow_validation_pool, validation_ids=...)` :
  - tout `case_id` → `cohort_from_case_id(case_id)` ∈ cohortes connues (rejet `UNK` = fatal) ;
  - si `allow_validation_pool=False`, `set(pool) ∩ set(validation_ids) == ∅` (fatal) ;
  - invariant `set(pool) ∩ set(outer_lodo_fold) == ∅` (fatal).
- Réutiliser `compliance.check_data_provenance` / `check_no_prior_weights` sur le pool et le
  teacher avant le 1er pseudo-label.
- Traçabilité : chaque cas pseudo-labellisé doit pouvoir être marqué dans le ledger
  `label_source="pseudo"` + `teacher_ckpt_sha` + seuils appliqués (champ `extra` du
  fingerprint). Le module expose les valeurs nécessaires ; l'écriture ledger reste côté
  `provenance.py` / `data-pipeline`.

## Anti-leakage (protocole inner-dev / outer-LODO)

- Le teacher n'est entraîné que sur l'inner-dev ; `teacher_folds` ne recoupe jamais l'outer-LODO.
- Le pool non labellisé est disjoint des splits de validation (inner + outer) — certifié en
  amont par `data-pipeline`, vérifié par assertion ici.
- Aucun hook SSL n'est tuné contre l'outer-LODO (réglage sur rapports inner-dev uniquement).

## Tests (TDD, sans GPU ni nnU-Net)

- `unset_hooks`/`assert_configured` : config fraîche = tous les hooks SPECIALIST manquants ;
  `allow_validation_pool` exclu du compte ; assert lève tant qu'un hook est None.
- `voxel_confidence_mask` : seuillage correct, effet de la température.
- `case_confidence_score`/`accept_case` : seuil cas, `min_fg`, `max_fg_fraction`, métriques.
- `select_pseudo_cases` : quota par cohorte on/off ; rejets ; n'inclut pas les cas refusés.
- `labeled_unlabeled_sampling_plan` : ratios extrêmes, normalisation.
- builders d'argv : 1 commande/membre d'ensemble, toggles TTA/ensemble, chemins/flags.
- `self_training_plan` : `n_rounds=1` vs `2`, chaînage teacher[r]=student[r-1].
- gardes : `UNK` → lève ; collision validation pool quand `allow_validation_pool=False` → lève ;
  `allow_validation_pool=True` → autorisé.

## Répartition

1. `python-dev` : implémente le module + tests (ce design), suite pytest verte.
2. `ai-specialist` : produit `configs/ssl.yaml` (header provenance : git SHA, dataset
   fingerprint, seed, parent run-id, date + valeurs des hooks). Gate humain avant run réel.
3. `data-pipeline` : garantit la disjonction pool ↔ validation/LODO (en amont).
4. `compliance-guard` : audit final ledger ↔ container.

## Hors scope

Pas de run GPU, pas de boucle d'entraînement réécrite, pas d'exposition CLI dans ce lot
(la CLI training/inférence/SSL sera un lot séparé). Pas d'usage du validation set tant que le
forum Synapse n'a pas confirmé.
