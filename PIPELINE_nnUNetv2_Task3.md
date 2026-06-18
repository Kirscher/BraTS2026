# Pipeline nnU-Net v2 — BraTS 2026 Task 3 (GoAT)

> **But du document.** Présenter le pipeline complet proposé pour la Task 3 (BraTS-GoAT)
> afin qu'il soit **relu et validé par le tuteur** avant de lancer la préparation des
> données et l'entraînement du modèle.
>
> **Convention importante.** Toutes les valeurs d'entraînement/inférence (LR, schedule,
> epochs, batch/patch size, poids de loss, magnitudes d'augmentation, TTA, seuils de
> post-traitement) sont laissées en **hooks `# SPECIALIST:`** : elles seront fixées par le
> spécialiste IA, **pas inventées ici**. Les points à confirmer côté tuteur/portail sont
> signalés par ⚠️.

---

## 1. Contexte du défi


**Task 3 = BraTS-GoAT** (*Generalizability Across Tumors*) : **un seul** modèle de
segmentation doit généraliser sur **5 cohortes** et à travers scanners/démographies.

| Cohorte | Code |
|---------|------|
| Glioblastome adulte | `GLI` |
| Afrique sub-saharienne | `SSA` |
| Méningiome | `MEN` |
| Métastases | `MET` |
| Pédiatrique | `PED` |

- **Entrée par cas** : 4 modalités `t1n, t1c, t2f, t2w` (toutes requises) → cible `seg`.
- **Labels harmonisés** : `NCR=1`, `ED=2`, `ET=3` (fond `0`).
- **Scoring** : régions **ET = {3}**, **TC = {1,3}**, **WT = {1,2,3}**, via **Dice + HD95**
  (agrégés par rang). Sensibilité/Spécificité/Précision = indicatifs seulement.
- La cohorte est encodée dans le préfixe d'ID : `BraTS-<GLI|SSA|MEN|MET|PED>-...`
  → **labels de domaine gratuits** pour le sampling équilibré et la validation
  *leave-one-domain-out*.

### Contraintes dures (non négociables)

- **★ Règle GoAT « no external data »** : **uniquement** les données fournies par
  BraTS-GoAT. Toute donnée externe/privée **ou** tout poids pré-entraîné issu d'un BraTS
  antérieur → **disqualification**. ⇒ **entraînement from scratch**. SSL/semi-supervisé
  autorisé **seulement** sur les données GoAT.
- **Container (inférence)** : A10G **24 GB**, **32 GiB** RAM, **200 GB** disque,
  **8 h d'inférence totale**, **pas de réseau**, `/input` read-only, `/output` plat.
  **Géométrie NIfTI préservée à l'identique.**
- **NAS read-only** → tout artefact dérivé écrit sous `work/`.

### Échéances

| Jalon | Date |
|-------|------|
| Ouverture des queues Docker | **17 juin** |
| Early final (sinon verrouillé hors finale) | **2 juillet** |
| Final | **23 juillet** |
| Papier court LNCS (8–10 pages, obligatoire) | avec la finale |

---

## 2. Vue d'ensemble du pipeline

```
PHASE 0  Données & conformité      ──►  PHASE 1  Préprocessing nnU-Net v2
   │                                          │
   ▼                                          ▼
PHASE 2  Config entraînement  ◄──boucle──►  PHASE 3  Entraînement & QC par cohorte
   │                                          │
   ▼                                          ▼
PHASE 4  Novelty SSL / semi-sup    ──►  PHASE 5  Inférence & ensemble
                                            │
                                            ▼
                                       PHASE 6  Packaging container  ──►  PHASE 7  Papier LNCS
```

---

## 3. Détail des phases

### Phase 0 — Données & conformité
- Vérifier le **data-root réel** sur disque. Défaut code `tasks.py:DEFAULT_DATA_ROOT =
  /mnt/CPS-RADT/...`, mais Task 3 annoncée à
  `/mnt/NAS2418_RADT/datasets/MICCAI/2026/BraTS2026/Task3`.
  `--data-root` = **parent de `Task3`** (le code ajoute `Task3`).
- ⚠️ Les roots `task3` du code utilisent les **noms 2024** (`MICCAI2024-BraTS-GoAT-*`) ;
  le layout **2026** peut différer. Si `brats2026 discover` renvoie **0 cas** → patcher les
  roots.
- Inspection **structure/headers/métadonnées uniquement** (`nib.load().shape/.affine/.header`,
  sans `get_fdata`) — données contrôlées sous accord Synapse.
- **Gate conformité** : aucune donnée externe, aucun poids prior-BraTS. Entraînement from
  scratch confirmé.

### Phase 1 — Préprocessing & conversion nnU-Net v2
- Convertir les 4 modalités → dataset nnU-Net v2 (channels `0..3` = `t1n, t1c, t2f, t2w`).
- Cible **region-based** dérivée des labels harmonisés (`ET / TC / WT`).
- Extraire le **domaine** depuis le préfixe d'ID → table de domaines pour folds groupés.
- `nnUNetv2_plan_and_preprocess` ; **préserver la géométrie** ; ⚠️ confirmer
  resampling/atlas/skull-stripping (skull-stripped ? atlas-resampled ?).
- Sorties sous `work/` : `nnUNet_raw/`, `nnUNet_preprocessed/`.

### Phase 2 — Configuration d'entraînement *(hand-off spécialiste)*
- **Architecture** : nnU-Net v2 **ResEnc-L**, configuration **3d_fullres**, sortie
  region-based.
- **Sampling équilibré par domaine** sur les 5 cohortes (éviter le biais GLI majoritaire).
- **Augmentation** : *domain randomization* (intensité / contraste / bruit) pour pousser la
  généralisation inter-scanner.
- **Validation** : **5-fold** + split **leave-one-domain-out** (mesure directe de la
  généralisation hors-domaine).
- `# SPECIALIST:` LR · schedule · epochs · batch_size · patch_size · poids de loss ·
  magnitudes d'augmentation.

### Phase 3 — Entraînement & boucle qualité
- Entraîner les **5 folds** (from scratch, données GoAT uniquement).
- Scoring **Dice + HD95** sur **ET/TC/WT**, **par cohorte** (pas seulement la moyenne).
- **Boucle Phase 3 ↔ Phase 2** jusqu'à ce que **la pire cohorte** soit acceptable.

### Phase 4 — Novelty : SSL / semi-supervisé *(données GoAT seules)*
- Exploiter les cas **MEN/MET non labellisés** (GT seulement sur un sous-ensemble) →
  *self-training* / pseudo-labels.
- Ré-entraîner avec pseudo-labels ; **prouver le gain** par leave-one-domain-out.
- `# SPECIALIST:` seuils de confiance des pseudo-labels · ratio labellisé/non-labellisé.

### Phase 5 — Inférence & ensemble
- **TTA** par *mirroring*.
- **Ensemble** ResEnc-L + **MedNeXt** / **SegResNet**.
- **Post-traitement** : *ET-suppression* (suppression des petits ET sous seuil de volume).
- ⚠️ **Cohérence métrique** : *legacy-overlap* vs *lesion-wise* Dice/HD95 — **change le
  post-traitement**. À trancher avant de figer le seuil.
- `# SPECIALIST:` seuil de volume ET · membres de l'ensemble · poids.

### Phase 6 — Packaging container
- Image respectant : A10G 24 GB · 32 GiB RAM · 200 GB · **offline** · **8 h total**.
- `/input` read-only → `/output` plat ; **géométrie NIfTI identique** à l'entrée.
- **Vérifier le budget temps** sur l'ensemble du test set (les 8 h sont une limite dure).
- Soumission via la queue Docker (17 juin → early 2 juil → final 23 juil).

### Phase 7 — Papier LNCS
- 8–10 pages : méthode, protocole leave-one-domain-out, **ablation SSL**, résultats
  **par cohorte**, analyse de généralisation.

---

## 4. Points à valider en priorité (côté tuteur / portail Synapse)

1. ⚠️ **Type de métrique** : *legacy-overlap* vs *lesion-wise* Dice/HD95 — impacte
   directement le post-traitement (Phase 5).
2. ⚠️ **Layout des données 2026** : noms de dossiers réels vs noms 2024 codés — bloquant
   Phase 0/1.
3. ⚠️ **Prétraitement fourni** : volumes skull-stripped / atlas-resampled ? — conditionne
   le préprocessing (Phase 1).
4. **Choix de l'ensemble** (ResEnc-L seul vs + MedNeXt/SegResNet) vs **budget 8 h**
   d'inférence — arbitrage performance/temps.

---

## 5. Récapitulatif de l'approche

> nnU-Net v2 **ResEnc-L** (3d_fullres) **5-fold**, **sampling équilibré par domaine** +
> **domain-randomization**, **mirroring TTA**, **post-proc ET-suppression** ; **ensemble**
> avec MedNeXt/SegResNet ; **novelty = SSL + self-training semi-supervisé** sur les cohortes
> non labellisées de GoAT, **prouvé par leave-one-domain-out**. Entraînement **from scratch**,
> **aucune donnée externe ni poids prior-BraTS**.
