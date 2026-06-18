# configs/ — specialist hyperparameter templates

Templates the **human AI specialist** fills in. The code never reads invented numbers from
here without a value being set explicitly. See `# SPECIALIST:` hooks in:

- `src/brats2026/nnunet/trainer.py` — LR, schedule, epochs, batch/patch size, loss weights,
  augmentation magnitudes
- `src/brats2026/ssl/pseudolabel.py` — pseudo-label confidence threshold, labeled/unlabeled ratio
- `src/brats2026/inference/postprocess.py` — ET-suppression volume threshold, TTA, ensemble weights
