"""Phase 1/2 — nnU-Net v2 conversion, domain-grouped splits, plan/preprocess and the
GoAT trainer scaffold.

Nothing in this subpackage invents training hyperparameters: every tunable is exposed as
a ``# SPECIALIST:`` hook to be set by the human AI specialist (see ``trainer.py``).
"""
