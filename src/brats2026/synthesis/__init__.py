"""Synthesis — the DiffTumor 3-stage data-synthesis track for GoAT (independent of MAE/SSL).

Unlike :mod:`brats2026.ssl` (which pre-trains / self-trains the *segmenter*), this package grows
extra training data by *painting* synthetic tumours into GoAT volumes, then scoring a segmenter on
the augmented set. It is a chain of three stages that run offline, from scratch, on GoAT data only:

1. **Stage 1 — autoencoder** (:mod:`brats2026.synthesis.autoencoder`): a 3D AutoencoderKL that
   compresses → reconstructs a whole volume and is frozen afterwards; it defines the latent space
   the rest of the track operates in. Trained on ALL patches (healthy + tumour-bearing, label-free).
2. **Stage 2 — masks**: sample plausible tumour masks / locations to condition synthesis on.
3. **Stage 3 — diffusion**: a latent diffusion model, conditioned on Stage-2 masks, paints tumours
   inside Stage-1's frozen latent; decode → synthetic labelled volumes that feed an nnU-Net, which
   is then scored (data → diffusion → nnU-Net → score).

All three stages are implemented here (Stage 1 as a frozen AE; Stages 2–3 as pure geometry / sizing
logic plus torch scaffolds). Like the rest of the package this module invents no numbers — every
tunable is a ``# SPECIALIST:`` hook resolved in ``configs/synthesis/*.yaml``.

To keep every module importable without torch, the Stage-1/2 torch classes are re-exported here but
:mod:`brats2026.synthesis.generate` imports them only *inside* its scaffold body — never at import
time. Names are grouped by stage; each stage owns its own ``SPECIALIST_HOOKS`` /
``assert_configured`` / ``unset_hooks`` (accessed via the submodule to avoid collision), while the
Stage-1 trio is re-exported at top level for backwards compatibility.
"""
from __future__ import annotations

from . import dataset, diffusion, generate, masks, pipeline
from .dataset import (
    SYNTH_MANIFEST_NAME,
    apply_synth_dataset,
    discover_synthetic_cases,
    is_synthetic_case_id,
    plan_synth_dataset,
    read_synthetic_ids,
    synth_dataset_name,
    train_only_splits,
)
from .autoencoder import (
    SPECIALIST_HOOKS,
    AutoencoderKLConfig,
    GoATAutoencoderKL,
    assert_configured,
    assert_patch_divisible,
    downsample_factor,
    kl_divergence_standard_normal,
    latent_shape_from_config,
    latent_spatial_shape,
    load_ae_config,
    num_latent_elements,
    reconstruction_l1,
    unset_hooks,
)
from .diffusion import (
    BETA_SCHEDULES,
    GoATLatentDiffusion,
    LatentDiffusionConfig,
    alphas_cumprod,
    cosine_beta_schedule,
    hole_latent,
    linear_beta_schedule,
    load_diffusion_config,
    q_sample_coefficients,
)
from .generate import (
    assert_synthetic_compliant,
    generate_synthetic_case,
    synthetic_case_id,
)
from .masks import (
    PLACEMENT_MODES,
    SyntheticMaskConfig,
    assert_region_nesting,
    brain_mask_from_volume,
    concentric_labels,
    contralateral_region,
    ellipsoid_mask,
    healthy_candidate_mask,
    load_mask_config,
    mm_to_voxels,
    place_synthetic_tumor,
)
from .pipeline import (
    SynthesisPoolConfig,
    allocate_per_cohort,
    load_pool_config,
    n_synthetic_from_ratio,
    synth_dataset_id,
    synthesis_plan,
)

__all__ = [
    # submodules
    "masks",
    "diffusion",
    "generate",
    "pipeline",
    "dataset",
    # Stage 3 — augmented dataset assembly
    "SYNTH_MANIFEST_NAME",
    "apply_synth_dataset",
    "discover_synthetic_cases",
    "is_synthetic_case_id",
    "plan_synth_dataset",
    "read_synthetic_ids",
    "synth_dataset_name",
    "train_only_splits",
    # Stage 1 — autoencoder
    "SPECIALIST_HOOKS",
    "AutoencoderKLConfig",
    "GoATAutoencoderKL",
    "assert_configured",
    "assert_patch_divisible",
    "downsample_factor",
    "kl_divergence_standard_normal",
    "latent_shape_from_config",
    "latent_spatial_shape",
    "load_ae_config",
    "num_latent_elements",
    "reconstruction_l1",
    "unset_hooks",
    # Stage 2 — diffusion
    "BETA_SCHEDULES",
    "GoATLatentDiffusion",
    "LatentDiffusionConfig",
    "alphas_cumprod",
    "cosine_beta_schedule",
    "hole_latent",
    "linear_beta_schedule",
    "load_diffusion_config",
    "q_sample_coefficients",
    # Stage 3 — masks
    "PLACEMENT_MODES",
    "SyntheticMaskConfig",
    "assert_region_nesting",
    "brain_mask_from_volume",
    "concentric_labels",
    "contralateral_region",
    "ellipsoid_mask",
    "healthy_candidate_mask",
    "load_mask_config",
    "mm_to_voxels",
    "place_synthetic_tumor",
    # Stage 3 — generate
    "assert_synthetic_compliant",
    "generate_synthetic_case",
    "synthetic_case_id",
    # Stage 3 — pipeline
    "SynthesisPoolConfig",
    "allocate_per_cohort",
    "load_pool_config",
    "n_synthetic_from_ratio",
    "synth_dataset_id",
    "synthesis_plan",
]
