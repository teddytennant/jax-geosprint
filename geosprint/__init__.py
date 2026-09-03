"""GeoSPRINT: geometric redundancy-aware step pruning for diffusion schedules.

Implements arXiv:2609.02160. The entry points most people want:

* :func:`geosprint.hyperplanarity.hyperplanarity_prune` (Algorithm 2)
* :func:`geosprint.hyperplanarity.alpha_traj` (Eq. 8)
* :func:`geosprint.schedule.geosprint_schedule` (Algorithm 1)
* :func:`geosprint.diffusion.ddim_sample`
"""

from .diffusion import Schedule, ddim_sample, linear_beta_schedule, log_snr, record_trajectory
from .hyperplanarity import (
    PruneResult,
    affine_basis,
    alpha_traj,
    hyperplanarity_prune,
    normalize_trajectory,
    residual_distance,
    search_threshold,
    search_threshold_batched,
    theorem1_bound,
)
from .schedule import (
    blend_densities,
    curvature_density_from_trajectories,
    curvature_on_training_grid,
    density_cdf,
    geosprint_schedule,
    logsnr_density,
    prune_trajectory,
    quantile_timesteps,
    retention_frequency,
    schedule_from_density,
    smooth_and_floor,
    uniform_schedule,
)

__all__ = [
    "PruneResult",
    "Schedule",
    "affine_basis",
    "alpha_traj",
    "blend_densities",
    "curvature_density_from_trajectories",
    "curvature_on_training_grid",
    "ddim_sample",
    "density_cdf",
    "geosprint_schedule",
    "hyperplanarity_prune",
    "linear_beta_schedule",
    "log_snr",
    "logsnr_density",
    "normalize_trajectory",
    "prune_trajectory",
    "quantile_timesteps",
    "record_trajectory",
    "residual_distance",
    "retention_frequency",
    "schedule_from_density",
    "search_threshold",
    "search_threshold_batched",
    "smooth_and_floor",
    "theorem1_bound",
    "uniform_schedule",
]
