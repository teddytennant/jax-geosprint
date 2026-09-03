"""Algorithm 1: turn a retention profile into a sampling schedule.

Section 3.2 and Algorithm 1 of GeoSPRINT (arXiv:2609.02160). The chain is
retention frequency ``w(t)`` (Eq. 6), smoothed and floored into a curvature
density, blended with a log-SNR density (Eq. 7), integrated into a CDF, and
inverted at ``K`` uniform quantiles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .diffusion import Schedule, log_snr
from .hyperplanarity import (
    PruneResult,
    hyperplanarity_prune,
    normalize_trajectory,
    search_threshold_batched,
)

__all__ = [
    "retention_frequency",
    "smooth_and_floor",
    "logsnr_density",
    "blend_densities",
    "density_cdf",
    "quantile_timesteps",
    "schedule_from_density",
    "curvature_density_from_trajectories",
    "curvature_on_training_grid",
    "geosprint_schedule",
    "prune_trajectory",
    "uniform_schedule",
]


def retention_frequency(retained: jax.Array) -> jax.Array:
    """Eq. 6: ``w(t) = (1/B) sum_b 1[t in R^(b)]``.

    ``retained`` is ``(B, N+1)`` boolean, one row per reference trajectory,
    ``True`` where GeoSPRINT kept that index. The result is a per-timestep
    curvature density: high where trajectories consistently curve, low where
    they are consistently straight enough to prune.
    """
    return jnp.mean(jnp.asarray(retained).astype(jnp.float32), axis=0)


def smooth_and_floor(
    w: jax.Array, sigma: float = 5.0, eps_floor_frac: float = 0.05
) -> jax.Array:
    """``SmoothAndFloor`` of Algorithm 1 line 10.

    A Gaussian kernel of bandwidth ``sigma=5`` timesteps, then a floor at
    ``eps_floor = 0.05 * max_t w_tilde(t)`` so no region of the trajectory can
    be starved of steps entirely. The kernel is normalized per output position,
    which keeps the ends of the grid from being pulled toward zero by the tail
    of the kernel running off the edge.

    Two things the paper leaves open. It writes the floor as
    ``0.05 max w`` in Algorithm 1 and as ``0.05 max_t w_tilde(t)`` in Section
    3.2; the smoothed maximum is used here, following the section text, and the
    two differ because smoothing lowers peaks. It also does not say whether to
    renormalize afterwards, so the output is left unnormalized;
    :func:`blend_densities` normalizes both components before mixing, which is
    what makes ``beta`` a meaningful mixing weight.
    """
    w = jnp.asarray(w, dtype=jnp.float32)
    grid = jnp.arange(w.shape[0], dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * ((grid[:, None] - grid[None, :]) / sigma) ** 2)
    smoothed = (kernel @ w) / jnp.sum(kernel, axis=1)
    return jnp.maximum(smoothed, eps_floor_frac * jnp.max(smoothed))


def logsnr_density(schedule: Schedule) -> jax.Array:
    """``rho_logSNR(t)``, the density whose CDF gives log-SNR-uniform spacing.

    Section 3.2 asks for "uniform density in log-SNR space". Placing steps at
    uniform quantiles of ``F`` is a change of variables, so uniform in
    ``lambda`` means the density in ``t`` is ``|d lambda / dt|``. Taking the
    ``beta=0`` branch of Eq. 7 through :func:`schedule_from_density` therefore
    reproduces the log-SNR-uniform baseline the paper says it should.

    The derivative is a central difference on the integer training grid, and is
    clamped away from zero so the CDF stays strictly increasing.
    """
    lam = log_snr(schedule)
    d = jnp.abs(jnp.gradient(lam))
    return jnp.maximum(d, 1e-12 * jnp.max(d))


def blend_densities(rho_logsnr: jax.Array, rho_curv: jax.Array, beta: float) -> jax.Array:
    """Eq. 7: ``rho = (1 - beta) rho_logSNR + beta rho_curv``.

    Both components are normalized to unit mass first. Without that the blend
    weight would be swamped by whichever density happened to have the larger
    raw scale, and ``beta=0.6`` would not mean 60 percent curvature.
    """
    a = jnp.asarray(rho_logsnr, dtype=jnp.float32)
    b = jnp.asarray(rho_curv, dtype=jnp.float32)
    a = a / jnp.sum(a)
    b = b / jnp.sum(b)
    return (1.0 - beta) * a + beta * b


def density_cdf(rho: jax.Array) -> jax.Array:
    """Algorithm 1 line 12: ``F(t) = int_0^t rho / int_0^T rho``.

    Trapezoidal quadrature on the integer timestep grid, so ``F`` has the same
    length as ``rho``, starts at 0, ends at 1, and is nondecreasing. It is
    strictly increasing whenever ``rho`` is positive everywhere, which the
    floor in :func:`smooth_and_floor` and the clamp in :func:`logsnr_density`
    both guarantee.
    """
    rho = jnp.asarray(rho, dtype=jnp.float32)
    mid = 0.5 * (rho[1:] + rho[:-1])
    cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(mid)])
    return cum / cum[-1]


def quantile_timesteps(cdf: jax.Array, k: int) -> jax.Array:
    """Algorithm 1 line 13: ``K`` timesteps at uniform quantiles of ``F``.

    Returns continuous positions on the timestep grid, ascending. ``F`` is
    inverted by linear interpolation, which is the natural partner to the
    trapezoidal CDF.

    The paper says "uniform quantiles" without pinning down the endpoints.
    ``linspace(0, 1, K)`` is used, so the first and last timestep of the grid
    are always in the schedule. That matters for a sampler: dropping the
    highest-noise step would start the trajectory in the wrong place, and
    dropping the lowest would leave a visible gap before ``t=0``.
    """
    cdf = jnp.asarray(cdf)
    grid = jnp.arange(cdf.shape[0], dtype=jnp.float32)
    q = jnp.linspace(0.0, 1.0, k)
    return jnp.interp(q, cdf, grid)


def _to_integer_timesteps(positions: jax.Array, t_max: int) -> jax.Array:
    """Round continuous positions to distinct integer timesteps, descending.

    Rounding can collide when ``K`` approaches the grid size. Collisions are
    resolved by a forward sweep that pushes each entry at least one step past
    its predecessor, which keeps the schedule strictly monotone and inside
    ``[0, t_max]`` at the cost of moving a few steps by one index.
    """
    n = positions.shape[0]
    if n > t_max + 1:
        raise ValueError(f"cannot place K={n} distinct timesteps on a grid of {t_max + 1}")
    idx = [int(round(float(p))) for p in positions]
    idx[0] = max(0, min(idx[0], t_max - (n - 1)))
    for i in range(1, n):
        idx[i] = max(idx[i], idx[i - 1] + 1)
        idx[i] = min(idx[i], t_max)
    for i in range(n - 2, -1, -1):
        idx[i] = min(idx[i], idx[i + 1] - 1)
    return jnp.asarray(idx[::-1], dtype=jnp.int32)


def schedule_from_density(rho: jax.Array, k: int) -> jax.Array:
    """CDF, invert at ``K`` uniform quantiles, return descending timesteps."""
    positions = quantile_timesteps(density_cdf(rho), k)
    return _to_integer_timesteps(positions, int(rho.shape[0]) - 1)


def curvature_density_from_trajectories(
    trajectories: jax.Array,
    k: int = 2,
    alpha_target: float = 1e-3,
    normalize: bool = True,
    sigma: float = 5.0,
    eps_floor_frac: float = 0.05,
) -> tuple[jax.Array, jax.Array, PruneResult]:
    """Algorithm 1 lines 1-10, on trajectories that have already been recorded.

    ``trajectories`` is ``(B, N+1, d)``, ordered ``t=T`` to ``t=0``. For each
    one, ``tau`` is binary-searched to hit ``alpha_target`` (lines 4-7), the
    pruning is run, and the retained indices become the ``R^(b)`` of line 8.
    Appendix C.2's per-dimension standardization is applied first when
    ``normalize`` is set, so the test reads direction changes rather than raw
    coordinate scale.

    Returns the curvature density on the ``N+1`` index grid, the ``(B,)``
    per-trajectory thresholds, and a batched :class:`PruneResult` whose fields
    all carry a leading ``B`` axis.
    """
    trajectories = jnp.asarray(trajectories)
    z = jax.vmap(normalize_trajectory)(trajectories) if normalize else trajectories
    taus, results = search_threshold_batched(z, k=k, alpha_target=alpha_target)
    w = retention_frequency(results.retained)
    return smooth_and_floor(w, sigma, eps_floor_frac), taus, results


def curvature_on_training_grid(
    curvature: jax.Array, diffusion: Schedule, ref_timesteps: jax.Array | None = None
) -> jax.Array:
    """Algorithm 1 line 8, the index-to-timestep map, made explicit.

    The curvature density comes out of pruning indexed by position along the
    reference trajectory, ``0 .. N``. Eq. 7 blends it with a density defined on
    the training timestep grid, so it has to be moved there first. Pass
    ``ref_timesteps``, the timesteps the reference trajectory was recorded at,
    and index ``i`` is placed at ``ref_timesteps[i]`` and linearly interpolated
    between.

    With ``ref_timesteps`` left out the reference pass is assumed to be uniform
    in ``t``, which is what the paper's 200-step reference is. Note the
    direction: trajectories are ordered ``t=T`` first down to ``t=0`` last
    (Section 3.1), so index 0 is the *highest* timestep. Reading the default
    the other way silently mirrors the curvature profile, which for a bimodal
    profile like Figure 3(a)'s is not obviously wrong from the shape alone.
    """
    curvature = jnp.asarray(curvature)
    n_train = diffusion.num_train_timesteps
    if ref_timesteps is None:
        src = jnp.linspace(float(n_train - 1), 0.0, curvature.shape[0])
    else:
        src = jnp.asarray(ref_timesteps, dtype=jnp.float32)
        if src.shape[0] != curvature.shape[0]:
            raise ValueError(
                f"curvature has {curvature.shape[0]} entries but ref_timesteps has "
                f"{src.shape[0]}"
            )
    order = jnp.argsort(src)
    return jnp.interp(jnp.arange(n_train, dtype=jnp.float32), src[order], curvature[order])


def geosprint_schedule(
    curvature: jax.Array,
    diffusion: Schedule,
    k_steps: int,
    beta: float = 0.6,
    ref_timesteps: jax.Array | None = None,
) -> jax.Array:
    """Algorithm 1 lines 11-14, the blended schedule.

    ``curvature`` is the output of :func:`curvature_density_from_trajectories`,
    defined on the ``N+1`` indices of the reference trajectory. It is mapped
    onto the training-timestep grid by
    :func:`curvature_on_training_grid` before blending, because the reference
    pass has ``N=200`` steps while the log-SNR density lives on all 1000
    training timesteps.

    ``beta=0.6`` is the paper's setting, tuned on CIFAR-10 with a broad optimum
    over 0.5 to 0.7. ``beta=0`` gives pure log-SNR spacing and ``beta=1`` pure
    curvature spacing, which Section 5.3 reports are both worse than the blend.

    Returns ``K`` strictly decreasing integer timesteps.
    """
    curvature = curvature_on_training_grid(curvature, diffusion, ref_timesteps)
    rho = blend_densities(logsnr_density(diffusion), curvature, beta)
    return schedule_from_density(rho, k_steps)


def uniform_schedule(diffusion: Schedule, k_steps: int) -> jax.Array:
    """The DDIM baseline: ``K`` uniformly spaced timesteps, descending.

    Spaced across the full grid with both endpoints included, so it and a
    GeoSPRINT schedule of the same ``K`` cover the same range and differ only
    in where the interior steps land.
    """
    n_train = diffusion.num_train_timesteps
    positions = jnp.linspace(0.0, float(n_train - 1), k_steps)
    return _to_integer_timesteps(positions, n_train - 1)


def prune_trajectory(
    z: jax.Array, k: int = 2, tau: float = 1e-3, normalize: bool = True
) -> PruneResult:
    """Convenience wrapper: standardize, then run Algorithm 2 at fixed ``tau``."""
    z = normalize_trajectory(z) if normalize else jnp.asarray(z)
    return hyperplanarity_prune(z, k=k, tau=tau)
