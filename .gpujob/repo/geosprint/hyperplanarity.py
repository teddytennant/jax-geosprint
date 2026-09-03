"""Hyperplanarity test and trajectory projection score.

This is Section 3.1 and 3.3 of GeoSPRINT (arXiv:2609.02160):

* Definition 1 / Eq. 4 and 5, the residual distance of a candidate point from
  the affine subspace spanned by a window of retained points.
* Algorithm 2, ``HyperplanarityPrune``, the causal streaming version of that
  test with a sliding window of the ``k`` most recently retained points.
* Definition 2 / Eq. 8, the trajectory projection score ``alpha_traj``.
* Theorem 1 / Eq. 9, the bound ``alpha_traj <= |S| tau^2 / ((N+1) tr(C_Z))``.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "PruneResult",
    "affine_basis",
    "residual_distance",
    "hyperplanarity_prune",
    "alpha_traj",
    "theorem1_bound",
    "search_threshold",
    "search_threshold_batched",
    "normalize_trajectory",
]


class PruneResult(NamedTuple):
    """Output of Algorithm 2.

    Attributes:
        pruned: boolean mask over the ``N+1`` trajectory indices, ``True`` where
            the point was found geometrically redundant and removed.
        residuals: the hyperplanarity residual ``r_i`` of Eq. 5 at every index.
            Entries at retained indices are the residual that failed the test
            (``>= tau``); the first ``k`` window-initializing indices are 0.
        retained: ``~pruned``, the set ``R`` of Algorithm 1 line 8, as a mask.
    """

    pruned: jax.Array
    residuals: jax.Array
    retained: jax.Array


def affine_basis(window: jax.Array, rcond: float = 1e-10) -> jax.Array:
    """Orthonormal basis for the affine span of a window, via thin QR.

    ``window`` is ``(k, d)`` holding ``w_1 ... w_k``. Builds ``M_W`` of Eq. 4,
    the ``d x (k-1)`` matrix of offsets ``w_j - w_1``, and returns the ``Q``
    factor of its thin QR factorization. Eq. 5 is written with the
    Moore-Penrose pseudoinverse, ``I - M_W M_W^+``, and the paper then swaps in
    ``I - Q Q^T``; the two projectors agree as long as ``M_W`` has full column
    rank.

    Rank deficiency breaks that agreement, and the factorization is written out
    as modified Gram-Schmidt rather than handed to ``jnp.linalg.qr`` so the
    deficiency can be caught. Householder QR without column pivoting does not
    reveal rank in its diagonal: give it ``[0 | e_1]`` and it returns
    ``R = [[0, 1], [0, 0]]``, so both diagonal entries vanish while the matrix
    has rank one. Gram-Schmidt instead exposes the rank directly, as the norm
    of each column after the earlier ones are projected out, and a column that
    contributes nothing is dropped to zero. Cost is still ``O(d k^2)``.

    The degenerate case is not reachable from Algorithm 2 at ``k=2``: the
    window can only stall if a retained point coincides with the previous one,
    and a coincident point has residual 0 and would have been pruned. It is
    reachable at ``k >= 3``, where the initial window ``z_0 ... z_{k-1}`` can be
    collinear from the start.
    """
    offsets = (window[1:] - window[0]).T  # (d, k-1), this is M_W
    scale = jnp.max(jnp.linalg.norm(offsets, axis=0))
    cutoff = rcond * jnp.maximum(scale, jnp.finfo(offsets.dtype).tiny)

    basis = jnp.zeros_like(offsets)
    for j in range(offsets.shape[1]):
        v = offsets[:, j]
        v = v - basis @ (basis.T @ v)
        v = v - basis @ (basis.T @ v)  # one reorthogonalization for stability
        norm = jnp.linalg.norm(v)
        column = jnp.where(norm > cutoff, v / jnp.where(norm > 0, norm, 1.0), 0.0)
        basis = basis.at[:, j].set(column)
    return basis


def residual_distance(window: jax.Array, z: jax.Array, rcond: float = 1e-10) -> jax.Array:
    """Eq. 5: ``r_i = ||(I - M_W M_W^+)(z_i - w_1)||_2``, computed from the QR.

    ``window`` is ``(k, d)``, ``z`` is ``(d,)``. Forming ``I - Q Q^T`` densely
    would cost ``O(d^2)``; the residual is taken as ``v - Q (Q^T v)`` instead,
    which is the ``O(d k)`` the paper's complexity claim assumes.
    """
    q = affine_basis(window, rcond)
    v = z - window[0]
    return jnp.linalg.norm(v - q @ (q.T @ v))


def hyperplanarity_prune(
    z: jax.Array, k: int = 2, tau: float = 1e-3, rcond: float = 1e-10
) -> PruneResult:
    """Algorithm 2, ``HyperplanarityPrune``.

    Args:
        z: ``(N+1, d)`` ordered trajectory, from ``t=T`` down to ``t=0``.
        k: window size. ``k=2`` is the collinearity test the paper uses
            everywhere; ``k=3`` is coplanarity, and so on.
        tau: threshold. A point with ``r_i < tau`` is pruned.
        rcond: relative rank cutoff passed to :func:`affine_basis`.

    Window semantics, which the paper states twice and are easy to get wrong:
    the window holds the ``k`` most recently *retained* points, not the ``k``
    immediately preceding points. It is initialized to ``z_0 ... z_{k-1}``,
    which are therefore always retained. A pruned point leaves the window
    untouched, so a long redundant run keeps being measured against the same
    affine subspace. Only a retained point slides the window, dropping ``w_1``
    and appending ``z_i``.

    The QR is recomputed at every candidate rather than cached across the
    pruned run as in Algorithm 2 line 12, because the loop is a ``lax.scan``
    with a fixed body. That changes the constant in the cost, not the output:
    a cached factorization and a recomputed one describe the same subspace.

    Returns:
        A :class:`PruneResult`. Indices ``0 .. k-1`` are never pruned.
    """
    z = jnp.asarray(z)
    n_points, _ = z.shape
    if k < 2:
        raise ValueError(f"window size k must be at least 2, got {k}")
    if n_points <= k:
        raise ValueError(f"need more than k={k} points to prune, got {n_points}")
    return _prune_core(z, k, jnp.asarray(tau, dtype=z.dtype), jnp.asarray(rcond, dtype=z.dtype))


@partial(jax.jit, static_argnums=(1,))
def _prune_core(z: jax.Array, k: int, tau: jax.Array, rcond: jax.Array) -> PruneResult:
    """Jitted body of Algorithm 2. ``tau`` is traced, so the binary search of
    Algorithm 1 can sweep it without triggering a recompilation per candidate.
    """

    def step(window, zi):
        r = residual_distance(window, zi, rcond)
        prune = r < tau
        slid = jnp.concatenate([window[1:], zi[None, :]], axis=0)
        window = jnp.where(prune, window, slid)
        return window, (prune, r)

    _, (prune_tail, resid_tail) = jax.lax.scan(step, z[:k], z[k:])

    head = jnp.zeros((k,), dtype=bool)
    pruned = jnp.concatenate([head, prune_tail])
    residuals = jnp.concatenate([jnp.zeros((k,), dtype=z.dtype), resid_tail])
    return PruneResult(pruned=pruned, residuals=residuals, retained=~pruned)


def alpha_traj(z: jax.Array, result: PruneResult) -> jax.Array:
    """Eq. 8, the trajectory projection score.

    ``sum_{s in S} r_s^2`` over ``sum_{z in Z} ||z - z_bar||^2``, where
    ``z_bar`` is the global trajectory mean. The denominator is exactly
    ``(N+1) tr(C_Z)`` for the sample covariance ``C_Z`` normalized by ``N+1``,
    which is the form Theorem 1 uses.

    Note what the numerator is not: it is the residual variance orthogonal to
    the local affine span of the retained neighbors, not the raw variance of
    the pruned points. The paper's remark after Theorem 1 is explicit that
    pruning interior points of a collinear trajectory shrinks the retained
    sample covariance while leaving this score at zero.
    """
    z = jnp.asarray(z)
    num = jnp.sum(jnp.where(result.pruned, result.residuals, 0.0) ** 2)
    den = jnp.sum((z - jnp.mean(z, axis=0)) ** 2)
    return num / den


def theorem1_bound(z: jax.Array, result: PruneResult, tau: float) -> jax.Array:
    """Right-hand side of Eq. 9, ``|S| tau^2 / ((N+1) tr(C_Z))``.

    Written the way the proof in Appendix B reaches it, so the denominator is
    the same total squared deviation :func:`alpha_traj` divides by.
    """
    z = jnp.asarray(z)
    n_pruned = jnp.sum(result.pruned)
    den = jnp.sum((z - jnp.mean(z, axis=0)) ** 2)
    return n_pruned * tau**2 / den


def normalize_trajectory(z: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Zero mean, unit variance per dimension, as Appendix C.2 specifies.

    The paper applies this before curvature analysis so the hyperplanarity test
    responds to direction changes rather than to the raw scale of each
    coordinate. Dimensions with no variation are left alone instead of being
    divided by roughly zero.
    """
    z = jnp.asarray(z)
    std = jnp.std(z, axis=0)
    return (z - jnp.mean(z, axis=0)) / jnp.where(std > eps, std, 1.0)


def search_threshold(
    z: jax.Array,
    k: int = 2,
    alpha_target: float = 1e-3,
    lo: float = 1e-12,
    hi: float = 1e3,
    iters: int = 40,
) -> tuple[jax.Array, PruneResult]:
    """Algorithm 1 lines 4-7: binary search ``tau`` to hit ``alpha_target``.

    The paper says only "adjust ``tau`` until ``alpha_traj`` is approximately
    ``alpha_target``", so the bracketing is a choice made here. Bisection runs
    on ``log tau``, since useful thresholds span many orders of magnitude, and
    keeps the invariant that ``lo`` is feasible. The largest feasible ``tau``
    is returned, which is what "while maximizing the number of removed steps"
    asks for: at a fixed window state, raising the threshold cannot prune fewer
    points.

    ``alpha_traj`` is not exactly monotone in ``tau``, because raising the
    threshold changes which points enter the window and so changes later
    residuals. Bisection on a non-monotone objective can settle on a local
    crossing rather than the global one; in practice the score is close enough
    to monotone for this to be the procedure the paper intends.

    Returns the chosen ``tau`` and the pruning it produces. If even ``lo``
    already overshoots ``alpha_target``, ``lo`` comes back unchanged.
    """
    tau = _bisect_tau(jnp.asarray(z), k, alpha_target, lo, hi, iters)
    return tau, hyperplanarity_prune(z, k=k, tau=tau)


@partial(jax.jit, static_argnums=(1, 5))
def _bisect_tau(
    z: jax.Array, k: int, alpha_target: float, lo: float, hi: float, iters: int
) -> jax.Array:
    """Bisection on ``log tau``, written so it can be jitted and vmapped."""
    dtype = z.dtype

    def alpha_of(tau):
        res = _prune_core(z, k, tau, jnp.asarray(1e-10, dtype=dtype))
        return alpha_traj(z, res)

    def body(_, carry):
        log_lo, log_hi, best = carry
        log_mid = 0.5 * (log_lo + log_hi)
        feasible = alpha_of(jnp.exp(log_mid)) <= alpha_target
        return (
            jnp.where(feasible, log_mid, log_lo),
            jnp.where(feasible, log_hi, log_mid),
            jnp.where(feasible, jnp.exp(log_mid), best),
        )

    log_lo = jnp.asarray(jnp.log(lo), dtype=dtype)
    log_hi = jnp.asarray(jnp.log(hi), dtype=dtype)
    init = (log_lo, log_hi, jnp.asarray(lo, dtype=dtype))
    _, _, best = jax.lax.fori_loop(0, iters, body, init)
    # If the loosest threshold is already feasible, take it outright.
    return jnp.where(alpha_of(jnp.asarray(hi, dtype=dtype)) <= alpha_target, hi, best)


def search_threshold_batched(
    z: jax.Array,
    k: int = 2,
    alpha_target: float = 1e-3,
    lo: float = 1e-12,
    hi: float = 1e3,
    iters: int = 40,
) -> tuple[jax.Array, PruneResult]:
    """:func:`search_threshold` over a stack of ``(B, N+1, d)`` trajectories.

    Algorithm 1 searches a separate ``tau`` per reference trajectory, which is
    embarrassingly parallel; this vmaps it rather than looping in Python.
    """
    z = jnp.asarray(z)
    taus = jax.vmap(lambda zb: _bisect_tau(zb, k, alpha_target, lo, hi, iters))(z)
    results = jax.vmap(lambda zb, t: _prune_core(zb, k, t, jnp.asarray(1e-10, dtype=z.dtype)))(
        z, taus
    )
    return taus, results
