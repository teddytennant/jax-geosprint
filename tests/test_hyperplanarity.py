"""Definition 1, Algorithm 2, Eq. 8, Proposition 1 and Theorem 1."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geosprint import (
    affine_basis,
    alpha_traj,
    hyperplanarity_prune,
    normalize_trajectory,
    residual_distance,
    search_threshold,
    theorem1_bound,
)

D = 16


def collinear_trajectory(n=40, d=D, seed=0):
    """Points exactly on a line, at non-uniform spacing along it."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    direction = jax.random.normal(k1, (d,))
    direction = direction / jnp.linalg.norm(direction)
    origin = 0.3 * jax.random.normal(k2, (d,))
    s = jnp.sort(jax.random.uniform(k3, (n,)))
    return origin + s[:, None] * direction


def planar_trajectory(n=40, d=D, seed=3):
    """Points exactly in a 2D affine subspace, so a k=3 window spans them."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    basis, _ = jnp.linalg.qr(jax.random.normal(k1, (d, 2)))
    origin = 0.3 * jax.random.normal(k2, (d,))
    coeffs = jax.random.uniform(k3, (n, 2))
    return origin + coeffs @ basis.T


def curved_trajectory(n=40, d=D, seed=5, curvature=1.0):
    """A line plus a sinusoidal excursion orthogonal to it."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    basis, _ = jnp.linalg.qr(jax.random.normal(k1, (d, 2)))
    tangent, normal = basis[:, 0], basis[:, 1]
    s = jnp.linspace(0.0, 1.0, n)
    bend = curvature * jnp.sin(2.0 * jnp.pi * s)
    return 0.3 * jax.random.normal(k2, (d,)) + s[:, None] * tangent + bend[:, None] * normal


def test_collinear_trajectory_has_zero_residuals_and_zero_alpha():
    """Proposition 1: a perfectly straight trajectory prunes to the window.

    Everything past the initial ``k`` points sits exactly on the line the
    window spans, so every residual is zero, the window never slides, and the
    numerator of Eq. 8 is empty.
    """
    z = collinear_trajectory()
    result = hyperplanarity_prune(z, k=2, tau=1e-4)

    assert np.max(np.asarray(result.residuals)) < 1e-5
    assert int(jnp.sum(result.retained)) == 2
    assert bool(jnp.all(result.retained[:2]))
    assert float(alpha_traj(z, result)) == pytest.approx(0.0, abs=1e-10)


def test_planar_trajectory_needs_the_level_two_test():
    """A planar sweep is caught by k=3 and missed by k=2.

    Section 3.1's progressive hierarchy: level 1 tests collinearity, level 2
    coplanarity. A 2D affine subspace is invisible to level 1.
    """
    z = planar_trajectory()

    level2 = hyperplanarity_prune(z, k=3, tau=1e-4)
    assert np.max(np.asarray(level2.residuals)) < 1e-5
    assert int(jnp.sum(level2.retained)) == 3
    assert float(alpha_traj(z, level2)) == pytest.approx(0.0, abs=1e-10)

    level1 = hyperplanarity_prune(z, k=2, tau=1e-4)
    assert int(jnp.sum(level1.pruned)) == 0


def test_qr_residual_matches_least_squares_oracle():
    """Eq. 5 via QR agrees with an independent pseudoinverse computation.

    ``lstsq`` solves ``min_x ||M_W x - (z - w_1)||``, so the norm of its
    residual is exactly ``||(I - M_W M_W^+)(z - w_1)||``. Checked on random
    windows of several sizes, including ``k-1`` larger than 1 where the QR and
    the pseudoinverse could plausibly disagree.
    """
    rng = np.random.default_rng(0)
    for k in (2, 3, 5):
        for _ in range(20):
            window = jnp.asarray(rng.normal(size=(k, D)), dtype=jnp.float32)
            z = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float32)

            got = float(residual_distance(window, z))

            m = np.asarray(window[1:] - window[0]).T
            v = np.asarray(z - window[0])
            coef, *_ = np.linalg.lstsq(m, v, rcond=None)
            want = float(np.linalg.norm(v - m @ coef))

            assert got == pytest.approx(want, rel=1e-4, abs=1e-5)


def test_affine_basis_drops_rank_deficient_columns():
    """A stalled window has a zero column in ``M_W`` and must not span it.

    ``[0 | e_1]`` has rank one. A factorization that returned two orthonormal
    columns here would project away a direction the affine span does not
    contain, and the residual would come out smaller than the pseudoinverse
    form of Eq. 5 says it is.
    """
    window = jnp.stack([jnp.zeros((D,)), jnp.zeros((D,)), jnp.eye(D)[0]])
    q = affine_basis(window)
    assert int(np.sum(np.linalg.norm(np.asarray(q), axis=0) > 1e-6)) == 1

    z = jnp.eye(D)[1]
    assert float(residual_distance(window, z)) == pytest.approx(1.0, rel=1e-5)


def test_theorem1_bound_holds_on_random_trajectories():
    """Eq. 9: ``alpha_traj <= |S| tau^2 / ((N+1) tr(C_Z))``.

    Random walks, which are about as far from straight as a trajectory gets,
    swept over five orders of magnitude of ``tau`` so the pruned fraction runs
    from nearly none to nearly all.
    """
    for seed in range(5):
        steps = jax.random.normal(jax.random.PRNGKey(seed), (60, D))
        z = jnp.cumsum(steps, axis=0)
        for tau in (1e-2, 1e-1, 0.5, 1.0, 2.0, 5.0, 20.0):
            result = hyperplanarity_prune(z, k=2, tau=tau)
            got = float(alpha_traj(z, result))
            bound = float(theorem1_bound(z, result, tau))
            assert got <= bound + 1e-9, f"seed={seed} tau={tau}: {got} > {bound}"


def test_bound_is_tight_when_every_residual_sits_at_the_threshold():
    """The proof bounds each ``r_s`` by ``tau``, so a slack ratio near 1 needs
    residuals near ``tau``. This just confirms the bound is not vacuous: at a
    threshold that prunes a lot, the ratio stays within a couple of orders of
    magnitude rather than being astronomically loose."""
    z = jnp.cumsum(jax.random.normal(jax.random.PRNGKey(1), (60, D)), axis=0)
    result = hyperplanarity_prune(z, k=2, tau=5.0)
    assert int(jnp.sum(result.pruned)) > 10
    ratio = float(alpha_traj(z, result)) / float(theorem1_bound(z, result, 5.0))
    assert 0.01 < ratio <= 1.0


def test_curved_trajectory_scores_higher_than_a_near_straight_one():
    """``alpha_traj`` is a measure of non-straightness (Corollary 1).

    Same length, same ambient dimension, same threshold; the only difference is
    how far the path bends away from its own chord.

    The threshold is set high enough that every candidate is pruned in all five
    cases, so the pruned set is identical and the comparison is over residuals
    alone. That is not decoration. At a threshold that prunes only some points,
    ``alpha_traj`` is not monotone in curvature: a more curved path fails the
    test more often, so it contributes fewer terms to the numerator and a
    larger denominator, and the score can fall as the path bends further. The
    paper's own Table 2 reads ``alpha_traj`` next to the pruning rate for this
    reason. It is a score for a given pruning, not a curvature functional.
    """
    tau = 100.0
    scores = []
    for curvature in (0.001, 0.01, 0.05, 0.1, 0.3):
        z = curved_trajectory(curvature=curvature)
        result = hyperplanarity_prune(z, k=2, tau=tau)
        assert int(jnp.sum(result.pruned)) == z.shape[0] - 2
        scores.append(float(alpha_traj(z, result)))

    assert scores == sorted(scores)
    assert scores[0] < 1e-3 < scores[-1]


def test_reparameterization_along_a_straight_path_scores_zero():
    """Section 4.3: tangential acceleration does not contribute to the residual.

    Two trajectories on the same line, one evenly spaced and one that speeds up
    cubically along it. The residual measures orthogonal distance from the
    extrapolated line, so both score zero even though their velocity profiles
    are nothing alike.
    """
    direction = jnp.asarray(np.linalg.qr(np.random.default_rng(7).normal(size=(D, 1)))[0][:, 0])

    even = jnp.linspace(0.0, 1.0, 50)[:, None] * direction
    accelerating = (jnp.linspace(0.0, 1.0, 50) ** 3)[:, None] * direction

    for z in (even, accelerating):
        result = hyperplanarity_prune(z, k=2, tau=1e-4)
        assert float(alpha_traj(z, result)) == pytest.approx(0.0, abs=1e-10)
        assert int(jnp.sum(result.retained)) == 2
        assert np.max(np.asarray(result.residuals)) < 1e-5

    # At a threshold that prunes both of them, a bending path does not score
    # zero, so the zeros above are not just an empty pruned set.
    curved = curved_trajectory(n=50, curvature=0.3)
    assert float(alpha_traj(curved, hyperplanarity_prune(curved, k=2, tau=100.0))) > 0.1
    for z in (even, accelerating):
        assert float(alpha_traj(z, hyperplanarity_prune(z, k=2, tau=100.0))) < 1e-10


def test_pruning_is_monotone_in_tau_at_the_first_divergence():
    """Raising ``tau`` cannot un-prune a point before the first disagreement.

    Past that point the window states differ and the sets are free to diverge,
    which is exactly why :func:`search_threshold` documents its objective as
    only approximately monotone.
    """
    z = jnp.cumsum(jax.random.normal(jax.random.PRNGKey(2), (60, D)), axis=0)
    lo = np.asarray(hyperplanarity_prune(z, k=2, tau=0.5).pruned)
    hi = np.asarray(hyperplanarity_prune(z, k=2, tau=1.0).pruned)
    diff = np.flatnonzero(lo != hi)
    prefix = diff[0] if diff.size else len(lo)
    assert np.all(hi[:prefix] >= lo[:prefix])
    if diff.size:
        assert hi[diff[0]] and not lo[diff[0]]


def test_search_threshold_lands_under_the_target():
    """Algorithm 1 lines 4-7 return a ``tau`` whose score respects the target."""
    for seed in range(4):
        z = jnp.cumsum(jax.random.normal(jax.random.PRNGKey(seed), (80, D)), axis=0)
        for target in (1e-4, 1e-3, 1e-2):
            tau, result = search_threshold(z, k=2, alpha_target=target)
            score = float(alpha_traj(z, result))
            assert score <= target * (1 + 1e-6)
            assert float(tau) > 0.0


def test_search_threshold_prunes_more_as_the_target_loosens():
    z = jnp.cumsum(jax.random.normal(jax.random.PRNGKey(11), (120, D)), axis=0)
    counts = []
    for target in (1e-5, 1e-3, 1e-1):
        _, result = search_threshold(z, k=2, alpha_target=target)
        counts.append(int(jnp.sum(result.pruned)))
    assert counts[0] <= counts[1] <= counts[2]
    assert counts[2] > counts[0]


def test_normalize_trajectory_standardizes_and_leaves_flat_dimensions_alone():
    z = jnp.concatenate(
        [jax.random.normal(jax.random.PRNGKey(0), (30, 3)) * 100.0, jnp.ones((30, 1))], axis=1
    )
    out = normalize_trajectory(z)
    assert np.allclose(np.asarray(jnp.mean(out[:, :3], axis=0)), 0.0, atol=1e-5)
    assert np.allclose(np.asarray(jnp.std(out[:, :3], axis=0)), 1.0, atol=1e-4)
    assert np.all(np.isfinite(np.asarray(out)))


def test_prune_rejects_windows_it_cannot_initialize():
    with pytest.raises(ValueError):
        hyperplanarity_prune(jnp.zeros((3, D)), k=1)
    with pytest.raises(ValueError):
        hyperplanarity_prune(jnp.zeros((3, D)), k=3)
