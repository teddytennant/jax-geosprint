"""The whole pipeline on a 2D toy diffusion model.

What this can and cannot show. The pruning, the score and the schedule
construction are all exact and are tested as such elsewhere. The paper's
headline claim, that a GeoSPRINT schedule beats uniform DDIM at matched NFE,
is an empirical claim about CIFAR-10, LSUN Church and Stable Diffusion. It does
not reproduce here, and the tests below say so rather than dressing it up.

Measured on this model, over K in {10, 20, 30, 40} and three noise seeds of
4000 samples each, sliced Wasserstein to the two-moons target:

    K=10  geo(0.6) 0.0823   uniform 0.0741
    K=20  geo(0.6) 0.0442   uniform 0.0525
    K=30  geo(0.6) 0.0460   uniform 0.0494
    K=40  geo(0.6) 0.0380   uniform 0.0343

GeoSPRINT wins at 20 and 30, loses at 10 and 40. Wider sweeps over K and over
the reference batch size move the individual numbers around but never turn this
into a consistent win, so the assertions here are the parts that do hold every
time: the blend beats pure curvature spacing at every budget, error falls as
the budget grows, and the schedule stays within a small factor of uniform.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import geosprint as g
import geosprint.toy as toy

N_SAMPLES = 4000
SEEDS = (0, 1, 2)


def _swd(eps_fn, diffusion, timesteps, target):
    """Mean sliced Wasserstein to the target over a few noise seeds."""
    scores = []
    for seed in SEEDS:
        z_t = jax.random.normal(jax.random.PRNGKey(1000 + seed), (N_SAMPLES, toy.TOY_DIM))
        samples = g.ddim_sample(eps_fn, z_t, timesteps, diffusion)
        scores.append(toy.sliced_wasserstein(samples, target, jax.random.PRNGKey(7)))
    return float(np.mean(scores))


@pytest.fixture(scope="module")
def target():
    return toy.two_moons(jax.random.PRNGKey(9), N_SAMPLES)


def test_the_toy_model_actually_learned_the_target(toy_model, target):
    """Without this, every comparison below would be comparing noise."""
    eps_fn, diffusion = toy_model
    full = _swd(eps_fn, diffusion, g.uniform_schedule(diffusion, 200), target)
    floor = toy.sliced_wasserstein(
        toy.two_moons(jax.random.PRNGKey(21), N_SAMPLES), target, jax.random.PRNGKey(7)
    )
    assert floor < 0.02
    assert full < 0.05


def test_the_metric_notices_a_bad_schedule(toy_model, reference, target):
    """A schedule that spends its whole budget at high noise is far worse.

    If the metric could not separate that from a sane schedule, none of the
    comparisons in this file would mean anything.
    """
    eps_fn, diffusion = toy_model
    wasted = jnp.asarray(np.linspace(999, 700, 20).round().astype(np.int32))
    bad = _swd(eps_fn, diffusion, wasted, target)

    geo = _swd(eps_fn, diffusion, g.geosprint_schedule(reference["curvature"], diffusion, 20), target)
    uniform = _swd(eps_fn, diffusion, g.uniform_schedule(diffusion, 20), target)

    assert bad > 5 * geo
    assert bad > 5 * uniform


def test_blend_beats_pure_curvature_spacing_at_every_budget(toy_model, reference, target):
    """Section 3.2: beta=1 "fails at low NFE due to coverage gaps".

    Table 2 shows the same ordering on CIFAR-10 at every NFE. It is the one
    part of the paper's schedule ablation that does hold on this toy model.
    """
    eps_fn, diffusion = toy_model
    curvature = reference["curvature"]
    for k in (10, 20, 30, 40):
        blended = _swd(eps_fn, diffusion, g.geosprint_schedule(curvature, diffusion, k, 0.6), target)
        pure = _swd(eps_fn, diffusion, g.geosprint_schedule(curvature, diffusion, k, 1.0), target)
        assert blended < pure, f"K={k}: blend {blended:.4f} not better than curvature {pure:.4f}"


def test_geosprint_error_falls_as_the_budget_grows(toy_model, reference, target):
    eps_fn, diffusion = toy_model
    scores = [
        _swd(eps_fn, diffusion, g.geosprint_schedule(reference["curvature"], diffusion, k), target)
        for k in (10, 20, 40)
    ]
    assert scores[0] > scores[1] > scores[2]


def test_geosprint_stays_close_to_uniform_at_matched_nfe(toy_model, reference, target):
    """The honest version of the paper's headline claim at this scale.

    GeoSPRINT is better than uniform DDIM at two of these four budgets and
    worse at the other two, so what is asserted is that it is never much worse,
    and that it wins somewhere. The paper's FID improvements are not reproduced
    here and nothing in this file claims they are.
    """
    eps_fn, diffusion = toy_model
    curvature = reference["curvature"]
    ratios = []
    for k in (10, 20, 30, 40):
        geo = _swd(eps_fn, diffusion, g.geosprint_schedule(curvature, diffusion, k), target)
        uniform = _swd(eps_fn, diffusion, g.uniform_schedule(diffusion, k), target)
        ratios.append(geo / uniform)

    assert max(ratios) < 1.25
    assert min(ratios) < 1.0


def test_reference_pass_prunes_most_of_a_denoising_trajectory(reference):
    """Section 5.5 reports 82 percent of a DDIM trajectory pruned at the
    threshold that holds alpha_traj near 1e-3. A 2D toy trajectory is straighter
    still, so it prunes harder."""
    result = reference["result"]
    fraction = float(jnp.mean(result.pruned))
    assert 0.75 < fraction < 0.99
    assert np.all(np.asarray(reference["taus"]) > 0)


def test_reference_alpha_traj_lands_under_its_target(reference):
    """Every per-trajectory binary search respected alpha_target=1e-3."""
    z = jax.vmap(g.normalize_trajectory)(reference["trajectories"])
    scores = np.asarray(jax.vmap(g.alpha_traj)(z, reference["result"]))
    assert np.all(scores <= 1e-3 * (1 + 1e-6))
    assert np.median(scores) > 1e-6  # not trivially zero from an empty pruned set


def test_curvature_profile_is_bimodal(reference):
    """Figure 3(a): retention peaks at both ends of the trajectory.

    Part of the peak at t=T is forced, not measured: Algorithm 2 initializes
    its window with the first k points, so those two indices are retained by
    every trajectory and w(t)=1 there by construction. The peak at low t is
    the real signal, the fine-detail phase the paper attributes it to.
    """
    w = np.asarray(reference["curvature"])
    mid = w[len(w) // 4 : 3 * len(w) // 4]
    assert w[:10].max() > 2 * mid.mean()
    assert w[-40:].max() > 1.5 * mid.mean()
    assert int(np.argmax(w[len(w) // 2 :])) + len(w) // 2 > 3 * len(w) // 4


def test_ddim_sampler_respects_its_nfe_budget(toy_model):
    """The schedule is the NFE count: one model call per timestep, no more."""
    eps_fn, diffusion = toy_model
    calls = []

    def counting(z, t):
        calls.append(t)
        return eps_fn(z, t)

    timesteps = g.uniform_schedule(diffusion, 17)
    z_t = jax.random.normal(jax.random.PRNGKey(0), (8, toy.TOY_DIM))
    g.ddim_sample(counting, z_t, timesteps, diffusion)
    assert len(calls) == 1  # traced once inside lax.scan, applied 17 times

    traj = g.ddim_sample(counting, z_t, timesteps, diffusion, return_trajectory=True)
    assert traj.shape == (18, 8, toy.TOY_DIM)
