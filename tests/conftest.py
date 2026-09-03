"""Shared toy diffusion setup.

Training and the 200-step reference pass are the only slow things in the suite,
so they happen once per session and every end-to-end test reuses them.
"""

import jax
import pytest

import geosprint as g
import geosprint.toy as toy

TRAIN_STEPS = 6000
REFERENCE_BATCH = 64
REFERENCE_N = 200


@pytest.fixture(scope="session")
def toy_model():
    """A trained 2D eps-predictor and its forward-process constants."""
    params, diffusion = toy.train(jax.random.PRNGKey(0), steps=TRAIN_STEPS)
    return toy.make_eps_fn(params, diffusion), diffusion


@pytest.fixture(scope="session")
def reference(toy_model):
    """Algorithm 1 lines 1-10: reference trajectories and the curvature density.

    The reference pass runs at ``N=200`` steps as in Section 5.1, on a batch of
    64 rather than the paper's 100, which is enough for a 2D model.
    """
    eps_fn, diffusion = toy_model
    ref_ts = g.uniform_schedule(diffusion, REFERENCE_N + 1)
    z_t = jax.random.normal(jax.random.PRNGKey(11), (REFERENCE_BATCH, toy.TOY_DIM))
    traj = g.record_trajectory(eps_fn, z_t, ref_ts, diffusion)[:, : REFERENCE_N + 1]
    curvature, taus, result = g.curvature_density_from_trajectories(
        traj, k=2, alpha_target=1e-3
    )
    return dict(
        ref_timesteps=ref_ts,
        trajectories=traj,
        curvature=curvature,
        taus=taus,
        result=result,
    )
