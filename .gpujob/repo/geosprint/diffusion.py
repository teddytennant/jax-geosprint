"""Variance-preserving diffusion and a DDIM sampler that takes any schedule.

Section 5.1 of GeoSPRINT uses the standard linear beta schedule with 1000
training timesteps, and Appendix C.2 notes that the DDIM update has to be
written out by hand because the usual scheduler implementations assume the
timesteps are uniformly spaced. Both of those apply here: everything below
takes an explicit, arbitrarily spaced descending array of timesteps.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

__all__ = ["Schedule", "linear_beta_schedule", "log_snr", "ddim_sample", "record_trajectory"]


class Schedule(NamedTuple):
    """Forward-process constants on the full ``T``-step training grid.

    Attributes:
        betas: ``(T,)`` per-step noise variance.
        alphas_bar: ``(T,)`` cumulative product ``prod_{s<=t} (1 - beta_s)``,
            usually written ``alpha_bar_t``.
    """

    betas: jax.Array
    alphas_bar: jax.Array

    @property
    def num_train_timesteps(self) -> int:
        return self.betas.shape[0]


def linear_beta_schedule(
    num_train_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
) -> Schedule:
    """The linear beta schedule of Ho et al., as used by the paper's models.

    The endpoints are the values that go with ``T=1000``. Shrinking ``T``
    without rescaling them would leave far too little total noise at ``t=T``,
    so they are scaled by ``1000 / T``, which is the usual convention.
    """
    scale = 1000.0 / num_train_timesteps
    betas = jnp.linspace(beta_start * scale, beta_end * scale, num_train_timesteps)
    return Schedule(betas=betas, alphas_bar=jnp.cumprod(1.0 - betas))


def log_snr(schedule: Schedule) -> jax.Array:
    """``lambda_t = log(alpha_bar_t / (1 - alpha_bar_t))``, the log-SNR.

    This is the quantity Eq. 7's ``rho_logSNR`` is uniform in, and it decreases
    monotonically with ``t``. The paper does not print the formula; this is the
    standard variance-preserving definition that DPM-Solver and DPM-Solver++
    use, which is what Section 3.2 points at when it says "as in DPM-Solver".
    """
    ab = schedule.alphas_bar
    return jnp.log(ab) - jnp.log1p(-ab)


def ddim_sample(
    eps_fn: Callable[[jax.Array, jax.Array], jax.Array],
    z_t: jax.Array,
    timesteps: jax.Array,
    schedule: Schedule,
    return_trajectory: bool = False,
) -> jax.Array:
    """Deterministic DDIM (eta=0) over an arbitrary descending timestep list.

    ``timesteps`` is ``(K,)``, strictly decreasing integer indices into the
    training grid, highest noise first. The update for a step from ``t`` to the
    next entry ``t_prev`` is

        z_prev = sqrt(ab_prev) * x0_hat + sqrt(1 - ab_prev) * eps,
        x0_hat = (z_t - sqrt(1 - ab_t) * eps) / sqrt(ab_t),

    with ``ab_prev = 1`` on the final step, which lands on the clean sample.
    Nothing here assumes the gaps between timesteps are equal, which is the
    whole point: a GeoSPRINT schedule is deliberately non-uniform.

    ``eps_fn(z, t)`` takes a batch ``(B, d)`` and a scalar timestep index and
    returns the predicted noise. Exactly ``K`` calls are made, so ``K`` is the
    NFE budget.

    Returns the final sample, or if ``return_trajectory``, the ``(K+1, B, d)``
    stack of states from ``z_T`` through ``z_0``.
    """
    timesteps = jnp.asarray(timesteps)
    ab = schedule.alphas_bar[timesteps]
    ab_prev = jnp.concatenate([schedule.alphas_bar[timesteps[1:]], jnp.ones((1,))])

    def step(z, args):
        t, a_t, a_prev = args
        eps = eps_fn(z, t)
        x0 = (z - jnp.sqrt(1.0 - a_t) * eps) / jnp.sqrt(a_t)
        z_next = jnp.sqrt(a_prev) * x0 + jnp.sqrt(1.0 - a_prev) * eps
        return z_next, z_next

    z_final, states = jax.lax.scan(step, z_t, (timesteps, ab, ab_prev))
    if return_trajectory:
        return jnp.concatenate([z_t[None], states], axis=0)
    return z_final


def record_trajectory(
    eps_fn: Callable[[jax.Array, jax.Array], jax.Array],
    z_t: jax.Array,
    timesteps: jax.Array,
    schedule: Schedule,
) -> jax.Array:
    """Algorithm 1 line 2, ``SampleFull``: one reference trajectory per sample.

    Returns ``(B, K+1, d)``, so index ``b`` is the ordered point sequence
    ``{z_T, ..., z_0}`` that Section 3.1 feeds to the hyperplanarity test.
    """
    traj = ddim_sample(eps_fn, z_t, timesteps, schedule, return_trajectory=True)
    return jnp.swapaxes(traj, 0, 1)
