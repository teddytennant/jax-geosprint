"""A 2D toy diffusion model, small enough to train on CPU in a few seconds.

GeoSPRINT is evaluated in the paper on CIFAR-10, LSUN Church and Stable
Diffusion v1.5. None of those fit here. What this module gives instead is a
real eps-prediction model with a real curved denoising trajectory, which is
everything the schedule construction needs: reference trajectories to prune, a
curvature profile to extract, and a sampler to spend an NFE budget on.

The target is the two-moons distribution and the model is an MLP over
``(z, sinusoidal features of t)``, trained with the usual epsilon-matching
objective.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax

from .diffusion import Schedule, linear_beta_schedule

__all__ = [
    "TOY_DIM",
    "eps_apply",
    "init_params",
    "make_eps_fn",
    "sliced_wasserstein",
    "train",
    "two_moons",
]

TOY_DIM = 2
_HIDDEN = 128
_N_FREQS = 16


def two_moons(key: jax.Array, n: int, noise: float = 0.06) -> jax.Array:
    """Sample the two-moons target, scaled to roughly unit variance.

    Two interleaving half circles. The scaling matters: the diffusion forward
    process assumes data on a standard-normal-ish scale, and the trajectories
    that come out of an unscaled target are dominated by one direction.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    half = n // 2
    t_out = jax.random.uniform(k1, (half,)) * jnp.pi
    t_in = jax.random.uniform(k2, (n - half,)) * jnp.pi
    outer = jnp.stack([jnp.cos(t_out), jnp.sin(t_out)], axis=1)
    inner = jnp.stack([1.0 - jnp.cos(t_in), 0.5 - jnp.sin(t_in)], axis=1)
    x = jnp.concatenate([outer, inner], axis=0)
    x = x + noise * jax.random.normal(k3, x.shape)
    x = (x - jnp.array([0.5, 0.25])) / jnp.array([0.9, 0.55])
    return x


def _time_features(t: jax.Array, num_train_timesteps: int) -> jax.Array:
    """Sinusoidal features of the normalized timestep, broadcast over a batch."""
    frac = jnp.asarray(t, dtype=jnp.float32) / float(num_train_timesteps)
    freqs = 2.0 ** jnp.arange(_N_FREQS, dtype=jnp.float32)
    ang = 2.0 * jnp.pi * frac[..., None] * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def init_params(key: jax.Array) -> list[tuple[jax.Array, jax.Array]]:
    """Three hidden layers, He-scaled, plain lists of weight and bias pairs."""
    sizes = [TOY_DIM + 2 * _N_FREQS, _HIDDEN, _HIDDEN, _HIDDEN, TOY_DIM]
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        params.append((w, jnp.zeros((fan_out,))))
    return params


def eps_apply(
    params, z: jax.Array, t: jax.Array, num_train_timesteps: int
) -> jax.Array:
    """Predicted noise for a batch ``z`` of shape ``(B, 2)`` at timestep ``t``."""
    tf = _time_features(jnp.broadcast_to(jnp.asarray(t), z.shape[:1]), num_train_timesteps)
    h = jnp.concatenate([z, tf], axis=-1)
    for w, b in params[:-1]:
        h = jax.nn.silu(h @ w + b)
    w, b = params[-1]
    return h @ w + b


def train(
    key: jax.Array,
    schedule: Schedule | None = None,
    steps: int = 4000,
    batch: int = 512,
    lr: float = 2e-3,
) -> tuple[list, Schedule]:
    """Epsilon-matching training loop.

    Draws a fresh two-moons batch and a uniform timestep every step, noises the
    batch with the forward process, and regresses the sampled noise. Runs in a
    single jitted scan.
    """
    if schedule is None:
        schedule = linear_beta_schedule(1000)
    n_train = schedule.num_train_timesteps
    key, sub = jax.random.split(key)
    params = init_params(sub)
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps))
    opt_state = opt.init(params)

    def loss_fn(params, key):
        k1, k2, k3 = jax.random.split(key, 3)
        x0 = two_moons(k1, batch)
        t = jax.random.randint(k2, (batch,), 0, n_train)
        noise = jax.random.normal(k3, x0.shape)
        ab = schedule.alphas_bar[t][:, None]
        z = jnp.sqrt(ab) * x0 + jnp.sqrt(1.0 - ab) * noise
        tf = _time_features(t, n_train)
        h = jnp.concatenate([z, tf], axis=-1)
        for w, b in params[:-1]:
            h = jax.nn.silu(h @ w + b)
        w, b = params[-1]
        return jnp.mean((h @ w + b - noise) ** 2)

    @jax.jit
    def scan_body(carry, key):
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, key)
        updates, opt_state = opt.update(grads, opt_state)
        return (optax.apply_updates(params, updates), opt_state), loss

    keys = jax.random.split(key, steps)
    (params, _), _ = jax.lax.scan(scan_body, (params, opt_state), keys)
    return params, schedule


def make_eps_fn(params, schedule: Schedule) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Bind trained params into the ``eps_fn(z, t)`` the sampler expects."""
    n_train = schedule.num_train_timesteps

    def eps_fn(z, t):
        return eps_apply(params, z, t, n_train)

    return eps_fn


def sliced_wasserstein(a: jax.Array, b: jax.Array, key: jax.Array, n_proj: int = 256) -> float:
    """Sliced 1-Wasserstein distance between two point clouds of equal size.

    Project both onto random unit directions, sort each projection, and average
    the mean absolute difference. Used as the sample-quality metric in place of
    FID, which needs an Inception network and tens of thousands of images.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    dirs = jax.random.normal(key, (n_proj, a.shape[1]))
    dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
    pa = jnp.sort(a @ dirs.T, axis=0)
    pb = jnp.sort(b @ dirs.T, axis=0)
    return float(jnp.mean(jnp.abs(pa - pb)))
