"""Run the whole pipeline on the 2D toy model and print the comparison.

Trains an eps-predictor, records reference trajectories, prunes them, builds a
GeoSPRINT schedule at the requested NFE, and reports sample quality against the
uniform DDIM schedule of the same length.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import (
    alpha_traj,
    curvature_density_from_trajectories,
    ddim_sample,
    geosprint_schedule,
    normalize_trajectory,
    record_trajectory,
    uniform_schedule,
)
from .toy import TOY_DIM, make_eps_fn, sliced_wasserstein, train, two_moons


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="geosprint.demo", description=__doc__)
    parser.add_argument("--nfe", type=int, default=20, help="sampling budget K")
    parser.add_argument("--beta", type=float, default=0.6, help="blend weight of Eq. 7")
    parser.add_argument("--alpha-target", type=float, default=1e-3)
    parser.add_argument("--window", type=int, default=2, help="hyperplanarity window k")
    parser.add_argument("--reference-batch", type=int, default=64)
    parser.add_argument("--reference-steps", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    start = time.time()
    key = jax.random.PRNGKey(args.seed)
    train_key, ref_key, sample_key, target_key, proj_key = jax.random.split(key, 5)

    params, diffusion = train(train_key, steps=args.train_steps)
    eps_fn = make_eps_fn(params, diffusion)
    print(f"trained the toy model in {time.time() - start:.1f}s")

    n = args.reference_steps
    ref_ts = uniform_schedule(diffusion, n + 1)
    z_t = jax.random.normal(ref_key, (args.reference_batch, TOY_DIM))
    traj = record_trajectory(eps_fn, z_t, ref_ts, diffusion)[:, : n + 1]

    curvature, taus, result = curvature_density_from_trajectories(
        traj, k=args.window, alpha_target=args.alpha_target
    )
    scores = jax.vmap(alpha_traj)(jax.vmap(normalize_trajectory)(traj), result)
    print(
        f"pruned {float(jnp.mean(result.pruned)):.1%} of {n + 1} points per trajectory "
        f"at median tau {float(jnp.median(taus)):.4f}, "
        f"median alpha_traj {float(jnp.median(scores)):.2e}"
    )

    geo = geosprint_schedule(curvature, diffusion, args.nfe, args.beta)
    uni = uniform_schedule(diffusion, args.nfe)
    print(f"geosprint (beta={args.beta}): {np.asarray(geo)}")
    print(f"uniform ddim:                {np.asarray(uni)}")

    target = two_moons(target_key, args.samples)
    z_t = jax.random.normal(sample_key, (args.samples, TOY_DIM))
    for name, timesteps in (("geosprint", geo), ("uniform", uni)):
        samples = ddim_sample(eps_fn, z_t, timesteps, diffusion)
        print(f"{name:>10}  NFE={args.nfe}  sliced W1 to target = "
              f"{sliced_wasserstein(samples, target, proj_key):.4f}")
    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
