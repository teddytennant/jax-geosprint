"""Seed sweep for test_blend_beats_pure_curvature_spacing_at_every_budget.

Repeats the full pipeline (train toy model, reference pass, curvature density,
blend vs pure schedule, SWD to target) at N independent seeds and records the
per-budget margin (pure - blended). Positive margin means the test's claim
holds; negative means it flips.

Usage: python sweep.py <n_seeds> <out.json> [seed_offset]
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import geosprint as g
import geosprint.toy as toy

TRAIN_STEPS = 6000
REFERENCE_BATCH = 64
REFERENCE_N = 200
N_SAMPLES = 4000
BUDGETS = (10, 20, 30, 40)


def run_one(seed_idx: int):
    key = jax.random.PRNGKey(seed_idx)
    k_train, k_ref, k_target, k_eval = jax.random.split(key, 4)

    params, diffusion = toy.train(k_train, steps=TRAIN_STEPS)
    eps_fn = toy.make_eps_fn(params, diffusion)

    ref_ts = g.uniform_schedule(diffusion, REFERENCE_N + 1)
    z_t = jax.random.normal(k_ref, (REFERENCE_BATCH, toy.TOY_DIM))
    traj = g.record_trajectory(eps_fn, z_t, ref_ts, diffusion)[:, : REFERENCE_N + 1]
    curvature, taus, result = g.curvature_density_from_trajectories(traj, k=2, alpha_target=1e-3)

    target = toy.two_moons(k_target, N_SAMPLES)
    eval_keys = jax.random.split(k_eval, 3)

    def swd(timesteps):
        scores = []
        for ek in eval_keys:
            z0 = jax.random.normal(ek, (N_SAMPLES, toy.TOY_DIM))
            samples = g.ddim_sample(eps_fn, z0, timesteps, diffusion)
            scores.append(toy.sliced_wasserstein(samples, target, jax.random.PRNGKey(7)))
        return float(np.mean(scores))

    out = {}
    for k in BUDGETS:
        blended = swd(g.geosprint_schedule(curvature, diffusion, k, 0.6))
        pure = swd(g.geosprint_schedule(curvature, diffusion, k, 1.0))
        out[k] = {"blended": blended, "pure": pure, "margin": pure - blended}
    return out


def main():
    n_seeds = int(sys.argv[1])
    out_path = sys.argv[2]
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    print(f"jax {jax.__version__}, devices: {jax.devices()}", flush=True)

    results = {}
    t0 = time.time()
    for i in range(offset, offset + n_seeds):
        r = run_one(i)
        results[i] = r
        elapsed = time.time() - t0
        print(f"seed {i}: " + " ".join(f"K={k} margin={v['margin']:+.5f}" for k, v in r.items())
              + f"  ({elapsed:.1f}s elapsed)", flush=True)

    with open(out_path, "w") as f:
        json.dump({"jax_version": jax.__version__, "devices": [str(d) for d in jax.devices()],
                    "results": results}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
