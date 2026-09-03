"""Feed a fixed upstream.npz (trained params, curvature, target, eval noise,
all pinned as plain numpy from a single CPU run) through the downstream
schedule + sampling + SWD stage on whichever backend this process runs on.

If CPU and GPU runs of this script agree to float32 epsilon, the divergence
in the full pipeline is not a backend bug in the downstream stage -- it has
to come from the training stage (chaotic sensitivity of 6000 SGD steps to
tiny fp differences), or upstream of that.

Optionally set FORCE_HIGHEST=1 to force jax_default_matmul_precision=highest
before touching jax (disables TF32 on GPU) as a diagnostic.
"""
import json
import os
import sys

if os.environ.get("FORCE_HIGHEST") == "1":
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")

import jax
import jax.numpy as jnp
import numpy as np

import geosprint as g
import geosprint.toy as toy
from geosprint.diffusion import Schedule

BUDGETS = (10, 20, 30, 40)


def main():
    npz_path = sys.argv[1]
    out_path = sys.argv[2]

    print(f"jax {jax.__version__}, devices: {jax.devices()}, "
          f"matmul_precision={jax.config.jax_default_matmul_precision}")

    d = np.load(npz_path)
    n_layers = int(d["n_layers"])
    params = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(n_layers)]
    diffusion = Schedule(betas=jnp.asarray(d["betas"]), alphas_bar=jnp.asarray(d["alphas_bar"]))
    curvature = jnp.asarray(d["curvature"])
    target = jnp.asarray(d["target"])
    eval_z = jnp.asarray(d["eval_z"])  # (3, N, 2)

    eps_fn = toy.make_eps_fn(params, diffusion)

    def swd(timesteps):
        scores = []
        for i in range(eval_z.shape[0]):
            samples = g.ddim_sample(eps_fn, eval_z[i], timesteps, diffusion)
            scores.append(toy.sliced_wasserstein(samples, target, jax.random.PRNGKey(7)))
        return float(np.mean(scores))

    out = {}
    for k in BUDGETS:
        blended = swd(g.geosprint_schedule(curvature, diffusion, k, 0.6))
        pure = swd(g.geosprint_schedule(curvature, diffusion, k, 1.0))
        out[str(k)] = {"blended": repr(blended), "pure": repr(pure), "margin": repr(pure - blended)}
        print(f"K={k}: blended={blended!r} pure={pure!r} margin={pure - blended:+.10e}")

    with open(out_path, "w") as f:
        json.dump({"jax_version": jax.__version__, "devices": [str(dv) for dv in jax.devices()],
                    "matmul_precision": str(jax.config.jax_default_matmul_precision),
                    "results": out}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
