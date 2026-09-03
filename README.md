# jax-geosprint

JAX implementation of "GeoSPRINT: Geometric Redundancy-Aware Step Pruning for
Inference in Diffusion Trajectories" (arXiv:2609.02160).

## install

uv pip install -e .

## run

python -m geosprint.demo --nfe 20

Everything here runs on a 2D two-moons diffusion model that trains in a couple of
seconds on CPU. The CIFAR-10, LSUN Church and Stable Diffusion v1.5 FID numbers are
not reproduced, and no pretrained model is downloaded. At toy scale the schedule
beats uniform DDIM at 20 and 30 NFE and loses at 10 and 40, so the paper's
consistent win does not appear; the end-to-end tests assert what does hold and say
which numbers they measured. Three things the paper leaves open, and what I picked:
rho_logSNR is |dlambda/dt| for the variance-preserving log-SNR, since the formula is
never written out; the K quantiles are linspace(0, 1, K), so both ends of the range
are always in the schedule; and both densities are normalized before Eq. 7 blends
them, which is what makes beta a mixing weight. Equation and algorithm line numbers
are in the docstrings.
