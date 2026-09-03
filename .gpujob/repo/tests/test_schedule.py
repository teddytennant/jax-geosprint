"""Algorithm 1: densities, SmoothAndFloor, the CDF and the quantile placement."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from geosprint import (
    blend_densities,
    curvature_on_training_grid,
    density_cdf,
    geosprint_schedule,
    linear_beta_schedule,
    log_snr,
    logsnr_density,
    quantile_timesteps,
    retention_frequency,
    schedule_from_density,
    smooth_and_floor,
    uniform_schedule,
)

T = 1000


@pytest.fixture(scope="module")
def diffusion():
    return linear_beta_schedule(T)


@pytest.fixture(scope="module")
def curvature():
    """A bimodal retention profile, the shape Figure 3(a) reports."""
    idx = jnp.arange(201, dtype=jnp.float32)
    w = jnp.exp(-((idx - 5) ** 2) / 50.0) + 0.8 * jnp.exp(-((idx - 190) ** 2) / 200.0)
    return smooth_and_floor(0.05 + 0.9 * w)


def test_retention_frequency_averages_indicators():
    """Eq. 6 is a mean of indicators over the reference batch."""
    retained = jnp.asarray([[True, False, True], [True, True, False], [False, False, True]])
    got = np.asarray(retention_frequency(retained))
    assert got == pytest.approx([2 / 3, 1 / 3, 2 / 3])


def test_smooth_and_floor_respects_its_floor():
    """Nothing survives below ``eps_floor = 0.05 max_t w_tilde(t)``."""
    w = jnp.zeros((200,)).at[100].set(1.0)
    out = smooth_and_floor(w, sigma=5.0, eps_floor_frac=0.05)
    floor = 0.05 * float(jnp.max(out))
    assert float(jnp.min(out)) >= floor - 1e-7
    assert float(jnp.min(out)) == pytest.approx(floor, rel=1e-5)


def test_smooth_and_floor_actually_smooths():
    """A single spike is spread over roughly the kernel bandwidth."""
    w = jnp.zeros((200,)).at[100].set(1.0)
    out = np.asarray(smooth_and_floor(w, sigma=5.0, eps_floor_frac=0.0))

    assert out[100] < 1.0 and out[100] > out[105] > out[115]
    above_half = np.flatnonzero(out > 0.5 * out.max())
    width = above_half[-1] - above_half[0]
    assert 8 <= width <= 16  # a sigma=5 Gaussian has FWHM about 11.8

    # A flat input comes back flat, kernel edge effects and all.
    flat = np.asarray(smooth_and_floor(jnp.ones((200,)), sigma=5.0, eps_floor_frac=0.0))
    assert np.allclose(flat, 1.0, atol=1e-5)


def test_smooth_and_floor_is_never_below_the_floor_on_real_shapes(curvature):
    floor = 0.05 * float(jnp.max(curvature))
    assert float(jnp.min(curvature)) >= floor - 1e-7


def test_logsnr_density_gives_uniform_log_snr_spacing(diffusion):
    """``rho_logSNR ~ |d lambda / dt|``, so its quantiles are uniform in lambda.

    This is the check that the change of variables is right. If the density
    were, say, uniform in ``t``, these gaps would not be equal.

    Measured on the continuous quantile positions, since rounding to integer
    timesteps is what limits this at large ``K``: the linear beta schedule
    piles up so much log-SNR in the last handful of timesteps that a 100-step
    budget wants positions the integer grid cannot separate. The 3 percent
    tolerance is the trapezoidal quadrature error in the first step or two,
    where ``lambda`` is steepest; uniform-t spacing misses by more than 50
    percent on the same measure, which is the next test.
    """
    lam = np.asarray(log_snr(diffusion), dtype=np.float64)
    grid = np.arange(T, dtype=np.float64)
    cdf = density_cdf(logsnr_density(diffusion))

    for k in (10, 25, 50, 100):
        pos = np.asarray(quantile_timesteps(cdf, k), dtype=np.float64)
        gaps = np.abs(np.diff(np.interp(pos, grid, lam)))
        assert np.std(gaps) / np.mean(gaps) < 0.03

    # After rounding, moderate budgets still come out close to uniform.
    for k in (10, 25):
        gaps = np.diff(lam[np.asarray(schedule_from_density(logsnr_density(diffusion), k))])
        assert np.std(gaps) / np.abs(np.mean(gaps)) < 0.05


def test_uniform_t_spacing_is_not_uniform_in_log_snr(diffusion):
    """The contrast the previous test needs to be worth anything."""
    lam = np.asarray(log_snr(diffusion))
    gaps = np.diff(lam[np.asarray(uniform_schedule(diffusion, 25))])
    assert np.std(gaps) / np.abs(np.mean(gaps)) > 0.5


def test_blend_weights_sum_to_one_and_interpolate():
    a = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    b = jnp.asarray([4.0, 3.0, 2.0, 1.0])
    for beta in (0.0, 0.25, 0.6, 1.0):
        out = blend_densities(a, b, beta)
        assert float(jnp.sum(out)) == pytest.approx(1.0, rel=1e-5)
        want = (1 - beta) * np.asarray(a) / 10.0 + beta * np.asarray(b) / 10.0
        assert np.asarray(out) == pytest.approx(want, rel=1e-5)


def test_cdf_is_nondecreasing_and_spans_the_unit_interval(diffusion, curvature):
    for beta in (0.0, 0.6, 1.0):
        rho = blend_densities(
            logsnr_density(diffusion), curvature_on_training_grid(curvature, diffusion), beta
        )
        cdf = np.asarray(density_cdf(rho))
        assert cdf[0] == pytest.approx(0.0, abs=1e-7)
        assert cdf[-1] == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.diff(cdf) >= 0)
        assert np.all(np.diff(cdf) > 0)  # the floor keeps rho strictly positive


def test_quantile_mapping_inverts_the_cdf(diffusion, curvature):
    """``F(F^{-1}(q)) = q`` at the quantiles the schedule is placed at."""
    rho = blend_densities(
        logsnr_density(diffusion), curvature_on_training_grid(curvature, diffusion), 0.6
    )
    cdf = density_cdf(rho)
    k = 20
    pos = np.asarray(quantile_timesteps(cdf, k))

    assert np.all(np.diff(pos) > 0)
    assert pos[0] == pytest.approx(0.0, abs=1e-5)
    assert pos[-1] == pytest.approx(float(rho.shape[0] - 1), abs=1e-5)

    grid = np.arange(rho.shape[0], dtype=np.float64)
    back = np.interp(pos, grid, np.asarray(cdf, dtype=np.float64))
    assert back == pytest.approx(np.linspace(0.0, 1.0, k), abs=1e-5)


def test_a_uniform_density_places_uniformly_spaced_steps():
    """The quantile machinery has no bias of its own to hide behind."""
    ts = np.asarray(schedule_from_density(jnp.ones((101,)), 11))
    assert np.array_equal(ts, np.arange(100, -1, -10))


def test_schedule_shape_monotonicity_and_range(diffusion, curvature):
    for k in (5, 10, 20, 50, 100):
        for beta in (0.0, 0.3, 0.6, 1.0):
            ts = np.asarray(geosprint_schedule(curvature, diffusion, k, beta))
            assert ts.shape == (k,)
            assert np.all(np.diff(ts) < 0), f"not strictly decreasing at K={k}, beta={beta}"
            assert ts.min() >= 0 and ts.max() <= T - 1
            assert ts.dtype == np.int32


def test_beta_zero_and_one_recover_the_pure_schedules(diffusion, curvature):
    """Eq. 7's endpoints, exactly, not approximately.

    Section 3.2: "Setting beta=0 recovers log-SNR-uniform spacing. Setting
    beta=1 gives pure curvature-weighted spacing."
    """
    on_grid = curvature_on_training_grid(curvature, diffusion)
    for k in (8, 20, 40):
        pure_logsnr = schedule_from_density(logsnr_density(diffusion), k)
        pure_curv = schedule_from_density(on_grid, k)

        assert np.array_equal(
            np.asarray(geosprint_schedule(curvature, diffusion, k, 0.0)),
            np.asarray(pure_logsnr),
        )
        assert np.array_equal(
            np.asarray(geosprint_schedule(curvature, diffusion, k, 1.0)),
            np.asarray(pure_curv),
        )
        # and the blend is not silently equal to either endpoint
        blended = np.asarray(geosprint_schedule(curvature, diffusion, k, 0.6))
        assert not np.array_equal(blended, np.asarray(pure_logsnr))
        assert not np.array_equal(blended, np.asarray(pure_curv))


def test_blend_moves_steps_toward_the_curvature_peaks(diffusion, curvature):
    """Raising beta pulls steps into the region the retention profile favors.

    The fixture profile peaks near index 190 of 200, which maps to low ``t``.
    """
    late = []
    for beta in (0.0, 0.3, 0.6, 1.0):
        ts = np.asarray(geosprint_schedule(curvature, diffusion, 40, beta))
        late.append(int(np.sum(ts > 0.8 * T)))
    assert late == sorted(late)
    assert late[-1] > late[0]


def test_uniform_schedule_is_the_ddim_baseline(diffusion):
    ts = np.asarray(uniform_schedule(diffusion, 20))
    assert ts.shape == (20,)
    assert np.all(np.diff(ts) < 0)
    assert ts[0] == T - 1 and ts[-1] == 0
    assert np.std(np.diff(ts)) < 1.0


def test_curvature_grid_mapping_honors_reference_timesteps(diffusion):
    """Algorithm 1 line 8. Index ``i`` lands at ``ref_timesteps[i]``.

    Fed a spike at the index whose reference timestep is 300, the density on
    the training grid has to peak at 300 and not at the position a naive
    linear index map would put it.
    """
    ref = uniform_schedule(diffusion, 101)
    spike_index = int(np.flatnonzero(np.asarray(ref) == 300)[0])
    w = jnp.zeros((101,)).at[spike_index].set(1.0)

    mapped = np.asarray(curvature_on_training_grid(w, diffusion, ref))
    assert int(np.argmax(mapped)) == 300


def test_schedule_rejects_budgets_larger_than_the_grid(diffusion):
    with pytest.raises(ValueError):
        schedule_from_density(jnp.ones((10,)), 11)


def test_default_index_map_matches_a_uniform_reference_pass(diffusion):
    """The default assumes a reference trajectory ordered from t=T down to t=0.

    Getting that direction backwards mirrors the curvature profile, which is
    only visible downstream as a schedule that spends its steps in the wrong
    half of the trajectory.
    """
    ref = uniform_schedule(diffusion, 201)
    w = jnp.linspace(0.0, 1.0, 201)

    explicit = np.asarray(curvature_on_training_grid(w, diffusion, ref))
    default = np.asarray(curvature_on_training_grid(w, diffusion))

    assert explicit == pytest.approx(default, abs=1e-3)
    assert default[0] > default[-1]  # index 0 is t=T, where w is smallest
