"""Advection-diffusion preset: exact transport of Fourier modes, stationarity, translation invariance."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_da as jd
from jax_da import problems
from jax_da.geometry import Ring, Torus2D
from jax_da.oracles import KalmanOracle


def test_fourier_mode_is_advected_and_damped_exactly():
    # u = cos(k i) evolves to exp(-(kappa k^2 + lambda) dt) cos(k (i - c dt)), including fractional shifts.
    n, m, c, kappa, lam, dt = 64, 3, 0.37, 0.2, 0.05, 1.5
    ssm = problems.advection_diffusion((n,), dt=dt, velocity=c, diffusivity=kappa, damping=lam)
    k, i = 2 * np.pi * m / n, np.arange(n)
    expected = np.exp(-(kappa * k ** 2 + lam) * dt) * np.cos(k * (i - c * dt))
    np.testing.assert_allclose(ssm.mean_transition(jnp.cos(k * i)), expected, atol=1e-12)


def test_pure_advection_by_whole_cells_is_a_roll():
    ssm = problems.advection_diffusion((32,), velocity=3.0, diffusivity=0.0, damping=1e-3)
    x = jax.random.normal(jax.random.PRNGKey(0), (32,))
    np.testing.assert_allclose(ssm.mean_transition(x), np.exp(-1e-3) * jnp.roll(x, 3), atol=1e-12)


def test_two_dimensional_advection_along_each_axis():
    # Odd shifts on even grids exercise the Nyquist modes, whose translation factor is -1.
    ssm = problems.advection_diffusion((8, 6), velocity=(1.0, 3.0), diffusivity=0.0, damping=1e-3, obs_every=2)
    field = jax.random.normal(jax.random.PRNGKey(1), (8, 6))
    moved = ssm.mean_transition(field.ravel()).reshape(8, 6)
    np.testing.assert_allclose(moved, np.exp(-1e-3) * jnp.roll(field, (1, 3), axis=(0, 1)), atol=1e-12)


@pytest.mark.parametrize("shape", [(64,), (8, 8)])
def test_initial_law_is_the_stationary_climatology(shape):
    ssm = problems.advection_diffusion(shape, obs_every=4)
    A, Q, S = (np.asarray(m) for m in (ssm.transition.map.matrix, ssm.transition.noise.cov(), ssm.initial.cov()))
    np.testing.assert_allclose(A @ S @ A.T + Q, S, atol=1e-12)


@pytest.mark.parametrize("shape, shift", [((64,), (5,)), ((8, 8), (3, 2))])
def test_covariances_are_translation_invariant(shape, shift):
    ssm = problems.advection_diffusion(shape, obs_every=4)
    idx = np.arange(np.prod(shape)).reshape(shape)
    perm = np.roll(idx, shift, axis=tuple(range(len(shape)))).ravel()  # a periodic translation
    for cov in (np.asarray(ssm.transition.noise.cov()), np.asarray(ssm.initial.cov())):
        np.testing.assert_allclose(cov[np.ix_(perm, perm)], cov, atol=1e-12)


def test_model_error_marginal_std_and_correlation_shape():
    ssm = problems.advection_diffusion((256,), noise_length=10.0, model_error_std=0.2, nugget=0.01)
    Q = np.asarray(ssm.transition.noise.cov())
    np.testing.assert_allclose(np.diag(Q), 0.04, rtol=1e-10)
    # Away from zero lag the correlation is the squared-exponential kernel scaled by (1 - nugget).
    np.testing.assert_allclose(Q[0, 10] / Q[0, 0], 0.99 * np.exp(-0.5), rtol=0.01)


def test_simulated_truth_stays_in_equilibrium():
    ssm = problems.advection_diffusion((32,), obs_every=4)
    runs = jax.vmap(lambda k: ssm.simulate(k, 30).states[-1])(jax.random.split(jax.random.PRNGKey(2), 4000))
    var = np.asarray(runs).var(0)
    clim = np.diag(np.asarray(ssm.initial.cov()))
    # A variance from 4000 Gaussian draws has relative standard error sqrt(2 / 4000) = 0.022; sites are
    # strongly correlated, so averaging them barely reduces it. Allow four standard errors.
    np.testing.assert_allclose(var.mean() / clim.mean(), 1.0, atol=4 * np.sqrt(2 / 4000))


def test_oracle_posterior_is_tighter_at_observed_points():
    ssm = problems.advection_diffusion((128,), obs_every=16)
    result = KalmanOracle.from_ssm(ssm).filter(ssm.simulate(jax.random.PRNGKey(3), 60).observations)
    sd = np.sqrt(np.diagonal(np.asarray(result.covs[-1])))
    clim = np.sqrt(np.diag(np.asarray(ssm.initial.cov())))
    assert np.all(sd < clim)
    assert sd[::16].mean() < sd[8::16].mean()


def test_geometry_network_and_validation():
    ring, torus = problems.advection_diffusion((40,), obs_every=10), problems.advection_diffusion((6, 4), obs_every=2)
    assert ring.geometry == Ring(40) and ring.observation.map.indices == (0, 10, 20, 30)
    assert torus.geometry == Torus2D(6, 4) and torus.observation.map.indices == (0, 2, 8, 10, 16, 18)
    with pytest.raises(ValueError, match="damping"):
        problems.advection_diffusion((16,), damping=0.0)


def upstream_band(n=64, width=8):
    mask = np.zeros(n, dtype=bool)
    mask[:width] = True
    return mask


def test_forcing_mask_confines_noise_and_keeps_the_initial_law_stationary():
    ssm = problems.advection_diffusion((64,), obs_every=8, forcing_mask=upstream_band())
    draws = np.asarray(ssm.transition.noise.sample(jax.random.PRNGKey(0), (1000,)))
    assert np.all(draws[:, 8:] == 0.0) and np.all(draws[:, :8].std(0) > 0)
    A, Q, S = (np.asarray(m) for m in (ssm.transition.map.matrix, ssm.transition.noise.cov(), ssm.initial.cov()))
    np.testing.assert_allclose(A @ S @ A.T + Q, S, atol=1e-10)


def test_forced_noise_reaches_a_downstream_sensor_after_the_travel_time():
    # Pure advection by one cell per step: noise injected in cells 0..7 reaches cell 30 after 23 steps.
    ssm = problems.advection_diffusion((64,), velocity=1.0, diffusivity=0.0, damping=0.05, obs_every=8,
                                       forcing_mask=upstream_band())
    A, G = np.asarray(ssm.transition.map.matrix), np.asarray(ssm.transition.noise.matrix)
    reach = [float(np.abs((np.linalg.matrix_power(A, k) @ G)[30]).max()) for k in range(30)]
    assert max(reach[:23]) < 1e-12 and reach[23] > 0.1


def test_forcing_mask_model_is_exact_for_the_oracle_and_buildable_in_float32():
    ssm = problems.advection_diffusion((64,), obs_every=8, forcing_mask=upstream_band())
    f = KalmanOracle.from_ssm(ssm).filter(ssm.simulate(jax.random.PRNGKey(1), 20).observations)
    assert bool(jnp.all(jnp.isfinite(f.covs)))
    with jax.experimental.enable_x64(False):
        problems.advection_diffusion((64,), obs_every=8, forcing_mask=upstream_band())
    # The initial law keeps the singular stationary covariance exactly, as a low-rank law.
    assert isinstance(ssm.initial, jd.Embedded) and ssm.initial.matrix.shape[1] < 64
