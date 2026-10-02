"""Presets: deterministic construction, finite simulation, documented networks."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_da import problems
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.oracles import KalmanOracle

PRESETS = {
    "linear_gaussian": lambda: problems.linear_gaussian(),
    "lorenz63": lambda: problems.lorenz63(),
    "lorenz96": lambda: problems.lorenz96(),
    "lorenz96_sparse": lambda: problems.lorenz96(obs_every=4),
    "lorenz96_two_scale": lambda: problems.lorenz96_two_scale(n_fast=8, n_spinup=300),
}


@pytest.mark.parametrize("name", PRESETS)
def test_preset_is_deterministic_and_simulates_finite_data(name):
    a, b = PRESETS[name](), PRESETS[name]()
    np.testing.assert_array_equal(a.initial.loc, b.initial.loc)
    traj = a.simulate(jax.random.PRNGKey(0), 25)
    assert traj.states.shape == (25, a.state_dim) and traj.observations.shape == (25, a.obs_dim)
    assert bool(jnp.all(jnp.isfinite(traj.states)))


def test_lorenz96_network_and_attractor_scale():
    ssm = problems.lorenz96(obs_every=4)
    assert ssm.observation.map.indices == tuple(range(0, 40, 4))
    traj = ssm.simulate(jax.random.PRNGKey(1), 2000)
    # Climatological mean and std of L96 with F = 8 are about 2.3 and 3.6.
    assert 1.8 < float(traj.states.mean()) < 2.8 and 3.2 < float(traj.states.std()) < 4.0


def test_two_scale_truth_pairs_with_single_scale_forecast_model():
    truth = problems.lorenz96_two_scale(n_fast=8, n_spinup=300)
    forecast = Lorenz96(dim=8, forcing=20.0)
    x = truth.simulate(jax.random.PRNGKey(2), 5).states
    assert forecast(x[:, :8]).shape == (5, 8)
    assert truth.observation.map.indices == tuple(range(8))


def test_linear_gaussian_preset_is_oracle_compatible():
    ssm = problems.linear_gaussian()
    result = KalmanOracle.from_ssm(ssm).filter(ssm.simulate(jax.random.PRNGKey(3), 10).observations)
    assert bool(jnp.all(jnp.isfinite(result.means)))


@pytest.mark.pde
def test_pde_presets_build_and_simulate():
    pytest.importorskip("exponax")
    for ssm, n_obs in ((problems.kuramoto_sivashinsky(n_spinup=50), 16),
                       (problems.kolmogorov(resolution=16, obs_per_side=4, n_spinup=20), 16)):
        assert ssm.obs_dim == n_obs
        traj = ssm.simulate(jax.random.PRNGKey(4), 5)
        assert bool(jnp.all(jnp.isfinite(traj.states)))


def test_linear_gaussian_full_has_documented_structure():
    ssm = problems.linear_gaussian_full(state_dim=6, obs_dim=3, spectral_radius=0.9, model_error_std=0.3,
                                        obs_std=0.5, initial_std=1.0, seed=7)
    A, H = np.asarray(ssm.transition.map.matrix), np.asarray(ssm.observation.map.matrix)
    assert H.shape == (3, 6) and np.count_nonzero(H) == H.size
    assert np.max(np.abs(np.linalg.eigvals(A))) == pytest.approx(0.9)
    for law, std in ((ssm.transition.noise, 0.3), (ssm.observation.noise, 0.5), (ssm.initial, 1.0)):
        cov = np.asarray(law.cov())
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.linalg.eigvalsh(cov).min() > 0
        assert np.mean(np.diag(cov)) == pytest.approx(std ** 2)
        off = cov - np.diag(np.diag(cov))
        assert np.abs(off).max() > 0.05 * std ** 2  # genuinely correlated, not diagonal


def test_linear_gaussian_full_is_seeded_and_oracle_ready():
    a, b, c = (problems.linear_gaussian_full(seed=s) for s in (3, 3, 4))
    np.testing.assert_array_equal(a.transition.map.matrix, b.transition.map.matrix)
    assert not np.allclose(a.transition.map.matrix, c.transition.map.matrix)
    traj = a.simulate(jax.random.PRNGKey(0), 20)
    result = KalmanOracle.from_ssm(a).filter(traj.observations)
    assert bool(jnp.all(jnp.isfinite(result.means))) and bool(jnp.isfinite(result.log_evidence))


def test_simulated_noise_has_the_full_covariances():
    ssm = problems.linear_gaussian_full(seed=1)
    traj = ssm.simulate(jax.random.PRNGKey(2), 100_000)
    eta = np.asarray(traj.states[1:] - ssm.mean_transition(traj.states[:-1]))
    eps = np.asarray(traj.observations - ssm.observe_mean(traj.states))
    np.testing.assert_allclose(np.cov(eta.T), ssm.transition.noise.cov(), atol=0.004)
    np.testing.assert_allclose(np.cov(eps.T), ssm.observation.noise.cov(), atol=0.01)


def test_stochastic_volatility_is_stationary_with_the_documented_observation_law():
    phi, sigma, beta = 0.95, 0.3, 0.7
    ssm = problems.stochastic_volatility(dim=2, phi=phi, sigma=sigma, beta=beta, mu=-0.5)
    stationary_var = sigma ** 2 / (1 - phi ** 2)
    np.testing.assert_allclose(ssm.initial.variance(), stationary_var)
    runs = jax.vmap(lambda k: ssm.simulate(k, 40))(jax.random.split(jax.random.PRNGKey(5), 20_000))
    final = np.asarray(runs.states[:, -1])
    # Variance from 20000 draws has relative standard error sqrt(2 / 20000) = 0.01; allow four.
    np.testing.assert_allclose(final.var(0), stationary_var, rtol=0.04)
    np.testing.assert_allclose(final.mean(0), -0.5, atol=4 * np.sqrt(stationary_var / 20_000))
    eps = np.asarray(runs.observations / (beta * jnp.exp(runs.states / 2)))
    np.testing.assert_allclose(eps.std(), 1.0, rtol=0.01)


def test_nonlinear_poisson_matches_the_documented_construction():
    ssm = problems.nonlinear_poisson(state_dim=4, obs_dim=32, seed=3)
    B, C = np.asarray(ssm.transition.map.B), np.asarray(ssm.observation.log_rate.matrix)
    assert np.linalg.norm(B, 2) <= 1.2 + 1e-12 and C.shape == (32, 4)
    np.testing.assert_allclose(ssm.observation.log_rate.offset, -0.2 + 0.1 * np.cos(np.arange(32)))
    x = jnp.array([0.3, -0.1, 0.5, 0.2])
    expected = 0.55 * x + 0.55 * (jnp.tanh(B @ x) ** 2 - 0.25)
    np.testing.assert_allclose(ssm.mean_transition(x), expected, rtol=1e-12)
    traj = ssm.simulate(jax.random.PRNGKey(6), 50)
    y = np.asarray(traj.observations)
    assert np.all(y >= 0) and np.all(y == np.round(y))
    np.testing.assert_array_equal(problems.nonlinear_poisson(seed=3).transition.map.B, ssm.transition.map.B)


def test_nonlinear_poisson_parameters_are_differentiable():
    ssm = problems.nonlinear_poisson()
    traj = ssm.simulate(jax.random.PRNGKey(7), 20)

    def loglik(rho):
        model = ssm.replace(transition=ssm.transition.replace(map=ssm.transition.map.replace(rho=rho)))
        return model.log_transition_density(traj.states[1:], traj.states[:-1]).sum()

    g = jax.grad(loglik)(0.55)
    eps = 1e-5
    assert float(g) == pytest.approx(float((loglik(0.55 + eps) - loglik(0.55 - eps)) / (2 * eps)), rel=1e-5)
