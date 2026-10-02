"""StateSpaceModel: densities, sampling, simulation alignment, transformations."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_da.dynamics.linear import LinearDynamics
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.noise import Gaussian, StudentT
from jax_da.observations import Linear, Selector
from jax_da.ssm import StateSpaceModel


def linear_ssm(model_error=True):
    dyn = LinearDynamics.damped_rotation(4, decay=0.95)
    return StateSpaceModel(
        dynamics=dyn,
        obs_operator=Linear(jnp.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, -1.0]])),
        obs_noise=Gaussian.diagonal(jnp.array([0.3, 0.6])),
        initial_mean=jnp.array([1.0, -1.0, 0.5, 0.0]),
        initial_noise=Gaussian.isotropic(4, 1.0),
        model_error=Gaussian.isotropic(4, 0.2) if model_error else None,
    )


def normal_logpdf(r, std):
    return float(np.sum(-0.5 * (r / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))


def test_densities_match_hand_gaussian_formulas():
    ssm = linear_ssm()
    x, x_next, y = jnp.array([0.2, 0.1, -0.3, 1.0]), jnp.array([0.0, 0.4, 0.1, 0.9]), jnp.array([0.5, -0.2])
    A, H = np.asarray(ssm.dynamics.matrix), np.asarray(ssm.obs_operator.matrix)
    assert float(ssm.log_transition_density(x_next, x)) == pytest.approx(normal_logpdf(x_next - A @ x, 0.2))
    assert float(ssm.log_likelihood(y, x)) == pytest.approx(normal_logpdf(y - H @ x, np.array([0.3, 0.6])))
    assert float(ssm.log_initial_density(x)) == pytest.approx(normal_logpdf(x - ssm.initial_mean, 1.0))


def test_sample_initial_moments_and_point_mass():
    ssm = linear_ssm()
    draws = ssm.sample_initial(jax.random.PRNGKey(4), (100_000,))
    np.testing.assert_allclose(draws.mean(0), ssm.initial_mean, atol=0.015)
    np.testing.assert_allclose(np.cov(np.asarray(draws).T), np.eye(4), atol=0.02)
    known = ssm.replace(initial_noise=None)
    np.testing.assert_array_equal(known.sample_initial(jax.random.PRNGKey(4), (3,)),
                                  jnp.broadcast_to(ssm.initial_mean, (3, 4)))


def test_sample_transition_moments():
    ssm = linear_ssm()
    x = jnp.array([1.0, 2.0, -1.0, 0.5])
    draws = ssm.sample_transition(jax.random.PRNGKey(0), jnp.broadcast_to(x, (100_000, 4)))
    np.testing.assert_allclose(draws.mean(0), ssm.mean_transition(x), atol=0.005)
    np.testing.assert_allclose(np.cov(np.asarray(draws).T), 0.04 * np.eye(4), atol=0.002)


def test_simulate_alignment_without_noise_is_exact():
    dyn = Lorenz96(dim=8)
    ssm = StateSpaceModel(dyn, Selector.every(8, 2), Gaussian.isotropic(4, 1e-12), jnp.full(8, 8.0) + 0.01 * jnp.arange(8))
    traj = ssm.simulate(jax.random.PRNGKey(0), 20)
    assert traj.states.shape == (20, 8) and traj.observations.shape == (20, 4)
    expected = traj.initial
    for t in range(20):
        expected = dyn.flow(expected)
        np.testing.assert_allclose(traj.states[t], expected, rtol=1e-10)
    np.testing.assert_allclose(traj.observations, traj.states[:, ::2], atol=1e-10)


def test_simulate_noise_residuals_have_the_noise_law():
    ssm = linear_ssm()
    traj = ssm.simulate(jax.random.PRNGKey(1), 50_000)
    eta = traj.states[1:] - ssm.mean_transition(traj.states[:-1])
    eps = traj.observations - ssm.observe_mean(traj.states)
    np.testing.assert_allclose(eta.std(0), 0.2, rtol=0.02)
    np.testing.assert_allclose(eps.std(0), [0.3, 0.6], rtol=0.02)
    np.testing.assert_allclose(np.corrcoef(np.asarray(eps[1:, 0]), np.asarray(eps[:-1, 0]))[0, 1], 0.0, atol=0.02)


def test_simulate_is_a_function_of_the_key():
    ssm = linear_ssm()
    a, b = ssm.simulate(jax.random.PRNGKey(3), 30), ssm.simulate(jax.random.PRNGKey(3), 30)
    np.testing.assert_array_equal(a.states, b.states)
    c = jax.jit(lambda m, k: m.simulate(k, 30))(ssm, jax.random.PRNGKey(3))
    np.testing.assert_allclose(c.states, a.states, rtol=1e-12)


def test_point_mass_laws_raise_on_density():
    ssm = linear_ssm(model_error=False)
    with pytest.raises(ValueError, match="no transition density"):
        ssm.log_transition_density(jnp.zeros(4), jnp.zeros(4))
    np.testing.assert_allclose(ssm.sample_transition(jax.random.PRNGKey(0), jnp.ones(4)), ssm.mean_transition(jnp.ones(4)))


def test_dimension_mismatches_rejected():
    with pytest.raises(ValueError, match="obs_noise.dim"):
        StateSpaceModel(Lorenz96(dim=8), Selector.every(8, 2), Gaussian.isotropic(3, 1.0), jnp.zeros(8))
    with pytest.raises(ValueError, match="model_error.dim"):
        StateSpaceModel(Lorenz96(dim=8), Selector.every(8, 2), Gaussian.isotropic(4, 1.0), jnp.zeros(8),
                        model_error=Gaussian.isotropic(7, 1.0))


def test_gradients_and_vmap_through_parameters():
    def loglik(forcing):
        ssm = StateSpaceModel(Lorenz96(dim=8, forcing=forcing), Selector.every(8, 2),
                              StudentT(4, df=4.0, scale=0.5), jnp.zeros(8), model_error=Gaussian.isotropic(8, 0.1))
        x = jnp.linspace(-1, 1, 8)
        return ssm.log_transition_density(ssm.mean_transition(x) + 0.05, x)

    g = jax.grad(loglik)(8.0)
    eps = 1e-5
    assert float(g) == pytest.approx(float((loglik(8.0 + eps) - loglik(8.0 - eps)) / (2 * eps)), rel=1e-5)
    assert jax.vmap(loglik)(jnp.array([7.0, 8.0, 9.0])).shape == (3,)
