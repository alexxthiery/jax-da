"""StateSpaceModel: densities, sampling, simulation alignment, pytree behaviour."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_da.conditional import Additive
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.geometry import Ring, Unstructured
from jax_da.laws import Gaussian, PointMass, StudentT
from jax_da.maps import Linear, Selector
from jax_da.ssm import StateSpaceModel

M0 = jnp.array([1.0, -1.0, 0.5, 0.0])
H = jnp.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, -1.0]])


def linear_ssm(model_error=True):
    return StateSpaceModel(
        initial=Gaussian.isotropic(4, 1.0, loc=M0),
        transition=Additive(Linear.damped_rotation(4, decay=0.95), Gaussian.isotropic(4, 0.2) if model_error else None),
        observation=Additive(Linear(H), Gaussian.diagonal(jnp.array([0.3, 0.6]))),
    )


def normal_logpdf(r, std):
    return float(np.sum(-0.5 * (r / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)))


def test_densities_match_hand_gaussian_formulas():
    ssm = linear_ssm()
    x, x_next, y = jnp.array([0.2, 0.1, -0.3, 1.0]), jnp.array([0.0, 0.4, 0.1, 0.9]), jnp.array([0.5, -0.2])
    A = np.asarray(ssm.transition.map.matrix)
    assert float(ssm.log_transition_density(x_next, x)) == pytest.approx(normal_logpdf(x_next - A @ x, 0.2))
    assert float(ssm.log_likelihood(y, x)) == pytest.approx(normal_logpdf(y - H @ x, np.array([0.3, 0.6])))
    assert float(ssm.log_initial_density(x)) == pytest.approx(normal_logpdf(x - M0, 1.0))


def test_sample_initial_moments_and_point_mass():
    ssm = linear_ssm()
    draws = ssm.sample_initial(jax.random.PRNGKey(4), (100_000,))
    np.testing.assert_allclose(draws.mean(0), M0, atol=0.015)
    np.testing.assert_allclose(np.cov(np.asarray(draws).T), np.eye(4), atol=0.02)
    known = ssm.replace(initial=PointMass(M0))
    np.testing.assert_array_equal(known.sample_initial(jax.random.PRNGKey(4), (3,)), jnp.broadcast_to(M0, (3, 4)))
    with pytest.raises(ValueError, match="no density"):
        known.log_initial_density(M0)


def test_sample_transition_moments():
    ssm = linear_ssm()
    x = jnp.array([1.0, 2.0, -1.0, 0.5])
    draws = ssm.sample_transition(jax.random.PRNGKey(0), jnp.broadcast_to(x, (100_000, 4)))
    np.testing.assert_allclose(draws.mean(0), ssm.mean_transition(x), atol=0.005)
    np.testing.assert_allclose(np.cov(np.asarray(draws).T), 0.04 * np.eye(4), atol=0.002)


def test_simulate_alignment_without_noise_is_exact():
    dyn = Lorenz96(dim=8)
    ssm = StateSpaceModel(PointMass(jnp.full(8, 8.0) + 0.01 * jnp.arange(8)), Additive(dyn),
                          Additive(Selector.every(8, 2), Gaussian.isotropic(4, 1e-12)))
    traj = ssm.simulate(jax.random.PRNGKey(0), 20)
    assert traj.states.shape == (20, 8) and traj.observations.shape == (20, 4)
    expected = traj.initial
    for t in range(20):
        expected = dyn(expected)
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


def test_simulate_starts_from_an_explicit_x0():
    ssm = linear_ssm(model_error=False)
    x0 = jnp.array([3.0, -2.0, 1.0, 0.5])
    traj = ssm.simulate(jax.random.PRNGKey(5), 3, x0=x0)
    np.testing.assert_array_equal(traj.initial, x0)
    np.testing.assert_allclose(traj.states[0], ssm.mean_transition(x0), rtol=1e-12)


def test_simulate_is_a_function_of_the_key():
    ssm = linear_ssm()
    a, b = ssm.simulate(jax.random.PRNGKey(3), 30), ssm.simulate(jax.random.PRNGKey(3), 30)
    np.testing.assert_array_equal(a.states, b.states)
    c = jax.jit(lambda m, k: m.simulate(k, 30))(ssm, jax.random.PRNGKey(3))
    np.testing.assert_allclose(c.states, a.states, rtol=1e-12)


def test_deterministic_transition_raises_on_density():
    ssm = linear_ssm(model_error=False)
    with pytest.raises(ValueError, match="no density"):
        ssm.log_transition_density(jnp.zeros(4), jnp.zeros(4))
    np.testing.assert_allclose(ssm.sample_transition(jax.random.PRNGKey(0), jnp.ones(4)), ssm.mean_transition(jnp.ones(4)))


def test_dimension_mismatches_rejected():
    with pytest.raises(ValueError, match="noise.dim"):
        Additive(Selector.every(8, 2), Gaussian.isotropic(3, 1.0))
    with pytest.raises(ValueError, match="initial.dim"):
        StateSpaceModel(Gaussian.isotropic(7, 1.0), Additive(Lorenz96(dim=8)), Additive(Selector.every(8, 2), Gaussian.isotropic(4, 1.0)))
    with pytest.raises(ValueError, match="observation.in_dim"):
        StateSpaceModel(Gaussian.isotropic(8, 1.0), Additive(Lorenz96(dim=8)), Additive(Selector.every(9, 2), Gaussian.isotropic(5, 1.0)))
    with pytest.raises(ValueError, match="geometry.dim"):
        linear_ssm().replace(geometry=Ring(5))


def test_geometry_defaults_to_unstructured_and_is_kept():
    assert linear_ssm().geometry == Unstructured(4)
    ssm = linear_ssm().replace(geometry=Ring(4))
    assert jax.jit(lambda m: m)(ssm).geometry == Ring(4)


def test_model_survives_pytree_rebuilds_and_batching():
    # Rebuilds with non-numeric leaves (tuples from tree.map(jnp.shape)) must not trip validation.
    # Integer leaves are validated like real parameters on purpose: StudentT(3, df=-1) must raise.
    ssm = linear_ssm()
    shapes = jax.tree.map(jnp.shape, ssm)
    assert shapes.geometry == ssm.geometry and shapes.initial.loc == (4,)
    # A model returned by vmap has a batch axis on every leaf; dimensions come from trailing axes.
    stds = jnp.array([[0.3, 0.6], [1.0, 2.0]])
    batched = jax.vmap(lambda s: ssm.replace(observation=Additive(Linear(H), Gaussian.diagonal(s))))(stds)
    assert batched.state_dim == 4 and batched.obs_dim == 2
    loglik = jax.vmap(lambda m: m.log_likelihood(jnp.zeros(2), jnp.zeros(4)))(batched)
    expected = [normal_logpdf(np.zeros(2), np.asarray(s)) for s in stds]
    np.testing.assert_allclose(loglik, expected, rtol=1e-10)


def test_gradients_and_vmap_through_parameters():
    def loglik(forcing):
        ssm = StateSpaceModel(Gaussian.isotropic(8, 1.0), Additive(Lorenz96(dim=8, forcing=forcing), Gaussian.isotropic(8, 0.1)),
                              Additive(Selector.every(8, 2), StudentT(4, df=4.0, scale=0.5)))
        x = jnp.linspace(-1, 1, 8)
        return ssm.log_transition_density(ssm.mean_transition(x) + 0.05, x)

    g = jax.grad(loglik)(8.0)
    eps = 1e-5
    assert float(g) == pytest.approx(float((loglik(8.0 + eps) - loglik(8.0 - eps)) / (2 * eps)), rel=1e-5)
    assert jax.vmap(loglik)(jnp.array([7.0, 8.0, 9.0])).shape == (3,)
