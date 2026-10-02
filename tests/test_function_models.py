"""Models built from plain JAX functions with the ``Function`` map."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_da as jd

D = 4
A = jnp.array([[0.9, 0.1, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.1, 0.0, 0.7, 0.1], [0.0, 0.0, 0.3, 0.6]])


def mixing(x):
    """Written for one state; reversing components is wrong if applied to an ensemble axis."""
    return x + 0.1 * jnp.sin(x[::-1])


def test_ensemble_call_equals_member_by_member():
    dyn = jd.Function(mixing, D, D)
    ens = jax.random.normal(jax.random.PRNGKey(0), (2, 5, D))
    np.testing.assert_allclose(dyn(ens), jax.vmap(jax.vmap(mixing))(ens), rtol=1e-12)
    op = jd.Function(lambda x: jnp.array([x[0] * x[3], x[1:].sum()]), D, 2)
    np.testing.assert_allclose(op(ens), jax.vmap(jax.vmap(op.fn))(ens), rtol=1e-12)


def test_function_maps_reproduce_the_built_in_linear_model():
    linear = jd.StateSpaceModel(jd.Gaussian.isotropic(D, 1.0, loc=1.0),
                                jd.Additive(jd.Linear(A), jd.Gaussian.isotropic(D, 0.2)),
                                jd.Additive(jd.Linear(A[:2]), jd.Gaussian.isotropic(2, 0.3)))
    wrapped = linear.replace(transition=jd.Additive(jd.Function(lambda x: A @ x, D, D), jd.Gaussian.isotropic(D, 0.2)),
                             observation=jd.Additive(jd.Function(lambda x: A[:2] @ x, D, 2), jd.Gaussian.isotropic(2, 0.3)))
    a, b = linear.simulate(jax.random.PRNGKey(1), 30), wrapped.simulate(jax.random.PRNGKey(1), 30)
    np.testing.assert_allclose(b.states, a.states, rtol=1e-12)
    np.testing.assert_allclose(b.observations, a.observations, rtol=1e-12)
    x, y = a.states[:-1], a.observations[1:]
    np.testing.assert_allclose(wrapped.log_transition_density(a.states[1:], x), linear.log_transition_density(a.states[1:], x))
    np.testing.assert_allclose(wrapped.log_likelihood(y, a.states[1:]), linear.log_likelihood(y, a.states[1:]))


def test_model_passes_through_jit_as_an_argument():
    ssm = jd.StateSpaceModel(jd.Gaussian.isotropic(D, 1.0), jd.Additive(jd.Function(mixing, D, D), jd.Gaussian.isotropic(D, 0.1)),
                             jd.Additive(jd.Function(jnp.tanh, D, D), jd.Gaussian.isotropic(D, 0.1)))
    run = jax.jit(lambda m, k: m.simulate(k, 10).states)
    np.testing.assert_allclose(run(ssm, jax.random.PRNGKey(2)), ssm.simulate(jax.random.PRNGKey(2), 10).states, rtol=1e-12)


def test_jacobian_of_function_map_is_exact():
    op = jd.Function(lambda x: jnp.stack([x[0] ** 2, x[1] * x[2]]), 3, 2)
    np.testing.assert_allclose(jax.jacfwd(op)(jnp.array([1.5, -2.0, 0.5])), [[3.0, 0.0, 0.0], [0.0, 0.5, -2.0]])


def test_wrong_output_shape_rejected():
    with pytest.raises(ValueError, match=r"fn\(x\)"):
        jd.Function(lambda x: x[:2], D, 3)(jnp.zeros(D))
