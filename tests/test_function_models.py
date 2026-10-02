"""Function-defined dynamics and observation operators."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_da

D = 4
A = jnp.array([[0.9, 0.1, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.1, 0.0, 0.7, 0.1], [0.0, 0.0, 0.3, 0.6]])


def mixing(x):
    """Written for one state; reversing components is wrong if applied to an ensemble axis."""
    return x + 0.1 * jnp.sin(x[::-1])


def test_ensemble_flow_equals_member_by_member():
    dyn = jax_da.FunctionDynamics(mixing, D)
    ens = jax.random.normal(jax.random.PRNGKey(0), (2, 5, D))
    np.testing.assert_allclose(dyn.flow(ens), jax.vmap(jax.vmap(mixing))(ens), rtol=1e-12)
    op = jax_da.FunctionOperator(lambda x: jnp.array([x[0] * x[3], x[1:].sum()]), D, 2)
    np.testing.assert_allclose(op.apply(ens), jax.vmap(jax.vmap(op.fn))(ens), rtol=1e-12)


def test_function_wrappers_reproduce_the_built_in_linear_model():
    linear = jax_da.StateSpaceModel(jax_da.LinearDynamics(A), jax_da.Linear(A[:2]), jax_da.Gaussian.isotropic(2, 0.3),
                                    jnp.ones(D), jax_da.Gaussian.isotropic(D, 1.0), jax_da.Gaussian.isotropic(D, 0.2))
    wrapped = linear.replace(dynamics=jax_da.FunctionDynamics(lambda x: A @ x, D),
                             obs_operator=jax_da.FunctionOperator(lambda x: A[:2] @ x, D, 2))
    a, b = linear.simulate(jax.random.PRNGKey(1), 30), wrapped.simulate(jax.random.PRNGKey(1), 30)
    np.testing.assert_allclose(b.states, a.states, rtol=1e-12)
    np.testing.assert_allclose(b.observations, a.observations, rtol=1e-12)
    x, y = a.states[:-1], a.observations[1:]
    np.testing.assert_allclose(wrapped.log_transition_density(a.states[1:], x), linear.log_transition_density(a.states[1:], x))
    np.testing.assert_allclose(wrapped.log_likelihood(y, a.states[1:]), linear.log_likelihood(y, a.states[1:]))


def test_model_passes_through_jit_and_vmap_as_an_argument():
    ssm = jax_da.StateSpaceModel(jax_da.FunctionDynamics(mixing, D), jax_da.FunctionOperator(jnp.tanh, D, D),
                                 jax_da.Gaussian.isotropic(D, 0.1), jnp.zeros(D), model_error=jax_da.Gaussian.isotropic(D, 0.1))
    run = jax.jit(lambda m, k: m.simulate(k, 10).states)
    np.testing.assert_allclose(run(ssm, jax.random.PRNGKey(2)), ssm.simulate(jax.random.PRNGKey(2), 10).states, rtol=1e-12)
    stds = jnp.array([0.1, 0.5])
    out = jax.vmap(lambda s: ssm.replace(obs_noise=jax_da.Gaussian.isotropic(D, s)).log_likelihood(jnp.zeros(D), jnp.ones(D)))(stds)
    assert out.shape == (2,)


def test_jacobian_of_function_operator_is_exact():
    op = jax_da.FunctionOperator(lambda x: jnp.stack([x[0] ** 2, x[1] * x[2]]), 3, 2)
    x = jnp.array([1.5, -2.0, 0.5])
    np.testing.assert_allclose(jax.jacfwd(op.apply)(x), [[3.0, 0.0, 0.0], [0.0, 0.5, -2.0]])


def test_wrong_output_shapes_and_layout_rejected():
    with pytest.raises(ValueError, match=r"fn\(x\)"):
        jax_da.FunctionDynamics(lambda x: x[:2], D).flow(jnp.zeros(D))
    with pytest.raises(ValueError, match=r"fn\(x\)"):
        jax_da.FunctionOperator(lambda x: x[:2], D, 3).apply(jnp.zeros(D))
    with pytest.raises(ValueError, match="layout"):
        jax_da.FunctionDynamics(mixing, D, layout=jax_da.Ring(5))
