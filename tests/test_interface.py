"""Shared contracts, parametrized over every public dynamics, noise law, and operator."""
import jax
import jax.numpy as jnp
import pytest

import jax_da

DYNAMICS = {
    "LinearDynamics": lambda: jax_da.LinearDynamics.damped_rotation(5),
    "Lorenz63": lambda: jax_da.Lorenz63(),
    "Lorenz96": lambda: jax_da.Lorenz96(dim=10),
    "Lorenz96TwoScale": lambda: jax_da.Lorenz96TwoScale(n_slow=4, n_fast=3),
    "KuramotoSivashinsky": lambda: jax_da.KuramotoSivashinsky(num_points=32),
    "KolmogorovFlow": lambda: jax_da.KolmogorovFlow(resolution=8),
}
PDE = {"KuramotoSivashinsky", "KolmogorovFlow"}
NOISE = {
    "Gaussian": lambda: jax_da.Gaussian.isotropic(3, 0.5),
    "StudentT": lambda: jax_da.StudentT(3, df=4.0),
    "Cauchy": lambda: jax_da.Cauchy(3),
    "Laplace": lambda: jax_da.Laplace(3),
    "GaussianMixture": lambda: jax_da.GaussianMixture(3),
}
OPERATORS = {
    "Selector": lambda: jax_da.Selector.every(6, 2),
    "Linear": lambda: jax_da.Linear.random_orthonormal(jax.random.PRNGKey(0), 6, 3),
    "Elementwise": lambda: jax_da.Elementwise(jax_da.Selector.every(6, 2), "arctan"),
}


def dynamics_params():
    return [pytest.param(name, marks=pytest.mark.pde) if name in PDE else name for name in DYNAMICS]


def test_every_public_object_is_covered():
    public = set(jax_da.dynamics.__all__) | {"Gaussian", "StudentT", "Cauchy", "Laplace", "GaussianMixture",
                                              "Selector", "Linear", "Elementwise"}
    assert public == set(DYNAMICS) | set(NOISE) | set(OPERATORS)


@pytest.mark.parametrize("name", dynamics_params())
def test_dynamics_contract(name):
    if name in PDE:
        pytest.importorskip("exponax")
    model = DYNAMICS[name]()
    D = model.dim
    assert model.geometry.dim == D
    x = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (2, 3, D))
    assert model.flow(x).shape == (2, 3, D)
    assert jnp.allclose(jax.jit(lambda m, v: m.flow(v))(model, x), model.flow(x), rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError):
        model.flow(jnp.zeros(D + 1))
    if hasattr(model, "initial_condition"):
        assert model.initial_condition(jax.random.PRNGKey(1)).shape == (D,)


@pytest.mark.parametrize("name", NOISE)
def test_noise_contract(name):
    law = NOISE[name]()
    draws = law.sample(jax.random.PRNGKey(0), (4, 2))
    assert draws.shape == (4, 2, 3)
    assert law.log_prob(draws).shape == (4, 2)
    assert jnp.allclose(jax.jit(lambda l, e: l.log_prob(e))(law, draws), law.log_prob(draws))
    with pytest.raises(ValueError):
        law.log_prob(jnp.zeros((4, 2)))


@pytest.mark.parametrize("name", OPERATORS)
def test_operator_contract(name):
    op = OPERATORS[name]()
    x = jax.random.normal(jax.random.PRNGKey(0), (5, op.in_dim))
    assert op.apply(x).shape == (5, op.dim)
    assert jnp.allclose(jax.jit(lambda o, v: o.apply(v))(op, x), op.apply(x))
    assert jax.jacfwd(op.apply)(x[0]).shape == (op.dim, op.in_dim)
    with pytest.raises(ValueError):
        op.apply(jnp.zeros(op.in_dim + 1))
