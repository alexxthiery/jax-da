"""ODE dynamics: invariants, integration order, chaos."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_da.dynamics.linear import LinearDynamics
from jax_da.dynamics.lorenz63 import Lorenz63
from jax_da.dynamics.lorenz96 import Lorenz96, lorenz96_rhs


def test_lorenz96_energy_budget():
    # The advection term conserves energy: sum_i x_i * dx_i/dt = -|x|^2 + F sum_i x_i.
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 40)) * 3
    lhs = (x * lorenz96_rhs(x, 8.0)).sum(-1)
    np.testing.assert_allclose(lhs, -(x ** 2).sum(-1) + 8.0 * x.sum(-1), rtol=1e-12)


def test_lorenz96_fixed_point_and_index_convention():
    model = Lorenz96(dim=6, forcing=8.0)
    np.testing.assert_allclose(model.rhs(jnp.full(6, 8.0)), 0.0, atol=1e-12)
    x = jnp.arange(6.0)
    # i = 2: (x_3 - x_0) * x_1 - x_2 + F = 3 * 1 - 2 + 8
    assert float(model.rhs(x)[2]) == pytest.approx(9.0)


def test_lorenz63_rhs_hand_value():
    np.testing.assert_allclose(Lorenz63().rhs(jnp.array([1.0, 2.0, 3.0])),
                               [10.0, 1.0 * 25.0 - 2.0, 2.0 - 8.0])


@pytest.mark.parametrize("model", [Lorenz63(dt=0.2), Lorenz96(dim=8, dt=0.2)])
def test_rk4_is_fourth_order(model):
    x = model.spinup(model.initial_condition(jax.random.PRNGKey(1)), 50)
    ref = type(model)(**{**model.__dict__, "substeps": 256}).flow(x)
    errors = [float(jnp.abs(type(model)(**{**model.__dict__, "substeps": n}).flow(x) - ref).max())
              for n in (8, 16)]
    assert 12 < errors[0] / errors[1] < 20


def test_lorenz96_leading_lyapunov_exponent():
    # Published value for D=40, F=8 is about 1.7 per model time unit.
    model = Lorenz96(dim=40, dt=0.05)
    x = model.spinup(model.initial_condition(jax.random.PRNGKey(2)), 500)
    d0 = 1e-8
    y = x + d0 * jax.random.normal(jax.random.PRNGKey(3), (40,)) / jnp.sqrt(40.0)

    def step(carry, _):
        a, b = carry
        a, b = model.flow(a), model.flow(b)
        d = jnp.linalg.norm(b - a)
        return (a, a + (b - a) * d0 / d), jnp.log(d / d0)

    _, logs = jax.lax.scan(step, (x, y), None, length=2000)
    lam = float(logs.sum() / (2000 * model.dt))
    assert 1.4 < lam < 2.0


def test_damped_rotation_is_stable_and_flows_batches():
    dyn = LinearDynamics.damped_rotation(5, decay=0.9)
    assert np.max(np.abs(np.linalg.eigvals(np.asarray(dyn.matrix)))) == pytest.approx(0.9)
    x = jnp.ones((3, 5))
    np.testing.assert_allclose(dyn.flow(x), x @ dyn.matrix.T)


def test_flow_rejects_wrong_state_shape():
    with pytest.raises(ValueError):
        Lorenz96(dim=8).flow(jnp.zeros(9))
    with pytest.raises(ValueError):
        Lorenz96(dim=3)
