"""Two-scale Lorenz-96: energy budget and index layout."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_da.dynamics.lorenz96 import lorenz96_rhs
from jax_da.dynamics.lorenz96_two_scale import Lorenz96TwoScale


def test_energy_budget_coupling_and_advection_conserve_energy():
    m = Lorenz96TwoScale(n_slow=6, n_fast=5)
    x = jax.random.normal(jax.random.PRNGKey(0), (4, m.dim)) * 2
    X, Y = x[:, :6], x[:, 6:]
    expected = -(X ** 2).sum(-1) + m.forcing * X.sum(-1) - m.time_scale_c * (Y ** 2).sum(-1)
    np.testing.assert_allclose((x * m.rhs(x)).sum(-1), expected, rtol=1e-11)


def test_zero_coupling_reduces_slow_part_to_lorenz96():
    m = Lorenz96TwoScale(n_slow=6, n_fast=5, coupling_h=0.0)
    x = jax.random.normal(jax.random.PRNGKey(1), (m.dim,))
    np.testing.assert_allclose(m.rhs(x)[:6], lorenz96_rhs(x[:6], m.forcing), rtol=1e-12)


def test_fast_ring_wraps_across_slow_sectors():
    # Y_{J,k} neighbors Y_{1,k+1}: the fast ring has length K * J, not J.
    m = Lorenz96TwoScale(n_slow=3, n_fast=2, coupling_h=0.0)
    x = jnp.zeros(m.dim).at[3 + 2].set(1.0).at[3 + 3].set(2.0).at[3 + 0].set(0.5)
    Y = x[3:]
    j = 1  # Y index 1 depends on Y[2] and Y[0], Y[3]
    expected = m.time_scale_c * m.amplitude_b * Y[2] * (Y[0] - Y[3]) - m.time_scale_c * Y[1]
    assert float(m.rhs(x)[3 + j]) == pytest.approx(float(expected))


def test_spun_up_trajectory_stays_bounded():
    m = Lorenz96TwoScale(n_slow=8, n_fast=8)
    x = m.spinup(m.initial_condition(jax.random.PRNGKey(2)), 200)
    assert bool(jnp.all(jnp.isfinite(x))) and float(jnp.abs(x[:8]).max()) < 50
