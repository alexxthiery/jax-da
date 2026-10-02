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
    np.testing.assert_array_equal(a.initial_mean, b.initial_mean)
    traj = a.simulate(jax.random.PRNGKey(0), 25)
    assert traj.states.shape == (25, a.state_dim) and traj.observations.shape == (25, a.obs_dim)
    assert bool(jnp.all(jnp.isfinite(traj.states)))


def test_lorenz96_network_and_attractor_scale():
    ssm = problems.lorenz96(obs_every=4)
    assert ssm.obs_operator.indices == tuple(range(0, 40, 4))
    traj = ssm.simulate(jax.random.PRNGKey(1), 2000)
    # Climatological mean and std of L96 with F = 8 are about 2.3 and 3.6.
    assert 1.8 < float(traj.states.mean()) < 2.8 and 3.2 < float(traj.states.std()) < 4.0


def test_two_scale_truth_pairs_with_single_scale_forecast_model():
    truth = problems.lorenz96_two_scale(n_fast=8, n_spinup=300)
    forecast = Lorenz96(dim=8, forcing=20.0)
    x = truth.simulate(jax.random.PRNGKey(2), 5).states
    assert forecast.flow(x[:, :8]).shape == (5, 8)
    assert truth.obs_operator.indices == tuple(range(8))


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
