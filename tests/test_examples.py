"""End-to-end acceptance: algorithms written against the public API reach the exact answers."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jax_da

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from bootstrap_pf_linear import bootstrap_pf  # noqa: E402
from enkf_lorenz96 import enkf  # noqa: E402


def test_particle_filter_is_consistent_with_the_kalman_oracle():
    ssm = jax_da.problems.linear_gaussian(dim=2, obs_every=1)
    traj = ssm.simulate(jax.random.PRNGKey(0), 10)
    exact = jax_da.KalmanOracle.from_ssm(ssm).filter(traj.observations)
    runs = 32
    keys = jax.random.split(jax.random.PRNGKey(1), runs)
    means, log_ev = jax.jit(jax.vmap(lambda k: bootstrap_pf(ssm, traj.observations, k, 5000)))(keys)
    # Filtering means: PF bias is O(1/N); the run average is within a few standard errors of the oracle.
    z = (means.mean(0) - exact.means) / (means.std(0, ddof=1) / np.sqrt(runs))
    assert float(jnp.abs(z).max()) < 4.5
    # Evidence: the PF estimate of Z (not log Z) is unbiased.
    ratio = jnp.exp(log_ev - exact.log_evidence)
    assert abs(float(ratio.mean()) - 1.0) < 4 * float(ratio.std(ddof=1)) / np.sqrt(runs)


def test_enkf_tracks_lorenz96_below_the_observation_noise():
    ssm = jax_da.problems.lorenz96()
    traj = ssm.simulate(jax.random.PRNGKey(0), 400)
    analyses = jax.jit(enkf, static_argnums=3)(ssm, traj.observations, jax.random.PRNGKey(1), 40)
    rmse = float(jax_da.metrics.rmse(analyses, traj.states)[100:].mean())
    assert rmse < 0.4  # observation noise std is 1.0
    assert 0.7 < float(jax_da.metrics.spread_skill_ratio(analyses[100:], traj.states[100:])) < 1.3
