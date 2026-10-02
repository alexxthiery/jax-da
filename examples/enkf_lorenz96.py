"""Stochastic (perturbed-observation) EnKF on the Lorenz-96 preset.

The filter forecasts with ``ssm.sample_transition`` and builds its gain from
the ``Additive`` observation: its map's matrix ``H`` and its noise covariance
``R``. Scoring uses ``jax_da.metrics``.

    python examples/enkf_lorenz96.py
"""

import jax
import jax.numpy as jnp

import jax_da


def enkf(ssm, observations, key, n_members, inflation=1.05):
    """Analysis ensembles ``(T, N, D)`` of a perturbed-observation EnKF with multiplicative inflation."""
    obs_map, obs_noise = ssm.observation.map, ssm.observation.noise
    H, R = obs_map.matrix, obs_noise.cov()
    k_init, k_run = jax.random.split(key)
    ensemble = ssm.sample_initial(k_init, (n_members,))

    def step(ensemble, inputs):
        y, k = inputs
        k_fc, k_obs = jax.random.split(k)
        forecast = ssm.sample_transition(k_fc, ensemble)
        mean = forecast.mean(0)
        anomalies = inflation * (forecast - mean)
        forecast = mean + anomalies
        P_HT = anomalies.T @ (anomalies @ H.T) / (n_members - 1)
        gain = jnp.linalg.solve(H @ P_HT + R, P_HT.T).T
        perturbed = y + obs_noise.sample(k_obs, (n_members,))
        analysis = forecast + (perturbed - ssm.observe_mean(forecast)) @ gain.T
        return analysis, analysis

    keys = jax.random.split(k_run, observations.shape[0])
    _, analyses = jax.lax.scan(step, ensemble, (observations, keys))
    return analyses


def main():
    ssm = jax_da.problems.lorenz96()
    traj = ssm.simulate(jax.random.PRNGKey(0), 1000)
    analyses = jax.jit(enkf, static_argnums=3)(ssm, traj.observations, jax.random.PRNGKey(1), 40)
    scored = slice(200, None)  # skip the filter's spin-up
    print(f"observation noise std   {float(jnp.sqrt(ssm.observation.noise.variance()[0])):.3f}")
    print(f"analysis RMSE           {float(jax_da.metrics.rmse(analyses, traj.states)[scored].mean()):.3f}")
    print(f"analysis CRPS           {float(jax_da.metrics.crps(analyses, traj.states)[scored].mean()):.3f}")
    print(f"spread-skill ratio      {float(jax_da.metrics.spread_skill_ratio(analyses[scored], traj.states[scored])):.3f}")


if __name__ == "__main__":
    main()
