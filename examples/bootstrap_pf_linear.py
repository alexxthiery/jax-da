"""Bootstrap particle filter on the linear-Gaussian preset, checked against the Kalman oracle.

Shows the jax-da workflow: take a model, simulate data, run your own
algorithm against the model's sampling and density methods, and compare
with an exact answer.

    python examples/bootstrap_pf_linear.py
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import jax_da


def bootstrap_pf(ssm, observations, key, n_particles):
    """Filtering means ``(T, D)`` and the log-evidence estimate of a bootstrap particle filter."""
    k_init, k_run = jax.random.split(key)
    particles = ssm.sample_initial(k_init, (n_particles,))

    def step(carry, inputs):
        particles, log_evidence = carry
        y, k = inputs
        k_move, k_resample = jax.random.split(k)
        particles = ssm.sample_transition(k_move, particles)
        log_w = ssm.log_likelihood(y, particles)
        log_evidence = log_evidence + logsumexp(log_w) - jnp.log(n_particles)
        w = jnp.exp(log_w - logsumexp(log_w))
        mean = w @ particles
        idx = jax.random.choice(k_resample, n_particles, (n_particles,), p=w)
        return (particles[idx], log_evidence), mean

    keys = jax.random.split(k_run, observations.shape[0])
    (_, log_evidence), means = jax.lax.scan(step, (particles, 0.0), (observations, keys))
    return means, log_evidence


def main():
    jax.config.update("jax_enable_x64", True)
    ssm = jax_da.problems.linear_gaussian(dim=4)
    traj = ssm.simulate(jax.random.PRNGKey(0), 50)
    exact = jax_da.KalmanOracle.from_ssm(ssm).filter(traj.observations)
    run = jax.jit(bootstrap_pf, static_argnums=3)
    # Errors are Monte Carlo noise; they shrink roughly like 1 / sqrt(particles).
    print(f"{'particles':>10} {'mean |mean - oracle|':>22} {'log-evidence error':>20}")
    for n in (100, 1_000, 10_000, 100_000):
        means, log_ev = run(ssm, traj.observations, jax.random.PRNGKey(1), n)
        print(f"{n:>10} {float(jnp.abs(means - exact.means).mean()):>22.4f} "
              f"{float(log_ev - exact.log_evidence):>20.4f}")


if __name__ == "__main__":
    main()
