"""A nonlinear model written as two plain functions, filtered by the generic bootstrap particle filter.

    x_{t+1} = F(x_t) + eta_t,   eta_t ~ N(0, Q)
    y_t     = h(x_t) + eps_t,   eps_t ~ N(0, R)

``F`` and ``h`` are ordinary JAX functions of one state; ``jax_da.Function``
applies them to ensembles. The particle filter from
``bootstrap_pf_linear.py`` only uses the model's sampling and density methods,
so it runs unchanged on any jax-da model.

    python examples/custom_model.py
"""

import jax
import jax.numpy as jnp

import jax_da
from bootstrap_pf_linear import bootstrap_pf


def F(x):
    """A damped, rotating pendulum-like map on R^2."""
    return 0.95 * jnp.array([x[0] + 0.3 * jnp.sin(x[1]), x[1] - 0.3 * jnp.sin(x[0])])


def h(x):
    """Squared radius and first coordinate: the sign of x[1] is invisible at one time, so the posterior is bimodal."""
    return jnp.array([x[0] ** 2 + x[1] ** 2, x[0]])


def build_model():
    return jax_da.StateSpaceModel(
        initial=jax_da.Gaussian.isotropic(2, 0.5, loc=jnp.array([1.0, 0.0])),
        transition=jax_da.Additive(jax_da.Function(F, in_dim=2, dim=2),
                                   jax_da.Gaussian.full(jnp.array([[0.02, 0.01], [0.01, 0.02]]))),
        observation=jax_da.Additive(jax_da.Function(h, in_dim=2, dim=2), jax_da.Gaussian.isotropic(2, 0.2)),
    )


def run(n_particles, n_steps=200, seed=0):
    """RMSE of the particle-filter mean, and the spread of the truth for comparison."""
    ssm = build_model()
    traj = ssm.simulate(jax.random.PRNGKey(seed), n_steps)
    means, _ = jax.jit(bootstrap_pf, static_argnums=3)(ssm, traj.observations, jax.random.PRNGKey(seed + 1), n_particles)
    rmse = float(jnp.sqrt(((means - traj.states) ** 2).mean()))
    return rmse, float(traj.states.std())


def main():
    for n in (100, 1000, 10_000):
        rmse, spread = run(n)
        print(f"particles {n:>6}   filter RMSE {rmse:.3f}   (truth std {spread:.3f})")


if __name__ == "__main__":
    main()
