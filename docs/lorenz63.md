# Lorenz63

Lorenz (1963) convection model:

$$
\dot x = \sigma (y - x), \qquad \dot y = x (\rho - z) - y, \qquad \dot z = x y - \beta z,
$$

with the classical $(\sigma, \rho, \beta) = (10, 28, 8/3)$ giving the butterfly attractor.
The state has dimension 3 and no spatial geometry.

| Field | Default | Kind | Meaning |
|-------|---------|------|---------|
| `dt` | 0.05 | numeric | interval between observations |
| `sigma`, `rho`, `beta` | 10, 28, 8/3 | numeric | model parameters |
| `substeps` | 5 | static | RK4 steps per interval |

Preset: `jax_da.problems.lorenz63()` observes all components every 0.25 time units with $R = 2I$ (Sakov et al. 2012).

Reference: Lorenz (1963), Deterministic nonperiodic flow.
