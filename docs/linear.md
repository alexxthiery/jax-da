# LinearDynamics

Discrete-time linear map $x_{t+1} = A x_t$ over one interval.
With Gaussian initial, model, and observation noise and a linear observation operator, the state-space model is linear-Gaussian and [KalmanOracle](kalman_oracle.md) gives exact answers.

`LinearDynamics.damped_rotation(dim, decay, angle)` builds block-diagonal $2 \times 2$ rotations by `angle` scaled by `decay`, stable for `decay < 1`; an odd `dim` gets a final `decay` on the diagonal.

| Field | Kind | Meaning |
|-------|------|---------|
| `matrix` | numeric | $A$, shape `(D, D)` |
| `layout` | static | geometry (`Ring`, `Torus2D`), or None for `Unstructured` |

Presets: `jax_da.problems.linear_gaussian()` (damped rotations, selector observations, isotropic noise) `jax_da.problems.linear_gaussian_full()` (dense $A$ and $H$, full $Q$, $R$, $P_0$), and `jax_da.problems.advection_diffusion()` (spatially structured, high-dimensional).
