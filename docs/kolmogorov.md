# KolmogorovFlow

Forced 2D incompressible Navier-Stokes in vorticity form on the doubly periodic square $[0, 2\pi)^2$:

$$
\omega_t + J(\psi, \omega) = \nu \Delta \omega + \lambda \omega + f, \qquad \Delta \psi = -\omega,
$$

with linear drag $\lambda$ (`drag`, negative) and sinusoidal forcing $f$ at wavenumber `forcing_wavenumber` (Exponax `KolmogorovFlowVorticity`).
Advection, diffusion, and the zero-mean forcing leave the mean vorticity unchanged, so it decays as $e^{\lambda t}$.
The state is the `resolution x resolution` vorticity flattened row-major (`flat = i * resolution + j`), with `Torus2D` geometry.
Needs the `pde` extra.

| Field | Default | Kind | Meaning |
|-------|---------|------|---------|
| `resolution` | 64 | static | grid points per side |
| `viscosity` | 0.01 | static | $\nu = 1/\mathrm{Re}$ |
| `drag` | -0.1 | static | $\lambda$ |
| `forcing_wavenumber`, `forcing_scale` | 4, 1.0 | static | forcing mode and amplitude |
| `dt` | 0.2 | static | interval between observations, a multiple of `dt_inner` |
| `dt_inner` | 0.01 | static | solver step; reduce it at higher resolution |

Preset: `jax_da.problems.kolmogorov()` observes an evenly spaced $8 \times 8$ sub-grid with noise std 0.1 and model error std 0.2.

References: Chandler and Kerswell (2013); Rozet and Louppe (2023), Score-based data assimilation; Exponax.
