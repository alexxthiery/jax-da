# KuramotoSivashinsky

1D Kuramoto-Sivashinsky equation in conservative form on a periodic interval $[0, L)$:

$$
u_t + u u_x + u_{xx} + u_{xxxx} = 0 .
$$

Integrated with Exponax's exponential time-differencing Runge-Kutta scheme: the stiff linear part is exact in Fourier space and the nonlinear term is pseudo-spectral with 2/3 dealiasing.
The conservative form preserves the spatial mean of $u$.
The number of unstable modes grows with $L$; $L = 32\pi$ with 128 points is spatio-temporally chaotic.
The state is the field on `num_points` equispaced points, with `Ring` geometry.
Needs the `pde` extra (`pip install 'jax-da[pde]'`).

| Field | Default | Kind | Meaning |
|-------|---------|------|---------|
| `num_points` | 128 | static | grid points |
| `domain_extent` | $32\pi$ | static | $L$ |
| `dt` | 1.0 | static | interval between observations, a multiple of `dt_inner` |
| `dt_inner` | 0.25 | static | solver step |
| `order` | 2 | static | ETDRK order (2 or 4) |

All fields are static because they define the spectral stepper, which is built once per configuration and cached.

Preset: `jax_da.problems.kuramoto_sivashinsky()` observes every 8th point every time unit, the network of Bach et al. (2025), with noise std 0.7.

References: Kassam and Trefethen (2005), Fourth-order time-stepping for stiff PDEs; Bach et al. (2025), Learning enhanced ensemble filters; Exponax.
