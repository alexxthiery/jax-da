# Presets

Functions in `jax_da.problems` returning a `StateSpaceModel` with settings from the literature.
Keyword arguments override any default.
Chaotic presets place `initial_mean` on the attractor by spinning up from `attractor_seed`, and draw $x_0$ around it with std `initial_std`; the same arguments always give the same model.

| Preset | Model | Observations | Basis |
|--------|-------|--------------|--------|
| `linear_gaussian()` | damped rotations, $D = 4$, model error std 0.3 | every 2nd component, std 0.5 | oracle test bed |
| `lorenz63()` | Lorenz-63, `dt = 0.25` | all components, $R = 2I$ | Sakov et al. (2012) |
| `lorenz96()` | Lorenz-96, $D = 40$, $F = 8$, `dt = 0.05` | all sites, $R = I$; `obs_every` thins | Sakov and Oke (2008) |
| `lorenz96_two_scale()` | two-scale Lorenz-96, $K = 8$, $J = 32$, $F = 20$ | slow variables, std 1 | Wilks (2005) |
| `kuramoto_sivashinsky()` | KS, 128 points, $L = 32\pi$, `dt = 1` | every 8th point, std 0.7 | grid and network of Bach et al. (2025) |
| `kolmogorov()` | Kolmogorov flow, $64 \times 64$, $\mathrm{Re} = 100$, `dt = 0.2`, model error std 0.2 | $8 \times 8$ sub-grid, std 0.1 | $64 \times 64$ grid as in Rozet and Louppe (2023); other settings from flow-da |

`kuramoto_sivashinsky` and `kolmogorov` need the `pde` extra; their spin-up takes seconds.
