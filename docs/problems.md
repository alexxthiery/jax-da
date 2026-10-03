# Presets

Functions in `jax_da.problems` returning a `StateSpaceModel` with settings from the literature.
Keyword arguments override any default.
Chaotic presets spin up from `attractor_seed` to a point on the attractor and draw $x_0$ around it with std `initial_std` (a `PointMass` when it is 0); the same arguments always give the same model.

| Preset | Model | Observations | Basis |
|--------|-------|--------------|--------|
| `linear_gaussian()` | damped rotations, $D = 4$, model error std 0.3 | every 2nd component, std 0.5 | oracle test bed |
| `linear_gaussian_full()` | dense $A$ with spectral radius 0.95, $D = 6$, full $Q$ (mean variance $0.3^2$), $x_0 \sim N(0, P_0)$ with full $P_0$ | dense $H$, $p = 3$, full $R$ (mean variance $0.5^2$) | general oracle test bed |
| `advection_diffusion()` | exact advection-diffusion with damping on a ring (256 cells) or torus, correlated model error, stationary $x_0$ | every 8th cell per axis, std 0.5 | high-dimensional oracle test bed; [advection_diffusion.md](advection_diffusion.md) |
| `stochastic_volatility()` | $x_t = \mu + \phi(x_{t-1} - \mu) + \sigma\eta_t$, $\phi = 0.98$, $\sigma = 0.15$, stationary $x_0$ | $y_t = \beta e^{x_t/2}\varepsilon_t$, $\beta = 0.8$ (`Multiplicative`) | Kim, Shephard, and Chib (1998); flowsmc example |
| `nonlinear_poisson()` | $x_t = \rho x_{t-1} + \alpha(\tanh^2(B x_{t-1}) - 1/4) + q\eta_t$, $D = 4$, $\rho = \alpha = 0.55$, $q = 0.45$ | $p = 32$ Poisson counts with log-rate $Cx + b$ | flowsmc "quick" benchmark |
| `integrated_random_walk()` | Brownian motion integrated `order - 1` times (`order=2`: constant velocity), `discretization` `"exact"` or `"euler"` | position, std 1 | standard tracking model; [partial_noise.md](partial_noise.md) |
| `lorenz63()` | Lorenz-63, `dt = 0.25` | all components, $R = 2I$ | Sakov et al. (2012) |
| `lorenz96()` | Lorenz-96, $D = 40$, $F = 8$, `dt = 0.05` | all sites, $R = I$; `obs_every` thins | Sakov and Oke (2008) |
| `lorenz96_two_scale()` | two-scale Lorenz-96, $K = 8$, $J = 32$, $F = 20$ | slow variables, std 1 | Wilks (2005) |
| `kuramoto_sivashinsky()` | KS, 128 points, $L = 32\pi$, `dt = 1` | every 8th point, std 0.7 | grid and network of Bach et al. (2025) |
| `kolmogorov()` | Kolmogorov flow, $64 \times 64$, $\mathrm{Re} = 100$, `dt = 0.2`, model error std 0.2 | $8 \times 8$ sub-grid, std 0.1 | $64 \times 64$ grid as in Rozet and Louppe (2023); other settings from flow-da |

`kuramoto_sivashinsky` and `kolmogorov` need the `pde` extra; their spin-up takes seconds.

`linear_gaussian_full(state_dim, obs_dim, spectral_radius, model_error_std, obs_std, initial_std, seed)` draws its matrices from `numpy.random.default_rng(seed)`, so it is identical across JAX versions and devices:
$A$ is a Gaussian matrix rescaled to `spectral_radius`; $H$ is Gaussian with entry variance $1/D$; each of $Q$, $R$, $P_0$ is $G G^\top / d + I$ rescaled to the requested mean variance, which is correlated and well conditioned.

`stochastic_volatility(dim, phi, sigma, beta, mu)` is independent per component; its likelihood is far from Gaussian in $x$, so Kalman-type methods are not consistent on it, while particle filters are.
`nonlinear_poisson(state_dim, obs_dim, rho, alpha, q, seed)` builds $B$ (spectral norm at most 1.2), $C$ (entry variance $1/D$), and $b_i = -0.2 + 0.1 \cos i$ as flowsmc does, from `numpy.random.default_rng(seed)`; $\rho$, $\alpha$, $q$ are pytree children, so they can be learned by gradient.

`advection_diffusion(..., forcing_mask=mask)` restricts the model error to the masked cells (an `Embedded` law); its singular stationary initial law is kept exactly as a low-rank `Embedded` Gaussian.
`jd.delayed(ssm, L)` builds the delayed-observation version of any preset ([partial_noise.md](partial_noise.md)).
