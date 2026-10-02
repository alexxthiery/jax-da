# Advection-diffusion

`jax_da.problems.advection_diffusion` is a high-dimensional, spatially structured linear-Gaussian benchmark with exact answers from [KalmanOracle](kalman_oracle.md).
It is the test bed for localized and high-dimensional methods: grid points are statistically equivalent, the correlation length is a model parameter, and information travels from observed to unobserved points by advection.

## Model

On a periodic ring of $n$ cells or torus of $h \times w$ cells (lengths in grid cells),

$$
\partial_t u + c \cdot \nabla u = \kappa \Delta u - \lambda u + \text{noise}.
$$

One interval $dt$ is applied exactly in Fourier space: mode $k$ is multiplied by

$$
a_k = e^{-(\kappa \lvert k \rvert^2 + \lambda)\, dt}\, \prod_{\text{axes}} e^{-i c_j k_j dt},
$$

so there is no discretization error and fractional-cell shifts are exact.
On an even grid the Nyquist mode $\cos(\pi i)$ has no sine partner, and translation acts on it as the real factor $\cos(\pi c_j dt)$; whole-cell shifts are therefore exact rolls.

The model error $\eta_t \sim N(0, Q)$ is translation invariant with spectrum

$$
q_k \propto (1 - \nu)\, e^{-\lvert k \rvert^2 \ell^2 / 2} + \nu,
$$

a squared-exponential correlation of length $\ell$ (`noise_length`) plus a white fraction $\nu$ (`nugget`) that keeps $Q$ well conditioned, scaled to marginal std `model_error_std`.

With damping $\lambda > 0$ the model has a stationary law $N(0, \Sigma)$, $\Sigma = A \Sigma A^\top + Q$, whose spectrum is $q_k / (1 - \lvert a_k \rvert^2)$.
The truth starts from it, so trajectories are in statistical equilibrium from $t = 0$ and need no spin-up.

Every `obs_every`-th cell along each axis is observed with noise std `obs_std`.

## Defaults

| Argument | Default | Meaning |
|----------|---------|---------|
| `grid_shape` | `(256,)` | `(n,)` ring or `(h, w)` torus |
| `dt` | 1.0 | interval |
| `velocity` | 1.0 | cells per time unit; scalar or one per axis |
| `diffusivity` | 0.1 | $\kappa$, cells$^2$ per time unit |
| `damping` | 0.02 | $\lambda$; must be positive |
| `noise_length`, `nugget`, `model_error_std` | 10, 0.01, 0.2 | model-error correlation length, white fraction, marginal std |
| `obs_every`, `obs_std` | 8, 0.5 | observation spacing and noise |

With the defaults on 256 cells the climatological std is about 1.0, the climatological correlation length about 15 cells, and the exact filtering std about 0.25.
`obs_every=16` makes unobserved cells measurably harder than observed ones; a $32 \times 32$ torus with `obs_every=4` has $D = 1024$ and $p = 64$.

The matrices are dense, so `KalmanOracle` costs $O(D^3)$ per step; $D$ up to about 1000 is practical.
