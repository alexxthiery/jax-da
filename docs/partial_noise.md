# Partial noise and delayed observations

Some models are hard for particle methods for a structural reason: the observations measure directions of the state that the current transition does not randomize.
Fresh noise reaches those directions only after several transitions, so a filter can only select among particles by randomness they inherited, not adjust the randomness it just drew.

For a linear model $x_t = A x_{t-1} + G \varepsilon_t$, $y_t = H x_t + \eta_t$, noise injected $k$ steps ago acts on the observation through $H A^k G$.
The models below have $H A^k G = 0$ (or tiny) for the first few $k$, while the observation is informative.
jax-da provides the models, not inference algorithms; the linear-Gaussian ones keep exact answers through `KalmanOracle`.

## Building blocks

| Object | What it is |
|--------|------------|
| `Embedded(law, G)` | the law of $G z$, $z \sim$ `law`: noise confined to the range of `G`; no density on the full space |
| `Precomposed(law, map)` | a conditional law applied to `map(x)`, for example an observation of one block of the state |
| `Lagged(transition, L)` | transition of the stacked history $(x_t, \dots, x_{t-L})$, newest block first |
| `History(initial, transition, L)` | law of the first $L + 1$ states of a trajectory, stacked newest first |
| `delayed(ssm, L)` | the delayed-observation version of any model, built from the three above |

Singular noise is always explicit: `Gaussian.full` still rejects a singular covariance, and `Embedded` declares the subspace.

## Delayed observation of any model

`jd.delayed(ssm, L)` makes $y_t$ observe $x_{t-L}$: the state is $u_t = (x_t, \dots, x_{t-L})$, the newest block evolves by the base transition, the others shift, and the base observation is applied to the oldest block.
Since the oldest block of $u_t$ is the base state $x_t$, the filtering law of that block given $y_{1:t}$ equals the base filter; for linear-Gaussian bases `KalmanOracle` computes it exactly.

```python
import jax
import jax_da as jd

base = jd.problems.lorenz96(dim=40, obs_every=2, model_error_std=0.1)
model = jd.delayed(base, 5)                                  # state dimension 6 * 40
traj = model.simulate(jax.random.PRNGKey(0), 200)

exact = jd.delayed(jd.problems.linear_gaussian_full(), 5)    # still solved exactly
f = jd.KalmanOracle.from_ssm(exact).filter(exact.simulate(jax.random.PRNGKey(1), 50).observations)
```

## Integrated random walks

`jd.problems.integrated_random_walk(order, discretization)` observes the position of a Brownian motion integrated `order - 1` times (`order=2`: constant velocity).
With `discretization="euler"` the noise enters only the highest derivative, so $H A^k G = 0$ exactly for $k <$ `order - 1`; with `"exact"` the covariance is full rank but $H Q H^\top = O(\Delta^{2n-1})$, the same geometry hidden as anisotropy.

## Localized forcing with remote sensors

`jd.problems.advection_diffusion(..., forcing_mask=mask)` injects the model error only in the masked cells; information reaches sensors elsewhere after the advection travel time.
The stationary initial law is then singular to machine precision (diffusion removes the fine structure of fields far from the forcing), and it is kept exactly as a low-rank `Embedded` Gaussian.

```python
import numpy as np

mask = np.zeros(256, dtype=bool)
mask[:16] = True                                             # noise only in the first 16 cells
model = jd.problems.advection_diffusion((256,), obs_every=8, forcing_mask=mask)
```

## Kuramoto-Sivashinsky forced in its lowest Fourier modes

A nonlinear version of the same geometry: stochastic forcing in a few large-scale modes, observations in physical space.
Fine scales are only reached through the nonlinear cascade, so the observed small-scale structure reflects forcing injected earlier.

```python
import jax.numpy as jnp

n, modes = 64, 3
x = 2 * jnp.pi * jnp.arange(n) / n
basis = jnp.stack([f(k * x) for k in range(1, modes + 1) for f in (jnp.cos, jnp.sin)], axis=1)  # (n, 2 * modes)
ks = jd.KuramotoSivashinsky(num_points=n, domain_extent=16 * jnp.pi, dt=0.5)
forcing = jd.Embedded(jd.Gaussian.isotropic(2 * modes, 0.05), basis)
x_star = ks.spinup(ks.initial_condition(jax.random.PRNGKey(2)), 200)
model = jd.StateSpaceModel(
    initial=jd.PointMass(x_star),
    transition=jd.Additive(ks, forcing),
    observation=jd.Additive(jd.Selector.every(n, 4), jd.Gaussian.isotropic(n // 4, 0.1)),
    geometry=ks.geometry,
)
traj = model.simulate(jax.random.PRNGKey(3), 50)
```
