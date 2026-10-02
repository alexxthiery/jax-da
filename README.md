# jax-da

State-space models, scoring, and oracles for testing data assimilation algorithms in JAX.

jax-da simulates benchmark data assimilation problems, exposes their transition and observation densities, scores ensembles against the truth, and computes exact Kalman answers for linear-Gaussian models.
It contains no assimilation algorithm: you bring the filter, smoother, or sampler, and jax-da provides the models it runs on and the yardsticks it is measured with.
Everything traces under `jit`, `vmap`, and `grad`.

## Installation

```bash
git clone <repository-url> jax-da
cd jax-da
pip install -e '.[dev]'        # add ,pde for Kuramoto-Sivashinsky and Kolmogorov flow (Exponax)
```

## Quick start

```python
import jax
import jax_da

ssm = jax_da.problems.lorenz96()                      # D = 40, F = 8, every site observed, R = I
traj = ssm.simulate(jax.random.PRNGKey(0), 200)       # traj.states (T, D), traj.observations (T, p)

# An algorithm uses the model's methods, batched over ensemble members ...
key = jax.random.PRNGKey(1)
ensemble = ssm.sample_initial(key, (40,))              # (N, D) draws of x_0
forecast = ssm.sample_transition(key, ensemble)        # (N, D) draws of x_1 | x_0
log_w = ssm.log_likelihood(traj.observations[0], forecast)   # (N,) log p(y_1 | x_1)

# ... and any (T, N, D) ensemble is scored the same way; here a free run without assimilation.
def step(members, k):
    members = ssm.sample_transition(k, members)
    return members, members

_, free_run = jax.lax.scan(step, ensemble, jax.random.split(key, 200))
print(jax_da.metrics.rmse(free_run, traj.states).mean())
```

Complete algorithms written against this interface are in [examples/](examples/): a bootstrap particle filter checked against the Kalman oracle, a stochastic EnKF on Lorenz-96, and the same particle filter on a nonlinear model defined by two plain functions.
Any model $x_{t+1} = F(x_t) + \eta_t$, $y_t = h(x_t) + \varepsilon_t$ can be built from JAX functions $F$ and $h$: [docs/function_models.md](docs/function_models.md).

## Unified interface

A `StateSpaceModel` is

$$
x_0 = m_0 + \xi, \qquad x_t = \mathcal{M}(x_{t-1}) + \eta_t, \qquad y_t = h(x_t) + \varepsilon_t, \qquad t = 1, \dots, T,
$$

where one step is one assimilation interval.
States are flat, `(..., D)`; leading axes are batch axes, so ensembles need no special handling.
Wrong trailing shapes raise `ValueError`.

| Method | Shapes | Returns |
|--------|--------|---------|
| `sample_initial(key, shape=())` | `shape + (D,)` | draws of $x_0$ |
| `mean_transition(x)` | `(..., D) -> (..., D)` | $\mathcal{M}(x)$ |
| `sample_transition(key, x)` | `(..., D) -> (..., D)` | draw of $x_{t+1} \mid x_t$ |
| `log_transition_density(x_next, x)` | `-> (...)` | $\log p(x_{t+1} \mid x_t)$; raises for deterministic dynamics |
| `observe_mean(x)` | `(..., D) -> (..., p)` | $h(x)$ |
| `sample_observation(key, x)` | `(..., D) -> (..., p)` | draw of $y_t \mid x_t$ |
| `log_likelihood(y, x)` | `-> (...)` | $\log p(y_t \mid x_t)$ |
| `log_initial_density(x0)` | `-> (...)` | $\log p(x_0)$ |
| `simulate(key, n_steps, x0=None)` | `Trajectory` | `initial (D,)`, `states (T, D)`, `observations (T, p)` |

Details: [docs/state_space_model.md](docs/state_space_model.md).
A model is assembled from dynamics, an observation operator, and noise laws; each piece also works on its own.

## Dynamics

| Dynamics | State dimension | Geometry | Docs |
|----------|-----------------|----------|------|
| `FunctionDynamics` | configurable | optional | [docs/function_models.md](docs/function_models.md) |
| `LinearDynamics` | configurable | none | [docs/linear.md](docs/linear.md) |
| `Lorenz63` | 3 | none | [docs/lorenz63.md](docs/lorenz63.md) |
| `Lorenz96` | configurable | ring | [docs/lorenz96.md](docs/lorenz96.md) |
| `Lorenz96TwoScale` | $K(1 + J)$ | none | [docs/lorenz96_two_scale.md](docs/lorenz96_two_scale.md) |
| `KuramotoSivashinsky` | grid points | ring | [docs/kuramoto_sivashinsky.md](docs/kuramoto_sivashinsky.md) |
| `KolmogorovFlow` | resolution$^2$ | 2D torus | [docs/kolmogorov.md](docs/kolmogorov.md) |

Every dynamics has `dim`, `geometry`, and `flow(x)` over one interval; the chaotic ones add `initial_condition(key)` and `spinup(x, n_steps)`.
`geometry.distances(indices)` gives the distances localized methods need.

## Observations and noise

- Operators ([docs/observations.md](docs/observations.md)): `Selector`, `Linear`, nonlinear `Elementwise` (polynomial or arctan of a linear map), and `FunctionOperator` for any JAX function.
- Noise laws ([docs/noise.md](docs/noise.md)): `Gaussian` (isotropic, diagonal, or full), `StudentT`, `Cauchy`, `Laplace`, `GaussianMixture`, each with `sample`, `log_prob`, and `cov`.

## Presets

`jax_da.problems` returns ready-made models with settings from the literature: `linear_gaussian`, `linear_gaussian_full` (dense $A$ and $H$, full $Q$, $R$, $P_0$), `advection_diffusion` (high-dimensional, spatially structured, exact; [docs/advection_diffusion.md](docs/advection_diffusion.md)), `lorenz63`, `lorenz96`, `lorenz96_two_scale`, `kuramoto_sivashinsky`, `kolmogorov`.
See [docs/problems.md](docs/problems.md).

## Scoring

`jax_da.metrics`: `rmse`, `spread`, `crps` (standard and fair), `crps_gaussian`, `energy_score`, `coverage`, `spread_skill_ratio`, `rank_histogram`.
See [docs/metrics.md](docs/metrics.md).

## Oracle

`KalmanOracle.from_ssm(ssm)` gives exact filtering and smoothing distributions and the log-evidence of a linear-Gaussian model.
See [docs/kalman_oracle.md](docs/kalman_oracle.md).

## Design

Scope, conventions, and what is deliberately left out: [docs/design.md](docs/design.md).
Adding a model: [CONTRIBUTING.md](CONTRIBUTING.md).
