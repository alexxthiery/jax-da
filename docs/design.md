# jax-da design

jax-da is a library of state-space models (SSMs) for testing data assimilation (DA) algorithms in JAX.
It simulates data, samples and evaluates transition and observation densities, scores ensembles against the truth, and gives exact answers where they exist.
It contains no assimilation algorithm: projects write their own filters, smoothers, and loops against these objects.

The library is to data assimilation what [jax-pdf](https://github.com/alexxthiery/jax-pdf) is to sampling: shared benchmark models behind one small interface.

## Scope

In:

- SSMs: dynamics, model-error laws, observation operators, observation-noise laws.
- Trajectory simulation from a key, on the fly; no files.
- Named presets with settings from the literature.
- Scoring functions for ensembles against the truth.
- `KalmanOracle`: exact filtering and smoothing for linear-Gaussian SSMs.
- Geometry metadata (ring, 2D torus) and distances, for localized methods.

Out:

- Assimilation algorithms, including baselines. Examples live in `examples/`.
- A cycling driver or a method protocol. Each project owns its loop.
- Training, plotting, experiment management, disk caching.

## The model

One time step is one assimilation interval:

$$
x_{t+1} = \mathcal{M}(x_t) + \eta_t, \qquad y_t = h(x_t) + \varepsilon_t,
$$

with $\mathcal{M}$ the deterministic flow over the model's `dt`, $\eta_t$ the model error (possibly absent), and $\varepsilon_t$ the observation noise.
Observations exist at every step; sparser observation in time means a larger `dt`.

## Interface

All objects are `flax.struct.dataclass` pytrees, as in jax-pdf: numeric parameters are pytree children (so they can be swept with `vmap` and differentiated), sizes and modes are static fields.
States are flat, shape `(..., D)`, with any leading batch axes; an ensemble is a batch axis.
Every method checks the trailing shape and raises `ValueError` on a mismatch.

### `StateSpaceModel`

| Member | Shapes | Meaning |
|--------|--------|---------|
| `state_dim`, `obs_dim`, `geometry` | static | dimensions; `Unstructured`, `Ring(n)`, or `Torus2D(height, width)` (row-major flattening) |
| `sample_initial(key, shape=())` | `shape + (D,)` | draws from the initial law |
| `mean_transition(x)` | `(..., D) -> (..., D)` | $\mathcal{M}(x)$ |
| `sample_transition(key, x)` | `(..., D) -> (..., D)` | draw of $x_{t+1} \mid x_t$ |
| `log_transition_density(x_next, x)` | `-> (...)` | $\log p(x_{t+1} \mid x_t)$; raises `ValueError` when the model has no model error (the transition is a point mass) |
| `observe_mean(x)` | `(..., D) -> (..., p)` | $h(x)$ |
| `sample_observation(key, x)` | `(..., D) -> (..., p)` | draw of $y_t \mid x_t$ |
| `log_likelihood(y, x)` | `-> (...)` | $\log p(y_t \mid x_t)$ |
| `simulate(key, n_steps, x0=None)` | `(T, D), (T, p)` | truth trajectory and observations |

A model-error study builds two SSMs (truth and forecast) with different parameters, for example with `dataclasses.replace`.

### Building blocks

Each is usable on its own.

- **Dynamics** (`jax_da.dynamics`): `dim`, `dt`, `geometry`, `flow(x)`, `initial_condition(key)`, `spinup(x, n_steps)`.
  `LinearGaussian`, `Lorenz63`, `Lorenz96`, `Lorenz96TwoScale`, `KuramotoSivashinsky` and `KolmogorovFlow` (the last two need the `pde` extra, Exponax).
- **Noise laws** (`jax_da.noise`): `dim`, `sample(key, shape)`, `log_prob(e)`, `cov()` (raises `NotImplementedError` when the covariance does not exist).
  `Gaussian` (scalar, diagonal, or full covariance), `StudentT`, `Laplace`, `Cauchy`, `GaussianMixture`.
- **Observation operators** (`jax_da.observations`): `in_dim`, `dim`, `apply(x)`.
  `Selector(indices)` exposes `indices`; `Linear(H)` exposes `matrix`; `Elementwise(base, kind)` is $g(Hx)$ with polynomial or arctan $g$.
  For a Jacobian, use `jax.jacfwd(op.apply)`.

### Extras

- **Presets** (`jax_da.problems`): functions returning an SSM with documented literature settings.
- **Metrics** (`jax_da.metrics`): RMSE, CRPS, spread, spread-skill ratio, coverage, rank histogram, energy score; pure functions of an ensemble `(..., N, D)` and a truth `(..., D)`.
- **Oracle** (`jax_da.oracles.KalmanOracle`): filtering means and covariances, RTS smoothing, and log-evidence for linear-Gaussian SSMs.

## Conventions

- PRNG keys are explicit; `simulate` splits all keys before its `lax.scan`, so a trajectory depends only on the key.
- float32 by default; enable `jax_enable_x64` for float64 (the oracle tests do).
- Methods trace under `jit`, `vmap`, and `grad`.
- Tests check semantics against independent references (closed forms, quadrature, Monte Carlo, the oracle), not snapshots.
