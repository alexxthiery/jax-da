# jax-da design

jax-da is a library of state-space models for testing data assimilation algorithms in JAX.
It simulates data, samples and evaluates transition and observation densities, scores ensembles against the truth, and gives exact answers where they exist.
It contains no assimilation algorithm.
This page records scope and the decisions behind the interface; the interface itself is in the [README](../README.md), and the component contracts are in `jax_da/protocols.py`.

## Scope

In: state-space models and their building blocks (maps, laws, conditional laws, geometry), trajectory simulation, presets, ensemble scores, and the Kalman oracle.

Out: assimilation algorithms (including baselines), a cycling driver or method protocol, training, plotting, experiment management, and disk caching.
Examples of algorithms live in `examples/`, outside the package.

## Decisions

**No driver.** An earlier draft had a cycling driver and a method protocol.
Any such protocol constrains which algorithms fit (lookahead filters, smoothers, variational windows, learned forecasters), while a library of models constrains none of them.
Each project writes its own loop, which is a few lines of `lax.scan` (see `examples/`).

**A model is three laws.** `initial`, `transition`, and `observation` are a law and two conditional laws, so transitions and observations are the same kind of object.
The classical form $f(x) + \varepsilon$ is one implementation (`Additive`, which exposes the map and noise that Kalman-type methods need); state-dependent scales (`Multiplicative`) and counts (`Poisson`) are others, and a new kind of transition or observation is a new small class, not an interface change.

**One step is one assimilation interval.** Every map covers its model's `dt`, and observations exist at every step.
Sparser observation in time means a larger `dt`, not a masked observation sequence.
This keeps `simulate` and every algorithm loop free of bookkeeping.

**Explicit initial law.** $x_0$ has its own law (`PointMass` when known), so $p(x_0)$ has a density whenever a filter or sampler needs one.
Chaotic presets put it around a point on the attractor reached by a deterministic spin-up from `attractor_seed`.

**Point masses raise.** A deterministic transition (`Additive(map)`) or a `PointMass` has no density, and its `log_prob` raises instead of returning a fake value.

**Flat states, structural typing, geometry on the model.** States are always `(..., D)`; the spatial layout is the model's `geometry` field, not the array shape, so every algorithm sees one convention.
Maps, laws, and conditional laws are duck-typed against `jax_da.protocols`, so user-written components plug in without subclassing.

**Pytree-safe validation.** Objects validate their parameters when built from real values; when JAX rebuilds a pytree with placeholder leaves (`None` or integer `in_axes` specs, `object()`), validation is skipped, and dimensions are read from trailing array axes so a model batched by `vmap` keeps them.

**Out of scope for now.** Time-varying models (a transition depending on $t$) and non-flat states (a discrete mode with a continuous state, as in switching linear dynamical systems).

**Static PDE parameters.** Kuramoto-Sivashinsky and Kolmogorov parameters define an Exponax spectral stepper, so they are static fields and the stepper is cached per configuration.
The cache is built under `jax.ensure_compile_time_eval`, so a first build inside `jit` cannot leak tracers.

**Scores comparable across ensemble sizes.** `spread_skill_ratio` includes the $(N+1)/N$ factor and `crps` offers the fair estimator, so calibration and skill do not depend on $N$ by construction.

## Conventions

- `flax.struct.dataclass` objects: numeric parameters are pytree children (sweep with `vmap`, differentiate with `grad`); sizes, modes, and loop counts are static.
- Explicit PRNG keys; `simulate` splits all keys before its `lax.scan`, so a trajectory depends only on its key.
- float32 by default; tests enable float64 for reference comparisons.
- Trailing shapes are checked and raise `ValueError`; inputs are never reshaped to make them fit.
- Tests compare against independent references (closed forms, SciPy, quadrature, conservation laws, Monte Carlo with standard-error tolerances, brute-force conditioning), never against stored outputs.
