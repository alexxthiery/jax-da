# StateSpaceModel

A state-space model is three laws:

$$
x_0 \sim \text{initial}, \qquad x_t \mid x_{t-1} \sim \text{transition}, \qquad y_t \mid x_t \sim \text{observation}, \qquad t = 1, \dots, T.
$$

There is no observation of $x_0$.
`initial` is a [law](laws.md); `transition` and `observation` are [conditional laws](laws.md#conditional-laws).
The classical data assimilation model $x_t = \mathcal{M}(x_{t-1}) + \eta_t$, $y_t = h(x_t) + \varepsilon_t$ uses `Additive` for both:

```python
import jax_da as jd

ssm = jd.StateSpaceModel(
    initial=jd.Gaussian.isotropic(40, 1.0, loc=x_on_attractor),
    transition=jd.Additive(jd.Lorenz96(dim=40)),                       # deterministic: no noise
    observation=jd.Additive(jd.Selector.every(40, 2), jd.Gaussian.isotropic(20, 1.0)),
    geometry=jd.Ring(40),                                             # optional, for localization
)
traj = ssm.simulate(key, 1000)    # traj.initial (D,), traj.states (T, D), traj.observations (T, p)
```

Other forms swap one law:

```python
jd.Additive(jd.Linear(A, offset=b), jd.Gaussian.full(Q))                  # affine-Gaussian transition
jd.Multiplicative(jd.Linear(0.5 * I, offset=log_beta), jd.Gaussian.isotropic(d, 1.0))  # y = beta exp(x / 2) eps
jd.Poisson(jd.Linear(C, offset=b))                                         # counts
jd.PointMass(x0)                                                           # known initial state
```

The method table is in the [README](../README.md#unified-interface).
Each method delegates to one law: `sample_transition` to `transition.sample`, `log_likelihood` to `observation.log_prob`, `mean_transition` to `transition.mean`, and so on.
A deterministic transition (`Additive(map)`) or a `PointMass` initial law raises on its density instead of returning a fake value.

Leading axes are batch axes, so an ensemble `(N, D)` goes through every method unchanged.
`simulate` splits all keys before its `lax.scan`, so a trajectory depends only on the key.
The model is a pytree: pass it through `jit`, batch models with `vmap` (dimensions are read from trailing axes), and differentiate with respect to any numeric parameter.

Methods that exploit structure read it from the laws, for example an EnKF takes `ssm.observation.map.matrix` and `ssm.observation.noise.cov()` from an `Additive` observation.
A model-error study uses two models, for example `truth.replace(transition=jd.Additive(jd.Lorenz96(dim=40, forcing=9.0)))`; `replace` is the Flax dataclass copy-with-changes.
