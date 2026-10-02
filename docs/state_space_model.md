# StateSpaceModel

$$
x_0 = m_0 + \xi, \qquad x_t = \mathcal{M}(x_{t-1}) + \eta_t, \qquad y_t = h(x_t) + \varepsilon_t, \qquad t = 1, \dots, T,
$$

with $\xi \sim$ `initial_noise`, $\eta_t \sim$ `model_error`, $\varepsilon_t \sim$ `obs_noise`, $\mathcal{M}$ = `dynamics.flow`, and $h$ = `obs_operator.apply`.
`initial_noise=None` means $x_0 = m_0$ exactly; `model_error=None` means deterministic dynamics.
There is no observation of $x_0$.

```python
ssm = jax_da.StateSpaceModel(
    dynamics=jax_da.Lorenz96(dim=40),
    obs_operator=jax_da.Selector.every(40, 2),
    obs_noise=jax_da.Gaussian.isotropic(20, 1.0),
    initial_mean=x_on_attractor,
    initial_noise=jax_da.Gaussian.isotropic(40, 1.0),
    model_error=None,
)
traj = ssm.simulate(key, 1000)    # traj.initial (D,), traj.states (T, D), traj.observations (T, p)
```

| Member | Shapes | Meaning |
|--------|--------|---------|
| `state_dim`, `obs_dim`, `geometry` | | $D$, $p$, and the dynamics' geometry |
| `sample_initial(key, shape=())` | `shape + (D,)` | draws of $x_0$ |
| `log_initial_density(x0)` | `(..., D) -> (...)` | $\log p(x_0)$; raises `ValueError` if `initial_noise is None` |
| `mean_transition(x)` | `(..., D) -> (..., D)` | $\mathcal{M}(x)$ |
| `sample_transition(key, x)` | `(..., D) -> (..., D)` | draw of $x_{t+1} \mid x_t$ |
| `log_transition_density(x_next, x)` | `-> (...)` | $\log p(x_{t+1} \mid x_t)$; raises `ValueError` if `model_error is None` |
| `observe_mean(x)` | `(..., D) -> (..., p)` | $h(x)$ |
| `sample_observation(key, x)` | `(..., D) -> (..., p)` | draw of $y_t \mid x_t$ |
| `log_likelihood(y, x)` | `(..., p), (..., D) -> (...)` | $\log p(y_t \mid x_t)$ |
| `simulate(key, n_steps, x0=None)` | `Trajectory` | truth and observations; `observations[t]` observes `states[t]` |

Leading axes are batch axes, so an ensemble `(N, D)` goes through every method unchanged.
`simulate` splits all keys before its `lax.scan`, so a trajectory depends only on the key.

A model-error study uses two models, for example `truth.replace(dynamics=jax_da.Lorenz96(dim=40, forcing=9.0))`; `replace` is the Flax dataclass copy-with-changes.
