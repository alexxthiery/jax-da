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

The full method table is in the [README](../README.md#unified-interface); each component only needs the members of its protocol in `jax_da/protocols.py`.

Leading axes are batch axes, so an ensemble `(N, D)` goes through every method unchanged.
`simulate` splits all keys before its `lax.scan`, so a trajectory depends only on the key.

A model-error study uses two models, for example `truth.replace(dynamics=jax_da.Lorenz96(dim=40, forcing=9.0))`; `replace` is the Flax dataclass copy-with-changes.
