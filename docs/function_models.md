# Function-defined models

Any model of the form

$$
x_{t+1} = F(x_t) + \eta_t, \quad \eta_t \sim N(0, Q), \qquad y_t = h(x_t) + \varepsilon_t, \quad \varepsilon_t \sim N(0, R),
$$

is built from two plain JAX functions, each written for a single state:

```python
ssm = jax_da.StateSpaceModel(
    dynamics=jax_da.FunctionDynamics(F, dim=D),             # F: (D,) -> (D,), one assimilation interval
    obs_operator=jax_da.FunctionOperator(h, in_dim=D, dim=p),  # h: (D,) -> (p,)
    obs_noise=jax_da.Gaussian.full(R),
    model_error=jax_da.Gaussian.full(Q),
    initial_mean=m0,
    initial_noise=jax_da.Gaussian.full(P0),
)
```

Any noise law can replace the Gaussians.
`examples/custom_model.py` is a complete nonlinear example with a bimodal posterior.

| Wrapper | Fields | Behaviour |
|---------|--------|-----------|
| `FunctionDynamics(fn, dim, layout=None)` | `fn`, `dim`, `layout` (all static) | `flow(x)` applies `fn` over any leading batch axes; `geometry` is `layout` or `Unstructured` |
| `FunctionOperator(fn, in_dim, dim)` | `fn`, `in_dim`, `dim` (all static) | `apply(x)` applies `fn` over any leading batch axes |

Both wrappers apply `fn` with `jnp.vectorize`, so an ensemble `(N, D)` is mapped member by member even when `fn` mixes components (for example `x[::-1]` or `A @ x`); calling such an `fn` on an ensemble directly would silently act on the wrong axis.
Both check that `fn` returns the declared trailing shape.

Because `fn` is a static field, the model passes through `jit` and `vmap`, and `jax.jacfwd(op.apply)` gives Jacobians.
Values captured by `fn`'s closure are compile-time constants: they cannot be swept with `vmap` or differentiated with `grad`.
When a parameter must be differentiable or batched, write a `flax.struct.dataclass` with the parameter as a field instead; `Lorenz96` is the template, and [CONTRIBUTING.md](../CONTRIBUTING.md) lists the steps.

The built-in classes are not thin wrappers of this kind on purpose: `LinearDynamics`, `Selector`, and `Linear` expose `matrix` and `indices` (the Kalman oracle and EnKF-type methods use them), and the ODE models expose their parameters as differentiable fields.
