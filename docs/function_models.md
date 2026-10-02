# Function-defined models

Any model

$$
x_{t+1} = F(x_t) + \eta_t, \quad \eta_t \sim N(0, Q), \qquad y_t = h(x_t) + \varepsilon_t, \quad \varepsilon_t \sim N(0, R),
$$

is built from two plain JAX functions, each written for a single state:

```python
ssm = jd.StateSpaceModel(
    initial=jd.Gaussian.full(P0, loc=m0),
    transition=jd.Additive(jd.Function(F, in_dim=D, dim=D), jd.Gaussian.full(Q)),   # F: (D,) -> (D,)
    observation=jd.Additive(jd.Function(h, in_dim=D, dim=p), jd.Gaussian.full(R)),  # h: (D,) -> (p,)
)
```

Any law can replace the Gaussians, and `Function` maps work inside `Multiplicative` and `Poisson` too.
`examples/custom_model.py` is a complete nonlinear example with a bimodal posterior.

`Function(fn, in_dim, dim)` applies `fn` with `jnp.vectorize`, so an ensemble `(N, D)` is mapped member by member even when `fn` mixes components (for example `x[::-1]` or `A @ x`); calling such an `fn` on an ensemble directly would silently act on the wrong axis.
It checks that `fn` returns the declared trailing shape.

Because `fn` is a static field, the model passes through `jit` and `vmap`, and `jax.jacfwd(map)` gives Jacobians.
Values captured by `fn`'s closure are compile-time constants: they cannot be swept with `vmap` or differentiated with `grad`.
When a parameter must be differentiable or batched, write a `flax.struct.dataclass` map with the parameter as a field instead; `TanhSquared` is a short template and [CONTRIBUTING.md](../CONTRIBUTING.md) lists the steps.

The built-in `Linear` and `Selector` are not `Function` wrappers on purpose: they expose `matrix`, `offset`, and `indices`, which the Kalman oracle and EnKF-type methods use.
