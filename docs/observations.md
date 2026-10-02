# Observation operators

Maps $h : \mathbb{R}^D \to \mathbb{R}^p$ in `jax_da.observations`, each with `in_dim` ($D$), `dim` ($p$), and `apply(x)`: `(..., D) -> (..., p)`.
For the Jacobian of any operator use `jax.jacfwd(op.apply)`.

| Operator | $h(x)$ | Structure exposed |
|----------|--------|-------------------|
| `Selector(in_dim, indices)`, `Selector.every(in_dim, stride, offset=0)` | `x[indices]` | `indices` (static tuple), `matrix` |
| `Linear(matrix)`, `Linear.random_orthonormal(key, in_dim, dim, row_scale_span=0.25)` | $H x$ | `matrix` |
| `Elementwise(base, kind, degree=3)` | $g(\text{base}(x))$ | `base`, `g` |

`Linear.random_orthonormal` returns orthonormal rows scaled by values spread linearly over $[1 - s, 1 + s]$: dense, well conditioned, and not exactly isometric.

`Elementwise` kinds, both monotone and therefore invertible:

- `polynomial`: $g(z) = \tfrac{z}{2}\big(1 + (|z|/2)^{\text{degree} - 1}\big)$ with odd `degree`; `degree=1` is the identity.
- `arctan`: $g(z) = \arctan z$, which saturates at $\pm\pi/2$.
