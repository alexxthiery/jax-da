# Maps

A map is a deterministic function $\mathbb{R}^{\text{in\_dim}} \to \mathbb{R}^{\text{dim}}$, called as `map(x)` on `(..., in_dim)` with any leading batch axes.
Maps are the deterministic parts of a model: dynamics over one interval (`in_dim == dim`) or observation operators.
The chaotic models in `jax_da.dynamics` ([Lorenz96](lorenz96.md), [KuramotoSivashinsky](kuramoto_sivashinsky.md), and the others) are maps too.
For the Jacobian of any map use `jax.jacfwd(map)`.

| Map | $x \mapsto$ | Structure exposed |
|-----|-------------|-------------------|
| `Linear(matrix, offset=0)` | $A x + b$ | `matrix`, `offset` |
| `Selector(in_dim, indices)`, `Selector.every(in_dim, stride, offset=0)` | `x[indices]` | `indices` (static tuple), `matrix` |
| `Elementwise(base, kind, degree=3)` | $g(\text{base}(x))$, componentwise $g$ | `base`, `g` |
| `Function(fn, in_dim, dim)` | `fn(x)` for any JAX function of one state | `fn`; see [function_models.md](function_models.md) |

Constructors:

- `Linear.damped_rotation(dim, decay=0.98, angle=0.2)`: block-diagonal $2 \times 2$ rotations by `angle` scaled by `decay`, stable dynamics for `decay < 1` (an odd `dim` gets a final `decay` on the diagonal).
- `Linear.random_orthonormal(key, in_dim, dim, row_scale_span=0.25)`: orthonormal rows scaled by values spread linearly over $[1 - s, 1 + s]$; dense, well conditioned, not exactly isometric.

`Elementwise` kinds, both monotone and therefore invertible:

- `polynomial`: $g(z) = \tfrac{z}{2}\big(1 + (|z|/2)^{\text{degree} - 1}\big)$ with odd `degree`; `degree=1` is the identity.
- `arctan`: $g(z) = \arctan z$, which saturates at $\pm\pi/2$.

`Linear` reads its dimensions from the trailing axes of `matrix`, so a model batched by `vmap` keeps them.
