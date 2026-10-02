"""Maps ``R^in_dim -> R^dim``: the deterministic parts of a model.

Every map has ``in_dim``, ``dim``, and is called as ``map(x)``, mapping
``(..., in_dim)`` to ``(..., dim)`` over any leading batch axes. A map serves as
dynamics (``in_dim == dim``) or as an observation operator; the chaotic models in
``jax_da.dynamics`` are maps too. Linear maps expose their structure
(``matrix``, ``offset``, ``indices``) for methods that exploit it; for the
Jacobian of any map use ``jax.jacfwd(map)``.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape, check_finite, is_numeric, per_component, require_map


@struct.dataclass
class Linear:
    """Affine map ``x -> matrix @ x + offset``.

    Attributes:
        matrix: Shape ``(dim, in_dim)``.
        offset: Scalar or ``(dim,)``; default 0.
    """

    matrix: Array
    offset: Array = 0.0

    def __post_init__(self):
        if not is_numeric(self.matrix):
            return
        if np.ndim(self.matrix) < 2:
            raise ValueError(f"matrix must have shape (..., dim, in_dim), got shape {np.shape(self.matrix)}")
        check_finite(self.matrix, "matrix")
        per_component(self, "offset", self.dim)
        check_finite(self.offset, "offset")

    @classmethod
    def damped_rotation(cls, dim: int, decay: float = 0.98, angle: float = 0.2) -> "Linear":
        """Block-diagonal 2x2 rotations by ``angle`` scaled by ``decay``; stable dynamics for ``decay < 1``.

        An odd ``dim`` gets a final ``decay`` on the diagonal.
        """
        c, s = jnp.cos(angle), jnp.sin(angle)
        block = decay * jnp.array([[c, -s], [s, c]])
        blocks = [block] * (dim // 2) + ([decay * jnp.eye(1)] if dim % 2 else [])
        return cls(matrix=jax.scipy.linalg.block_diag(*blocks))

    @classmethod
    def random_orthonormal(cls, key: Array, in_dim: int, dim: int, row_scale_span: float = 0.25) -> "Linear":
        """Random well-conditioned dense ``matrix``: orthonormal rows times mild scales.

        Row ``i`` is scaled by a value spread linearly over
        ``[1 - row_scale_span, 1 + row_scale_span]``, so the matrix is dense,
        well conditioned, and not exactly isometric.
        """
        if dim > in_dim:
            raise ValueError(f"dim={dim} must be <= in_dim={in_dim} for orthonormal rows")
        q, _ = jnp.linalg.qr(jax.random.normal(key, (in_dim, dim)))
        scales = jnp.linspace(1.0 - row_scale_span, 1.0 + row_scale_span, dim)
        return cls(matrix=scales[:, None] * q.T)

    # Trailing axes, so a model batched by vmap (leading axes on every leaf) keeps its dimensions.
    @property
    def in_dim(self) -> int:
        return int(self.matrix.shape[-1])

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[-2])

    def __call__(self, x: Array) -> Array:
        check_event_shape(x, (self.in_dim,))
        return x @ self.matrix.T + self.offset


@struct.dataclass
class Selector:
    """Select state components: ``x -> x[indices]``.

    Attributes:
        in_dim: State dimension (static).
        indices: Selected flat indices, a tuple of ints (static).
    """

    in_dim: int = struct.field(pytree_node=False)
    indices: tuple[int, ...] = struct.field(pytree_node=False)

    def __post_init__(self):
        idx = np.asarray(self.indices)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"indices must be a nonempty 1D sequence, got {self.indices}")
        if not all(isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in self.indices):
            raise ValueError(f"indices must be integers, got {self.indices}")
        if idx.min() < 0 or idx.max() >= self.in_dim:
            raise ValueError(f"indices must lie in [0, {self.in_dim}), got {self.indices}")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError(f"indices must be distinct, got {self.indices}; for repeated "
                             f"measurements of a component use Linear with repeated rows")

    @classmethod
    def every(cls, in_dim: int, stride: int, offset: int = 0) -> "Selector":
        """Select components ``offset, offset + stride, ...``."""
        if stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride}")
        return cls(in_dim=in_dim, indices=tuple(range(offset, in_dim, stride)))

    @property
    def dim(self) -> int:
        return len(self.indices)

    @property
    def matrix(self) -> Array:
        """Dense ``(dim, in_dim)`` 0/1 matrix of the selection."""
        return jnp.eye(self.in_dim)[jnp.asarray(self.indices)]

    def __call__(self, x: Array) -> Array:
        check_event_shape(x, (self.in_dim,))
        return x[..., jnp.asarray(self.indices)]


ELEMENTWISE_KINDS = ("polynomial", "arctan")


@struct.dataclass
class Elementwise:
    """``x -> g(base(x))`` with a monotone ``g`` applied per component.

    ``polynomial``: ``g(z) = z / 2 * (1 + (|z| / 2) ** (degree - 1))``, odd
    ``degree``; ``degree=1`` is the identity. ``arctan``: ``g(z) = arctan(z)``,
    a saturating map. Both are invertible.

    Attributes:
        base: Inner map, for example ``Selector`` or ``Linear``.
        kind: ``"polynomial"`` or ``"arctan"`` (static).
        degree: Polynomial degree (static; ignored for arctan).
    """

    base: object
    kind: str = struct.field(pytree_node=False, default="polynomial")
    degree: int = struct.field(pytree_node=False, default=3)

    def __post_init__(self):
        require_map(self.base, "Elementwise base")
        if self.kind not in ELEMENTWISE_KINDS:
            raise ValueError(f"kind must be one of {ELEMENTWISE_KINDS}, got {self.kind!r}")
        if self.kind == "polynomial" and (self.degree < 1 or self.degree % 2 == 0):
            raise ValueError(f"degree must be a positive odd integer, got {self.degree}")

    @property
    def in_dim(self) -> int:
        return self.base.in_dim

    @property
    def dim(self) -> int:
        return self.base.dim

    def g(self, z: Array) -> Array:
        if self.kind == "arctan":
            return jnp.arctan(z)
        return 0.5 * z * (1.0 + (jnp.abs(z) / 2.0) ** (self.degree - 1))

    def __call__(self, x: Array) -> Array:
        return self.g(self.base(x))


@struct.dataclass
class Function:
    """Any JAX function of a single state, ``fn: (in_dim,) -> (dim,)``, as a map.

    The call applies ``fn`` over any leading batch axes with ``jnp.vectorize``,
    so ensembles are handled correctly even when ``fn`` mixes components (for
    example ``x[::-1]`` or ``A @ x``); calling such an ``fn`` on an ensemble
    directly would act on the wrong axis without error. ``fn`` is static, so
    the map passes through ``jit`` and ``vmap``; values captured by its closure
    are compile-time constants and cannot be differentiated or batched. For
    differentiable parameters write a ``flax.struct.dataclass`` map instead.

    Attributes:
        fn: Map of a single state (static).
        in_dim: Input dimension (static).
        dim: Output dimension (static).
    """

    fn: Callable[[Array], Array] = struct.field(pytree_node=False)
    in_dim: int = struct.field(pytree_node=False)
    dim: int = struct.field(pytree_node=False)

    def __call__(self, x: Array) -> Array:
        check_event_shape(x, (self.in_dim,))
        out = jnp.vectorize(self.fn, signature="(d)->(e)")(x)
        check_event_shape(out, (self.dim,), "fn(x)")
        return out
