"""Observation operators ``h: R^D -> R^p``.

Every operator has ``in_dim`` (D), ``dim`` (p), and ``apply(x)`` mapping
``(..., D)`` to ``(..., p)``. Linear operators expose their structure
(``indices`` or ``matrix``) for methods that exploit it; for a Jacobian of any
operator use ``jax.jacfwd(op.apply)``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape


@struct.dataclass
class Selector:
    """Observe a subset of state components: ``h(x) = x[indices]``.

    Attributes:
        in_dim: State dimension D.
        indices: Observed flat indices, a tuple of ints (static).
    """

    in_dim: int = struct.field(pytree_node=False)
    indices: tuple[int, ...] = struct.field(pytree_node=False)

    def __post_init__(self):
        idx = np.asarray(self.indices)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError(f"indices must be a nonempty 1D sequence, got {self.indices}")
        if idx.min() < 0 or idx.max() >= self.in_dim:
            raise ValueError(f"indices must lie in [0, {self.in_dim}), got {self.indices}")

    @classmethod
    def every(cls, in_dim: int, stride: int, offset: int = 0) -> "Selector":
        """Observe components ``offset, offset + stride, ...``."""
        return cls(in_dim=in_dim, indices=tuple(range(offset, in_dim, stride)))

    @property
    def dim(self) -> int:
        return len(self.indices)

    @property
    def matrix(self) -> Array:
        """Dense ``(p, D)`` 0/1 matrix of the selection."""
        return jnp.eye(self.in_dim)[jnp.asarray(self.indices)]

    def apply(self, x: Array) -> Array:
        check_event_shape(x, (self.in_dim,))
        return x[..., jnp.asarray(self.indices)]


@struct.dataclass
class Linear:
    """Dense linear observation: ``h(x) = H x``.

    Attributes:
        matrix: ``H``, shape ``(p, D)``.
    """

    matrix: Array

    @classmethod
    def random_orthonormal(cls, key: Array, in_dim: int, dim: int, row_scale_span: float = 0.25) -> "Linear":
        """Random well-conditioned dense ``H``: orthonormal rows times mild scales.

        Row ``i`` is scaled by a value spread linearly over
        ``[1 - row_scale_span, 1 + row_scale_span]``, so ``H`` is dense,
        well conditioned, and not exactly isometric.
        """
        if dim > in_dim:
            raise ValueError(f"dim={dim} must be <= in_dim={in_dim} for orthonormal rows")
        q, _ = jnp.linalg.qr(jax.random.normal(key, (in_dim, dim)))
        scales = jnp.linspace(1.0 - row_scale_span, 1.0 + row_scale_span, dim)
        return cls(matrix=scales[:, None] * q.T)

    @property
    def in_dim(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    def apply(self, x: Array) -> Array:
        check_event_shape(x, (self.in_dim,))
        return x @ self.matrix.T


ELEMENTWISE_KINDS = ("polynomial", "arctan")


@struct.dataclass
class Elementwise:
    """Nonlinear observation ``h(x) = g(base(x))`` with ``g`` applied per component.

    ``polynomial``: ``g(z) = z / 2 * (1 + (|z| / 2) ** (degree - 1))``, odd
    ``degree``; ``degree=1`` is the identity. ``arctan``: ``g(z) = arctan(z)``,
    a saturating map. Both are monotone, so ``g`` is invertible.

    Attributes:
        base: Inner linear operator (``Selector`` or ``Linear``).
        kind: ``"polynomial"`` or ``"arctan"``.
        degree: Polynomial degree (ignored for arctan).
    """

    base: Selector | Linear
    kind: str = struct.field(pytree_node=False, default="polynomial")
    degree: int = struct.field(pytree_node=False, default=3)

    def __post_init__(self):
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

    def apply(self, x: Array) -> Array:
        return self.g(self.base.apply(x))
