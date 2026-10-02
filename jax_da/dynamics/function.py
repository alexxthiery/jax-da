"""Dynamics given by any JAX function of a single state."""

from typing import Callable

import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.geometry import Unstructured


@struct.dataclass
class FunctionDynamics:
    """Wrap ``fn: (D,) -> (D,)``, the deterministic map over one interval, as dynamics.

    ``fn`` is written for one state; ``flow`` applies it over any leading batch
    axes with ``jnp.vectorize``, so ensembles are handled correctly even when
    ``fn`` mixes components (for example ``x[::-1]`` or ``A @ x``).

    ``fn`` is a static field, so the model can be passed through ``jit`` and
    ``vmap``. Values captured by ``fn``'s closure are compile-time constants:
    they cannot be swept with ``vmap`` or differentiated with ``grad``. For
    that, write a ``flax.struct.dataclass`` with the parameter as a field
    (``Lorenz96`` is the template).

    Attributes:
        fn: Deterministic one-interval map of a single state (static).
        dim: State dimension ``D`` (static).
        layout: Spatial geometry, or None for ``Unstructured`` (static).
    """

    fn: Callable[[Array], Array] = struct.field(pytree_node=False)
    dim: int = struct.field(pytree_node=False)
    layout: object = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        if self.layout is not None and self.layout.dim != self.dim:
            raise ValueError(f"layout.dim={self.layout.dim} != dim={self.dim}")

    @property
    def geometry(self):
        return Unstructured(self.dim) if self.layout is None else self.layout

    def flow(self, x: Array) -> Array:
        """``fn`` applied to every state, ``(..., D) -> (..., D)``."""
        check_event_shape(x, (self.dim,))
        out = jnp.vectorize(self.fn, signature="(d)->(e)")(x)
        check_event_shape(out, (self.dim,), "fn(x)")
        return out
