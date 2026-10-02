"""The four contracts a ``StateSpaceModel`` is built from.

These are structural (duck-typed) protocols: any object with these members
works, including your own classes, with no base class or registration.
Every built-in object satisfies them; ``tests/test_interface.py`` checks it.
Shapes: states ``(..., D)``, observations ``(..., p)``; leading axes are batch axes.
"""

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Geometry(Protocol):
    """Spatial meaning of the flat state index (``Unstructured``, ``Ring``, ``Torus2D``)."""

    @property
    def dim(self) -> int: ...

    def distances(self, indices=None):
        """Distances from every state index to ``indices``, shape ``(D, p)``; raise if undefined."""


@runtime_checkable
class Dynamics(Protocol):
    """Deterministic map over one assimilation interval."""

    @property
    def dim(self) -> int: ...

    @property
    def geometry(self) -> Geometry: ...

    def flow(self, x: Array) -> Array:
        """``(..., D) -> (..., D)``."""


@runtime_checkable
class NoiseLaw(Protocol):
    """Zero-mean additive noise on ``R^dim``."""

    @property
    def dim(self) -> int: ...

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        """Draws of shape ``shape + (dim,)``."""

    def log_prob(self, e: Array) -> Array:
        """``(..., dim) -> (...)``."""

    def cov(self) -> Array:
        """``(dim, dim)``; raise ``NotImplementedError`` if undefined."""


@runtime_checkable
class ObservationOperator(Protocol):
    """Map ``h`` from states to observations."""

    @property
    def in_dim(self) -> int: ...

    @property
    def dim(self) -> int: ...

    def apply(self, x: Array) -> Array:
        """``(..., in_dim) -> (..., dim)``."""
