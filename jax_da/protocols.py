"""The contracts a ``StateSpaceModel`` is built from.

These are structural (duck-typed) protocols: any object with these members
works, including your own classes, with no base class or registration.
Every built-in object satisfies them; ``tests/test_interface.py`` checks it.
Leading axes of every array argument are batch axes.
"""

from typing import Protocol, runtime_checkable

from jax import Array


@runtime_checkable
class Map(Protocol):
    """Deterministic function ``R^in_dim -> R^dim``: dynamics over one interval, or an observation operator."""

    @property
    def in_dim(self) -> int: ...

    @property
    def dim(self) -> int: ...

    def __call__(self, x: Array) -> Array:
        """``(..., in_dim) -> (..., dim)``."""


@runtime_checkable
class Law(Protocol):
    """Probability law on ``R^dim``: an initial law, or the noise inside a conditional law."""

    @property
    def dim(self) -> int: ...

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        """Draws of shape ``shape + (dim,)``."""

    def log_prob(self, x: Array) -> Array:
        """``(..., dim) -> (...)``; raise if the law has no density."""


@runtime_checkable
class ConditionalLaw(Protocol):
    """Law of ``out`` given ``x``: a transition ``x_t | x_{t-1}`` or an observation ``y_t | x_t``."""

    @property
    def in_dim(self) -> int: ...

    @property
    def dim(self) -> int: ...

    def sample(self, key: Array, x: Array) -> Array:
        """``(..., in_dim) -> (..., dim)``."""

    def log_prob(self, out: Array, x: Array) -> Array:
        """``(..., dim), (..., in_dim) -> (...)``; raise if there is no density."""

    def mean(self, x: Array) -> Array:
        """``E[out | x]``, ``(..., in_dim) -> (..., dim)``."""


@runtime_checkable
class Geometry(Protocol):
    """Spatial meaning of the flat state index (``Unstructured``, ``Ring``, ``Torus2D``)."""

    @property
    def dim(self) -> int: ...

    def distances(self, indices=None):
        """Distances from every state index to ``indices``, shape ``(D, p)``; raise if undefined."""
