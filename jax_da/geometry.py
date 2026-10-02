"""Spatial layout of a flat state vector.

States are always flat ``(..., D)``. A geometry records what the flat index
means, so localized methods can compute distances without knowing the model.
"""

import numpy as np
from flax import struct


def _targets(dim: int, indices) -> np.ndarray:
    idx = np.arange(dim) if indices is None else np.asarray(indices)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(f"indices must be a 1D integer array, got {idx!r}")
    if idx.size and (idx.min() < 0 or idx.max() >= dim):
        raise ValueError(f"indices must lie in [0, {dim}), got range [{idx.min()}, {idx.max()}]")
    return idx


@struct.dataclass
class Unstructured:
    """No spatial structure: distances are undefined.

    Attributes:
        dim: State dimension.
    """

    dim: int = struct.field(pytree_node=False)

    def distances(self, indices=None) -> np.ndarray:
        """Raises: there is no notion of distance between components."""
        raise NotImplementedError("Unstructured geometry has no distances")


@struct.dataclass
class Ring:
    """Periodic 1D grid: component ``i`` sits at position ``i`` on a ring of ``n`` sites.

    Attributes:
        n: Number of sites (the state dimension).
    """

    n: int = struct.field(pytree_node=False)

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"Ring needs a positive number of sites, got {self.n}")

    @property
    def dim(self) -> int:
        return self.n

    def distances(self, indices=None) -> np.ndarray:
        """Shortest wrapped distance from every site to each target site.

        Args:
            indices: Target flat indices, shape ``(p,)``; all sites if None.

        Returns:
            Distances of shape ``(n, p)``, in grid units.
        """
        targets = _targets(self.n, indices)
        delta = np.abs(np.arange(self.n)[:, None] - targets[None, :])
        return np.minimum(delta, self.n - delta).astype(np.float64)


@struct.dataclass
class Torus2D:
    """Doubly periodic 2D grid with row-major flattening: ``flat = i * width + j``.

    Attributes:
        height: Number of rows.
        width: Number of columns.
    """

    height: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)

    def __post_init__(self):
        if self.height < 1 or self.width < 1:
            raise ValueError(f"Torus2D needs positive height and width, got {self.height} x {self.width}")

    @property
    def dim(self) -> int:
        return self.height * self.width

    def coordinates(self, indices=None) -> np.ndarray:
        """Grid coordinates ``(i, j)`` of flat indices, shape ``(p, 2)``."""
        targets = _targets(self.dim, indices)
        return np.stack(np.divmod(targets, self.width), axis=-1)

    def distances(self, indices=None) -> np.ndarray:
        """Euclidean distance on the torus from every site to each target site.

        Args:
            indices: Target flat indices, shape ``(p,)``; all sites if None.

        Returns:
            Distances of shape ``(height * width, p)``, in grid units.
        """
        sites = self.coordinates()
        targets = self.coordinates(indices)
        delta = np.abs(sites[:, None, :] - targets[None, :, :])
        sizes = np.array([self.height, self.width])
        delta = np.minimum(delta, sizes - delta)
        return np.sqrt((delta.astype(np.float64) ** 2).sum(axis=-1))
