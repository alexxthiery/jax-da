"""Lorenz (1996) single-scale model on a ring."""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.dynamics._integrate import rk4, spin
from jax_da.geometry import Ring


def lorenz96_rhs(x: Array, forcing) -> Array:
    """``dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`` with periodic indices."""
    return (jnp.roll(x, -1, axis=-1) - jnp.roll(x, 2, axis=-1)) * jnp.roll(x, 1, axis=-1) - x + forcing


@struct.dataclass
class Lorenz96:
    """Lorenz-96 flow over one interval ``dt``, integrated with RK4.

    With ``forcing = 8`` the system is chaotic for ``dim >= 4``; one model
    time unit is about 5 days of atmospheric error growth, so ``dt = 0.05``
    is roughly 6 hours (Lorenz 1996).

    Attributes:
        dim: Number of sites on the ring (static).
        dt: Interval between observations.
        forcing: Forcing constant ``F``.
        substeps: RK4 steps per interval (static).
    """

    dim: int = struct.field(pytree_node=False, default=40)
    dt: float = 0.05
    forcing: float = 8.0
    substeps: int = struct.field(pytree_node=False, default=4)

    def __post_init__(self):
        if self.dim < 4:
            raise ValueError(f"Lorenz-96 needs dim >= 4, got {self.dim}")

    @property
    def geometry(self) -> Ring:
        return Ring(self.dim)

    def rhs(self, x: Array) -> Array:
        return lorenz96_rhs(x, self.forcing)

    def flow(self, x: Array) -> Array:
        """State after one interval ``dt``, ``(..., D) -> (..., D)``."""
        check_event_shape(x, (self.dim,))
        return rk4(self.rhs, x, self.dt, self.substeps)

    def initial_condition(self, key: Array) -> Array:
        """``F`` plus a small perturbation; spin up to reach the attractor."""
        return self.forcing + 0.5 * jax.random.normal(key, (self.dim,))

    def spinup(self, x: Array, n_steps: int) -> Array:
        """Apply ``flow`` ``n_steps`` times."""
        return spin(self.flow, x, n_steps)
