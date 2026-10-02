"""Lorenz (1963) three-variable convection model."""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.dynamics._integrate import rk4, spin
from jax_da.geometry import Unstructured


@struct.dataclass
class Lorenz63:
    """Lorenz-63 one-interval map over ``dt``, integrated with RK4.

    ``dx/dt = sigma (y - x)``, ``dy/dt = x (rho - z) - y``, ``dz/dt = x y - beta z``.
    The classical parameters ``(10, 28, 8/3)`` give the butterfly attractor.

    Attributes:
        dt: Interval between observations.
        sigma, rho, beta: Model parameters.
        substeps: RK4 steps per interval (static).
    """

    dt: float = 0.05
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    substeps: int = struct.field(pytree_node=False, default=5)

    def __post_init__(self):
        if self.substeps < 1:
            raise ValueError(f"substeps must be a positive integer, got {self.substeps}")

    @property
    def dim(self) -> int:
        return 3

    @property
    def geometry(self) -> Unstructured:
        return Unstructured(3)

    def rhs(self, x: Array) -> Array:
        u, v, w = x[..., 0], x[..., 1], x[..., 2]
        return jnp.stack([self.sigma * (v - u), u * (self.rho - w) - v, u * v - self.beta * w], axis=-1)

    @property
    def in_dim(self) -> int:
        return self.dim

    def __call__(self, x: Array) -> Array:
        """State after one interval ``dt``, ``(..., 3) -> (..., 3)``."""
        check_event_shape(x, (3,))
        return rk4(self.rhs, x, self.dt, self.substeps)

    def initial_condition(self, key: Array) -> Array:
        """A perturbation of ``(1, 1, 1)``; spin up to reach the attractor."""
        return jnp.ones(3) + 0.1 * jax.random.normal(key, (3,))

    def spinup(self, x: Array, n_steps: int) -> Array:
        """Apply the one-interval map ``n_steps`` times."""
        return spin(self, x, n_steps)
