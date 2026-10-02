"""Kuramoto-Sivashinsky equation on a periodic interval (Exponax spectral solver)."""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.dynamics import _exponax
from jax_da.dynamics._integrate import spin
from jax_da.geometry import Ring


@struct.dataclass
class KuramotoSivashinsky:
    """1D Kuramoto-Sivashinsky flow ``u_t + u u_x + u_xx + u_xxxx = 0`` on ``[0, L)``, periodic.

    Integrated with Exponax's ETDRK scheme (stiff linear part exact in
    Fourier space, nonlinear term pseudo-spectral with 2/3 dealiasing). The
    state is the field on ``num_points`` equispaced points. ``L = 32 pi``
    with 128 points follows Kassam and Trefethen (2005) and Bach et al. (2025).
    All parameters are static because they define the spectral stepper.

    Attributes:
        num_points: Grid points, the state dimension.
        domain_extent: ``L``; the number of unstable modes grows with ``L``.
        dt: Interval between observations; must be a multiple of ``dt_inner``.
        dt_inner: ETDRK step.
        order: ETDRK order (2 or 4).
    """

    num_points: int = struct.field(pytree_node=False, default=128)
    domain_extent: float = struct.field(pytree_node=False, default=32 * 3.141592653589793)
    dt: float = struct.field(pytree_node=False, default=1.0)
    dt_inner: float = struct.field(pytree_node=False, default=0.25)
    order: int = struct.field(pytree_node=False, default=2)

    def __post_init__(self):
        _exponax.inner_steps(self.dt, self.dt_inner)

    @property
    def dim(self) -> int:
        return self.num_points

    @property
    def geometry(self) -> Ring:
        return Ring(self.num_points)

    def _stepper(self):
        return _exponax.stepper("KuramotoSivashinskyConservative", num_spatial_dims=1,
                                domain_extent=self.domain_extent, num_points=self.num_points,
                                dt=self.dt_inner, order=self.order)

    @property
    def in_dim(self) -> int:
        return self.dim

    def __call__(self, x: Array) -> Array:
        """Field after one interval ``dt``, ``(..., num_points) -> (..., num_points)``."""
        check_event_shape(x, (self.dim,))
        return _exponax.apply_steps(self._stepper(), x, _exponax.inner_steps(self.dt, self.dt_inner), (self.num_points,))

    def initial_condition(self, key: Array) -> Array:
        """``cos(2 x / L) (1 + sin(2 x / L))`` plus ``N(0, 0.01^2)``; spin up to reach the attractor."""
        grid = jnp.linspace(0.0, self.domain_extent, self.num_points, endpoint=False)
        u0 = jnp.cos(2 * grid / self.domain_extent) * (1 + jnp.sin(2 * grid / self.domain_extent))
        return u0 + 0.01 * jax.random.normal(key, (self.num_points,))

    def spinup(self, x: Array, n_steps: int) -> Array:
        """Apply the one-interval map ``n_steps`` times."""
        return spin(self, x, n_steps)
