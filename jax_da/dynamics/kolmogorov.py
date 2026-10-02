"""Kolmogorov flow: forced 2D incompressible Navier-Stokes in vorticity form (Exponax)."""

import jax
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.dynamics import _exponax
from jax_da.dynamics._integrate import spin
from jax_da.geometry import Torus2D


@struct.dataclass
class KolmogorovFlow:
    """Vorticity of forced 2D Navier-Stokes on the doubly periodic square ``[0, 2 pi)^2``.

        omega_t + J(psi, omega) = nu Laplacian(omega) + drag * omega + f,  Laplacian(psi) = -omega,

    with sinusoidal forcing ``f`` at wavenumber ``forcing_wavenumber`` (Exponax
    ``KolmogorovFlowVorticity``). The state is the ``resolution x resolution``
    vorticity, flattened row-major to ``(resolution**2,)``. All parameters are
    static because they define the spectral stepper.

    Attributes:
        resolution: Grid points per side.
        viscosity: ``nu = 1 / Re``.
        drag: Linear drag coefficient (negative removes energy).
        forcing_wavenumber: Forcing mode.
        forcing_scale: Forcing amplitude.
        dt: Interval between observations; must be a multiple of ``dt_inner``.
        dt_inner: Spectral solver step; keep small enough for stability at the chosen resolution.
    """

    resolution: int = struct.field(pytree_node=False, default=64)
    viscosity: float = struct.field(pytree_node=False, default=0.01)
    drag: float = struct.field(pytree_node=False, default=-0.1)
    forcing_wavenumber: int = struct.field(pytree_node=False, default=4)
    forcing_scale: float = struct.field(pytree_node=False, default=1.0)
    dt: float = struct.field(pytree_node=False, default=0.2)
    dt_inner: float = struct.field(pytree_node=False, default=0.01)

    def __post_init__(self):
        _exponax.inner_steps(self.dt, self.dt_inner)

    @property
    def dim(self) -> int:
        return self.resolution ** 2

    @property
    def geometry(self) -> Torus2D:
        return Torus2D(self.resolution, self.resolution)

    def _stepper(self):
        return _exponax.stepper("KolmogorovFlowVorticity", num_spatial_dims=2, domain_extent=2 * 3.141592653589793,
                                num_points=self.resolution, dt=self.dt_inner, diffusivity=self.viscosity,
                                drag=self.drag, injection_mode=self.forcing_wavenumber,
                                injection_scale=self.forcing_scale)

    def flow(self, x: Array) -> Array:
        """Vorticity after one interval ``dt``, ``(..., resolution**2) -> (..., resolution**2)``."""
        check_event_shape(x, (self.dim,))
        return _exponax.apply_steps(self._stepper(), x, _exponax.inner_steps(self.dt, self.dt_inner),
                                    (self.resolution, self.resolution))

    def initial_condition(self, key: Array) -> Array:
        """White-noise vorticity with std 0.5; spin up to reach the statistically stationary state."""
        return 0.5 * jax.random.normal(key, (self.dim,))

    def spinup(self, x: Array, n_steps: int) -> Array:
        """Apply ``flow`` ``n_steps`` times."""
        return spin(self.flow, x, n_steps)
