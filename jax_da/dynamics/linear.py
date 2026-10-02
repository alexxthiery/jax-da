"""Linear dynamics, the model class with exact Kalman answers."""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape, is_concrete
from jax_da.geometry import Unstructured


@struct.dataclass
class LinearDynamics:
    """Discrete-time linear map ``x_{t+1} = A x_t`` over one interval.

    With Gaussian model error, Gaussian observation noise, a linear operator,
    and a Gaussian initial law, the state-space model is linear-Gaussian and
    ``jax_da.oracles.KalmanOracle`` gives exact answers.

    Attributes:
        matrix: ``A``, shape ``(D, D)``.
        layout: Spatial geometry of the state (``Ring``, ``Torus2D``), or None for
            ``Unstructured`` (static).
    """

    matrix: Array
    layout: object = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        if self.layout is not None and is_concrete(self.matrix) and self.layout.dim != self.dim:
            raise ValueError(f"layout.dim={self.layout.dim} != matrix dimension {self.dim}")

    @classmethod
    def damped_rotation(cls, dim: int, decay: float = 0.98, angle: float = 0.2) -> "LinearDynamics":
        """Block-diagonal 2x2 rotations by ``angle`` scaled by ``decay``; stable for ``decay < 1``.

        An odd ``dim`` gets a final ``decay`` on the diagonal.
        """
        c, s = jnp.cos(angle), jnp.sin(angle)
        block = decay * jnp.array([[c, -s], [s, c]])
        a = jax.scipy.linalg.block_diag(*([block] * (dim // 2) + ([decay * jnp.eye(1)] if dim % 2 else [])))
        return cls(matrix=a)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def geometry(self):
        return Unstructured(self.dim) if self.layout is None else self.layout

    def flow(self, x: Array) -> Array:
        """``A x``, shape ``(..., D) -> (..., D)``."""
        check_event_shape(x, (self.dim,))
        return x @ self.matrix.T
