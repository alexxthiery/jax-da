"""Linear dynamics, the model class with exact Kalman answers."""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.geometry import Unstructured


@struct.dataclass
class LinearDynamics:
    """Discrete-time linear map ``x_{t+1} = A x_t`` over one interval.

    With Gaussian model error, Gaussian observation noise, a linear operator,
    and a Gaussian initial law, the state-space model is linear-Gaussian and
    ``jax_da.oracles.KalmanOracle`` gives exact answers.

    Attributes:
        matrix: ``A``, shape ``(D, D)``.
    """

    matrix: Array

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
    def geometry(self) -> Unstructured:
        return Unstructured(self.dim)

    def flow(self, x: Array) -> Array:
        """``A x``, shape ``(..., D) -> (..., D)``."""
        check_event_shape(x, (self.dim,))
        return x @ self.matrix.T
