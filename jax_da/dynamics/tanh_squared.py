"""Identifiable nonlinear autoregression used with Poisson counts (flowsmc benchmark)."""

import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.geometry import Unstructured


@struct.dataclass
class TanhSquared:
    """``x -> rho x + alpha (tanh(B x)**2 - offset)``.

    The squared feature has zero derivative at the origin, so ``rho`` sets the
    persistence near 0 and ``alpha`` the state-dependent level and curvature;
    the two are identifiable from data. All parameters are pytree children, so
    they can be learned by gradient.

    Attributes:
        B: Shape ``(D, D)``.
        rho: Linear autoregressive coefficient.
        alpha: Strength of the nonlinear feature.
        offset: Subtracted from ``tanh(B x)**2``; scalar or ``(D,)``.
    """

    B: Array
    rho: float = 0.55
    alpha: float = 0.55
    offset: Array = 0.25

    @property
    def dim(self) -> int:
        return int(self.B.shape[-1])

    @property
    def in_dim(self) -> int:
        return self.dim

    @property
    def geometry(self) -> Unstructured:
        return Unstructured(self.dim)

    def __call__(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
        return self.rho * x + self.alpha * (jnp.tanh(x @ self.B.T) ** 2 - self.offset)
