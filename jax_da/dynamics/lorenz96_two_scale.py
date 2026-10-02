"""Two-scale Lorenz (1996) model: slow ring variables each coupled to a ring of fast ones."""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape
from jax_da.dynamics._integrate import rk4, spin
from jax_da.geometry import Unstructured


@struct.dataclass
class Lorenz96TwoScale:
    """Two-scale Lorenz-96 one-interval map over ``dt``, integrated with RK4.

        dX_k/dt   = (X_{k+1} - X_{k-2}) X_{k-1} - X_k + F - (h c / b) sum_j Y_{j,k}
        dY_{j,k}/dt = c b Y_{j+1,k} (Y_{j-1,k} - Y_{j+2,k}) - c Y_{j,k} + (h c / b) X_k

    The fast variables form one ring of length ``K * J`` (``Y_{J,k}`` neighbors
    ``Y_{1,k+1}``), as in Lorenz (1996). The state is ``[X_1..X_K, Y_{1,1}..Y_{J,K}]``,
    shape ``(K + K * J,)``; observe the slow part with ``Selector(dim, range(K))``.
    Typical use: simulate the truth with this model and assimilate with
    single-scale ``Lorenz96(dim=K)``, a structural model error.

    Attributes:
        n_slow: ``K`` (static).
        n_fast: ``J``, fast variables per slow one (static).
        dt: Interval between observations.
        forcing, coupling_h, time_scale_c, amplitude_b: ``F, h, c, b``.
        substeps: RK4 steps per interval (static); the fast scale needs about ``c`` times more than single-scale L96.
    """

    n_slow: int = struct.field(pytree_node=False, default=8)
    n_fast: int = struct.field(pytree_node=False, default=32)
    dt: float = 0.05
    forcing: float = 20.0
    coupling_h: float = 1.0
    time_scale_c: float = 10.0
    amplitude_b: float = 10.0
    substeps: int = struct.field(pytree_node=False, default=50)

    @property
    def dim(self) -> int:
        return self.n_slow * (1 + self.n_fast)

    @property
    def geometry(self) -> Unstructured:
        return Unstructured(self.dim)

    def rhs(self, x: Array) -> Array:
        K = self.n_slow
        X, Y = x[..., :K], x[..., K:]
        h, c, b = self.coupling_h, self.time_scale_c, self.amplitude_b
        coupling = h * c / b
        y_sum = Y.reshape(Y.shape[:-1] + (K, self.n_fast)).sum(-1)
        dX = (jnp.roll(X, -1, -1) - jnp.roll(X, 2, -1)) * jnp.roll(X, 1, -1) - X + self.forcing - coupling * y_sum
        dY = (c * b * jnp.roll(Y, -1, -1) * (jnp.roll(Y, 1, -1) - jnp.roll(Y, -2, -1)) - c * Y
              + coupling * jnp.repeat(X, self.n_fast, axis=-1))
        return jnp.concatenate([dX, dY], axis=-1)

    @property
    def in_dim(self) -> int:
        return self.dim

    def __call__(self, x: Array) -> Array:
        """State after one interval ``dt``, ``(..., K + K J) -> (..., K + K J)``."""
        check_event_shape(x, (self.dim,))
        return rk4(self.rhs, x, self.dt, self.substeps)

    def initial_condition(self, key: Array) -> Array:
        """``X ~ F + N(0, 0.5^2)``, ``Y ~ N(0, b^-2)`` (fast amplitude of order ``X / b``, Wilks 2005)."""
        kx, ky = jax.random.split(key)
        X = self.forcing + 0.5 * jax.random.normal(kx, (self.n_slow,))
        Y = jax.random.normal(ky, (self.n_slow * self.n_fast,)) / self.amplitude_b
        return jnp.concatenate([X, Y])

    def spinup(self, x: Array, n_steps: int) -> Array:
        """Apply the one-interval map ``n_steps`` times."""
        return spin(self, x, n_steps)
