"""State-space model: an initial law, a transition, and an observation.

    x_0 ~ initial,   x_t | x_{t-1} ~ transition,   y_t | x_t ~ observation,   t = 1, ..., T

There is no observation of ``x_0``. The classical data assimilation model
``x_t = M(x_{t-1}) + eta_t``, ``y_t = h(x_t) + eps_t`` is
``transition=Additive(M, eta_law)``, ``observation=Additive(h, eps_law)``.
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape, check_finite, placeholder_dims
from jax_da.geometry import Unstructured
from jax_da.protocols import ConditionalLaw, Geometry, Law


@struct.dataclass
class Trajectory:
    """A simulated truth and its observations.

    Attributes:
        initial: ``x_0``, shape ``(D,)``.
        states: ``x_1, ..., x_T``, shape ``(T, D)``.
        observations: ``y_1, ..., y_T``, shape ``(T, p)``; ``observations[t]`` observes ``states[t]``.
    """

    initial: Array
    states: Array
    observations: Array


@struct.dataclass
class StateSpaceModel:
    """A data assimilation benchmark model.

    Each component is any object satisfying its protocol in ``jax_da.protocols``,
    so user-written laws plug in without subclassing.

    Attributes:
        initial: ``Law`` of ``x_0`` (``PointMass`` for a known state).
        transition: ``ConditionalLaw`` of ``x_t`` given ``x_{t-1}``.
        observation: ``ConditionalLaw`` of ``y_t`` given ``x_t``.
        geometry: Spatial layout of the state (``Ring``, ``Torus2D``) for localized
            methods (static); defaults to ``Unstructured``.
    """

    initial: Law
    transition: ConditionalLaw
    observation: ConditionalLaw
    geometry: Geometry | None = struct.field(pytree_node=False, default=None)

    def __post_init__(self):
        if hasattr(self.initial, "in_dim"):
            raise ValueError("initial must be an unconditional law (dim, sample(key, shape), log_prob(x)), "
                             f"got the conditional law {type(self.initial).__name__}")
        for name in ("transition", "observation"):
            law = getattr(self, name)
            if not isinstance(law, ConditionalLaw):
                raise ValueError(f"{name} must be a conditional law with in_dim, dim, sample(key, x), "
                                 f"log_prob(out, x), and mean(x); got {type(law).__name__}")
        if placeholder_dims(self.initial, self.transition, self.observation):
            return
        D = self.transition.dim
        checks = (("initial.dim", self.initial.dim), ("transition.in_dim", self.transition.in_dim),
                  ("observation.in_dim", self.observation.in_dim))
        for name, value in checks:
            if value != D:
                raise ValueError(f"{name}={value} != transition.dim={D}")
        if self.geometry is None:
            object.__setattr__(self, "geometry", Unstructured(D))  # frozen dataclass: set once at construction
        elif self.geometry.dim != D:
            raise ValueError(f"geometry.dim={self.geometry.dim} != state dimension {D}")

    @property
    def state_dim(self) -> int:
        return self.transition.dim

    @property
    def obs_dim(self) -> int:
        return self.observation.dim

    def sample_initial(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        """Draws of ``x_0``, shape ``shape + (D,)``."""
        return self.initial.sample(key, shape)

    def log_initial_density(self, x0: Array) -> Array:
        """``log p(x_0)``, ``(..., D) -> (...)``."""
        return self.initial.log_prob(x0)

    def mean_transition(self, x: Array) -> Array:
        """``E[x_t | x_{t-1} = x]``, ``(..., D) -> (..., D)``."""
        return self.transition.mean(x)

    def sample_transition(self, key: Array, x: Array) -> Array:
        """A draw of ``x_t | x_{t-1} = x``, ``(..., D) -> (..., D)``."""
        return self.transition.sample(key, x)

    def log_transition_density(self, x_next: Array, x: Array) -> Array:
        """``log p(x_t = x_next | x_{t-1} = x)``, ``(..., D), (..., D) -> (...)``."""
        return self.transition.log_prob(x_next, x)

    def observe_mean(self, x: Array) -> Array:
        """``E[y_t | x_t = x]``, ``(..., D) -> (..., p)``."""
        return self.observation.mean(x)

    def sample_observation(self, key: Array, x: Array) -> Array:
        """A draw of ``y_t | x_t = x``, ``(..., D) -> (..., p)``."""
        return self.observation.sample(key, x)

    def log_likelihood(self, y: Array, x: Array) -> Array:
        """``log p(y_t = y | x_t = x)``, ``(..., p), (..., D) -> (...)``."""
        return self.observation.log_prob(y, x)

    def simulate(self, key: Array, n_steps: int, x0: Array | None = None) -> Trajectory:
        """Simulate ``x_0, x_1..x_T`` and ``y_1..y_T`` with ``T = n_steps``.

        All keys are split before the scan, so the result depends only on
        ``key`` (and ``x0``), not on how the scan is compiled.

        Args:
            key: PRNG key.
            n_steps: Number of transitions ``T`` (static).
            x0: Initial state ``(D,)``; drawn from ``initial`` if None.

        Returns:
            ``Trajectory`` with ``states`` ``(T, D)`` and ``observations`` ``(T, p)``.
        """
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
        k_init, k_steps = jax.random.split(key)
        x0 = self.sample_initial(k_init) if x0 is None else jnp.asarray(x0)
        check_event_shape(x0, (self.state_dim,), "x0")
        check_finite(x0, "x0")
        step_keys = jax.random.split(k_steps, (n_steps, 2))

        def step(x, keys):
            x_next = self.sample_transition(keys[0], x)
            return x_next, (x_next, self.sample_observation(keys[1], x_next))

        _, (states, observations) = jax.lax.scan(step, x0, step_keys)
        return Trajectory(initial=x0, states=states, observations=observations)
