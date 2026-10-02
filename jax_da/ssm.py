"""State-space model: dynamics plus additive model error, observation operator, and noise.

    x_0 = m_0 + xi,            xi ~ initial_noise   (xi = 0 if initial_noise is None)
    x_t = M(x_{t-1}) + eta_t,  eta_t ~ model_error  (eta_t = 0 if model_error is None)
    y_t = h(x_t) + eps_t,      eps_t ~ obs_noise,   t = 1, ..., T

``M`` is ``dynamics.flow`` over one assimilation interval and ``h`` is
``obs_operator.apply``. There is no observation of ``x_0``.
"""

import jax
from flax import struct
from jax import Array

from jax_da._validation import check_event_shape


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

    Attributes:
        dynamics: Object with ``dim``, ``geometry``, and ``flow(x)`` over one interval.
        obs_operator: Object with ``in_dim``, ``dim``, and ``apply(x)``.
        obs_noise: Noise law of dimension ``obs_operator.dim``.
        initial_mean: ``m_0``, shape ``(D,)``.
        initial_noise: Noise law of dimension ``D``, or None for a known ``x_0 = m_0``.
        model_error: Noise law of dimension ``D``, or None for deterministic dynamics.
    """

    dynamics: object
    obs_operator: object
    obs_noise: object
    initial_mean: Array
    initial_noise: object = None
    model_error: object = None

    def __post_init__(self):
        D, p = self.dynamics.dim, self.obs_operator.dim
        if self.obs_operator.in_dim != D:
            raise ValueError(f"obs_operator.in_dim={self.obs_operator.in_dim} != dynamics.dim={D}")
        if self.obs_noise.dim != p:
            raise ValueError(f"obs_noise.dim={self.obs_noise.dim} != obs_operator.dim={p}")
        for name in ("initial_noise", "model_error"):
            law = getattr(self, name)
            if law is not None and law.dim != D:
                raise ValueError(f"{name}.dim={law.dim} != dynamics.dim={D}")
        shape = getattr(self.initial_mean, "shape", None)
        if shape is not None and tuple(shape) != (D,):
            raise ValueError(f"initial_mean must have shape ({D},), got {tuple(shape)}")

    @property
    def state_dim(self) -> int:
        return self.dynamics.dim

    @property
    def obs_dim(self) -> int:
        return self.obs_operator.dim

    @property
    def geometry(self):
        return self.dynamics.geometry

    def sample_initial(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        """Draws of ``x_0``, shape ``shape + (D,)``."""
        if self.initial_noise is None:
            return jax.numpy.broadcast_to(self.initial_mean, shape + (self.state_dim,))
        return self.initial_mean + self.initial_noise.sample(key, shape)

    def log_initial_density(self, x0: Array) -> Array:
        """``log p(x_0)``, ``(..., D) -> (...)``.

        Raises:
            ValueError: If ``initial_noise`` is None (``x_0`` is a point mass).
        """
        check_event_shape(x0, (self.state_dim,), "x0")
        if self.initial_noise is None:
            raise ValueError("x_0 is known exactly (initial_noise is None); it has no density")
        return self.initial_noise.log_prob(x0 - self.initial_mean)

    def mean_transition(self, x: Array) -> Array:
        """``M(x)``, the deterministic part of the transition, ``(..., D) -> (..., D)``."""
        return self.dynamics.flow(x)

    def sample_transition(self, key: Array, x: Array) -> Array:
        """A draw of ``x_{t+1} | x_t = x``, ``(..., D) -> (..., D)``."""
        mean = self.mean_transition(x)
        if self.model_error is None:
            return mean
        return mean + self.model_error.sample(key, mean.shape[:-1])

    def log_transition_density(self, x_next: Array, x: Array) -> Array:
        """``log p(x_{t+1} = x_next | x_t = x)``, ``(..., D), (..., D) -> (...)``.

        Raises:
            ValueError: If ``model_error`` is None (the transition is a point mass).
        """
        check_event_shape(x_next, (self.state_dim,), "x_next")
        if self.model_error is None:
            raise ValueError("deterministic dynamics (model_error is None) have no transition density")
        return self.model_error.log_prob(x_next - self.mean_transition(x))

    def observe_mean(self, x: Array) -> Array:
        """``h(x)``, ``(..., D) -> (..., p)``."""
        return self.obs_operator.apply(x)

    def sample_observation(self, key: Array, x: Array) -> Array:
        """A draw of ``y_t | x_t = x``, ``(..., D) -> (..., p)``."""
        mean = self.observe_mean(x)
        return mean + self.obs_noise.sample(key, mean.shape[:-1])

    def log_likelihood(self, y: Array, x: Array) -> Array:
        """``log p(y_t = y | x_t = x)``, ``(..., p), (..., D) -> (...)``."""
        check_event_shape(y, (self.obs_dim,), "y")
        return self.obs_noise.log_prob(y - self.observe_mean(x))

    def simulate(self, key: Array, n_steps: int, x0: Array | None = None) -> Trajectory:
        """Simulate ``x_0, x_1..x_T`` and ``y_1..y_T`` with ``T = n_steps``.

        All keys are split before the scan, so the result depends only on
        ``key`` (and ``x0``), not on how the scan is compiled.

        Args:
            key: PRNG key.
            n_steps: Number of transitions ``T`` (static).
            x0: Initial state ``(D,)``; drawn with ``sample_initial`` if None.

        Returns:
            ``Trajectory`` with ``states`` ``(T, D)`` and ``observations`` ``(T, p)``.
        """
        k_init, k_steps = jax.random.split(key)
        if x0 is None:
            x0 = self.sample_initial(k_init)
        check_event_shape(x0, (self.state_dim,), "x0")
        step_keys = jax.random.split(k_steps, (n_steps, 2))

        def step(x, keys):
            x_next = self.sample_transition(keys[0], x)
            return x_next, (x_next, self.sample_observation(keys[1], x_next))

        _, (states, observations) = jax.lax.scan(step, x0, step_keys)
        return Trajectory(initial=x0, states=states, observations=observations)
