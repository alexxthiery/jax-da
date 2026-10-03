"""Conditional laws ``p(out | x)``: transitions ``x_t | x_{t-1}`` and observations ``y_t | x_t``.

Every conditional law has ``in_dim``, ``dim``, and, for ``x`` of shape
``(..., in_dim)`` and ``out`` of shape ``(..., dim)``:

- ``sample(key, x)`` draws ``out``;
- ``log_prob(out, x)`` evaluates ``log p(out | x)``, shape ``(...)``;
- ``mean(x)`` is ``E[out | x]`` (the location for laws without a mean).

``Additive`` is the classical data assimilation form and exposes its map and
noise, which Kalman-type methods and ``KalmanOracle`` use. Anything with these
members works as a conditional law; ``docs/laws.md`` has a template.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array
from jax.scipy.special import gammaln

from jax_da._validation import check_event_shape, is_concrete, placeholder_dims, require_map


@struct.dataclass
class Additive:
    """``out = map(x) + noise``, or ``out = map(x)`` when ``noise`` is None.

    With ``noise=None`` the law is a point mass: ``sample`` is deterministic and
    ``log_prob`` raises.

    Attributes:
        map: Map ``(..., in_dim) -> (..., dim)``.
        noise: Law of dimension ``map.dim``, or None.
    """

    map: object
    noise: object = None

    def __post_init__(self):
        require_map(self.map, "Additive map")
        if self.noise is not None and not placeholder_dims(self.map, self.noise) and self.noise.dim != self.map.dim:
            raise ValueError(f"noise.dim={self.noise.dim} != map.dim={self.map.dim}")

    @property
    def in_dim(self) -> int:
        return self.map.in_dim

    @property
    def dim(self) -> int:
        return self.map.dim

    def mean(self, x: Array) -> Array:
        return self.map(x) if self.noise is None else self.map(x) + self.noise.loc

    def sample(self, key: Array, x: Array) -> Array:
        center = self.map(x)
        if self.noise is None:
            return center
        return center + self.noise.sample(key, center.shape[:-1])

    def log_prob(self, out: Array, x: Array) -> Array:
        check_event_shape(out, (self.dim,), "out")
        if self.noise is None:
            raise ValueError("a deterministic map (noise=None) has no density")
        return self.noise.log_prob(out - self.map(x))


@struct.dataclass
class Multiplicative:
    """``out = exp(log_scale(x)) * noise``, componentwise: state-dependent scale.

    The density follows by change of variables:
    ``log p(out | x) = noise.log_prob(out / s) - sum(log s)`` with ``s = exp(log_scale(x))``.
    With ``noise`` a standard Gaussian this is ``N(0, diag(s**2))``; the
    stochastic volatility observation ``y = beta exp(x / 2) eps`` is
    ``Multiplicative(Linear(I / 2, offset=log(beta)), Gaussian.isotropic(d, 1.0))``.

    Attributes:
        log_scale: Map ``(..., in_dim) -> (..., dim)`` giving ``log s``.
        noise: Law of dimension ``log_scale.dim``.
    """

    log_scale: object
    noise: object

    def __post_init__(self):
        require_map(self.log_scale, "Multiplicative log_scale")
        if not placeholder_dims(self.log_scale, self.noise) and self.noise.dim != self.log_scale.dim:
            raise ValueError(f"noise.dim={self.noise.dim} != log_scale.dim={self.log_scale.dim}")

    @property
    def in_dim(self) -> int:
        return self.log_scale.in_dim

    @property
    def dim(self) -> int:
        return self.log_scale.dim

    def mean(self, x: Array) -> Array:
        return jnp.exp(self.log_scale(x)) * self.noise.loc

    def sample(self, key: Array, x: Array) -> Array:
        log_s = self.log_scale(x)
        return jnp.exp(log_s) * self.noise.sample(key, log_s.shape[:-1])

    def log_prob(self, out: Array, x: Array) -> Array:
        check_event_shape(out, (self.dim,), "out")
        # log_s is used directly (never exp then log) so tiny scales cannot produce log(0).
        log_s = self.log_scale(x)
        return self.noise.log_prob(out * jnp.exp(-log_s)) - log_s.sum(-1)


@struct.dataclass
class Poisson:
    """Independent counts ``out_i ~ Poisson(exp(log_rate(x)_i))``.

    Counts are returned as floats so they stack with other observation arrays.

    Attributes:
        log_rate: Map ``(..., in_dim) -> (..., dim)``.
    """

    log_rate: object

    def __post_init__(self):
        require_map(self.log_rate, "Poisson log_rate")

    @property
    def in_dim(self) -> int:
        return self.log_rate.in_dim

    @property
    def dim(self) -> int:
        return self.log_rate.dim

    def mean(self, x: Array) -> Array:
        return jnp.exp(self.log_rate(x))

    def sample(self, key: Array, x: Array) -> Array:
        rate = jnp.exp(self.log_rate(x))
        return jax.random.poisson(key, rate).astype(rate.dtype)

    def log_prob(self, out: Array, x: Array) -> Array:
        check_event_shape(out, (self.dim,), "out")
        if is_concrete(out):
            counts = np.asarray(out)
            if not (np.all(counts >= 0) and np.all(counts == np.round(counts))):
                raise ValueError("Poisson counts must be nonnegative integers")
        log_rate = self.log_rate(x)
        return (out * log_rate - jnp.exp(log_rate) - gammaln(out + 1.0)).sum(-1)


@struct.dataclass
class Precomposed:
    """``out | x`` distributed as ``law`` given ``map(x)``: point any conditional law at a function of the state.

    For example ``Precomposed(observation, Selector(D, block))`` observes one block of a
    stacked state with an existing observation law.

    Attributes:
        law: Conditional law with ``in_dim == map.dim``.
        map: Map ``(..., in_dim) -> (..., law.in_dim)``.
    """

    law: object
    map: object

    def __post_init__(self):
        require_map(self.map, "Precomposed map")
        if not placeholder_dims(self.law, self.map) and self.law.in_dim != self.map.dim:
            raise ValueError(f"law.in_dim={self.law.in_dim} != map.dim={self.map.dim}")

    @property
    def in_dim(self) -> int:
        return self.map.in_dim

    @property
    def dim(self) -> int:
        return self.law.dim

    def mean(self, x: Array) -> Array:
        return self.law.mean(self.map(x))

    def sample(self, key: Array, x: Array) -> Array:
        return self.law.sample(key, self.map(x))

    def log_prob(self, out: Array, x: Array) -> Array:
        return self.law.log_prob(out, self.map(x))


@struct.dataclass
class Lagged:
    """Transition of the stacked history ``u_t = (x_t, x_{t-1}, ..., x_{t-lags})``, newest block first.

    The newest block is drawn from ``transition`` given ``x_{t-1}``; the other blocks
    shift by one. ``log_prob`` is the base transition density of the newest block when
    the shifted blocks agree exactly, and ``-inf`` otherwise (the shift is deterministic).

    Attributes:
        transition: Conditional law on ``R^D`` with ``in_dim == dim == D``.
        lags: Number of past states kept, ``L >= 1`` (static).
    """

    transition: object
    lags: int = struct.field(pytree_node=False)

    def __post_init__(self):
        if not isinstance(self.lags, int) or self.lags < 1:
            raise ValueError(f"lags must be a positive integer, got {self.lags!r}")
        if not placeholder_dims(self.transition) and self.transition.in_dim != self.transition.dim:
            raise ValueError(f"transition must map R^D to R^D, got in_dim={self.transition.in_dim}, "
                             f"dim={self.transition.dim}")

    @property
    def block(self) -> int:
        return self.transition.dim

    @property
    def in_dim(self) -> int:
        return (self.lags + 1) * self.block

    @property
    def dim(self) -> int:
        return self.in_dim

    def _shifted(self, x: Array) -> Array:
        return x[..., : self.lags * self.block]

    def mean(self, x: Array) -> Array:
        check_event_shape(x, (self.in_dim,))
        return jnp.concatenate([self.transition.mean(x[..., : self.block]), self._shifted(x)], axis=-1)

    def sample(self, key: Array, x: Array) -> Array:
        check_event_shape(x, (self.in_dim,))
        return jnp.concatenate([self.transition.sample(key, x[..., : self.block]), self._shifted(x)], axis=-1)

    def log_prob(self, out: Array, x: Array) -> Array:
        check_event_shape(out, (self.dim,), "out")
        check_event_shape(x, (self.in_dim,))
        consistent = jnp.all(out[..., self.block:] == self._shifted(x), axis=-1)
        density = self.transition.log_prob(out[..., : self.block], x[..., : self.block])
        return jnp.where(consistent, density, -jnp.inf)

