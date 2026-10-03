"""Probability laws on R^d: initial laws, and the noise inside conditional laws.

Every law has a location ``loc`` (default 0) and:

- ``sample(key, shape=())`` returning draws of shape ``shape + (dim,)``;
- ``log_prob(x)`` mapping ``(..., dim)`` to ``(...)``;
- ``variance()`` (per component, shape ``(dim,)``) and ``cov()`` (``(dim, dim)``),
  which raise ``NotImplementedError`` when the second moment does not exist.

``loc`` is the mean for laws that have one (all but ``Cauchy``, where it is the
median). Heavy-tailed laws use the textbook ``scale`` parameter; ``with_std``
builds one with a given standard deviation, to compare against a Gaussian of
equal spread.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array
from jax.scipy.special import gammaln

from jax_da._validation import (
    check_event_shape,
    check_finite,
    check_positive,
    is_concrete,
    is_numeric,
    per_component,
    placeholder_dims,
)


def _check_loc(law) -> None:
    per_component(law, "loc", law.dim)
    check_finite(law.loc, "loc")


@struct.dataclass
class Gaussian:
    """Zero-mean Gaussian, iid-scalar, diagonal, or full covariance.

    Build with ``Gaussian.isotropic``, ``Gaussian.diagonal``, or ``Gaussian.full``,
    each taking an optional ``loc``.
    For iid or diagonal noise ``scale_tril`` is None and ``std`` holds the
    per-component standard deviation (normalized to shape ``(dim,)``); for full
    covariance ``std`` is None and ``scale_tril`` is the Cholesky factor.

    Attributes:
        dim: Dimension.
        std: Per-component standard deviation, or None.
        scale_tril: Lower Cholesky factor of the covariance, ``(dim, dim)``, or None.
        loc: Mean; a scalar is broadcast to ``(dim,)``.
    """

    dim: int = struct.field(pytree_node=False)
    std: Array | None = None
    scale_tril: Array | None = None
    loc: Array = 0.0

    def __post_init__(self):
        # Both None happens when JAX rebuilds the pytree with None leaves (an in_axes tree);
        # build through isotropic, diagonal, or full, which always set exactly one.
        if self.std is not None and self.scale_tril is not None:
            raise ValueError("Gaussian takes std or scale_tril, not both")
        if self.std is not None:
            per_component(self, "std", self.dim)
            check_positive(self.std, "std")
        if is_numeric(self.scale_tril):
            if np.shape(self.scale_tril)[-2:] != (self.dim, self.dim):
                raise ValueError(f"scale_tril must have trailing shape ({self.dim}, {self.dim}), "
                                 f"got {np.shape(self.scale_tril)}")
            if is_concrete(self.scale_tril) and not bool(np.all(np.isfinite(np.asarray(self.scale_tril)))):
                raise ValueError("covariance must be finite and symmetric positive definite; "
                                 "its Cholesky factorization failed")
        _check_loc(self)

    @classmethod
    def isotropic(cls, dim: int, std: float, loc=0.0) -> "Gaussian":
        if np.ndim(std) != 0:
            raise ValueError(f"isotropic std must be a scalar, got shape {np.shape(std)}; use Gaussian.diagonal")
        return cls(dim=dim, std=jnp.asarray(std, dtype=float), loc=loc)

    @classmethod
    def diagonal(cls, std: Array, loc=0.0) -> "Gaussian":
        if np.ndim(std) != 1:
            raise ValueError(f"diagonal std must be a vector of shape (d,), got shape {np.shape(std)}")
        std = jnp.asarray(std, dtype=float)
        return cls(dim=int(std.shape[0]), std=std, loc=loc)

    @classmethod
    def full(cls, cov: Array, loc=0.0) -> "Gaussian":
        cov = jnp.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"cov must be a square matrix, got shape {cov.shape}")
        if is_concrete(cov):
            c = np.asarray(cov)
            if not np.all(np.isfinite(c)):
                raise ValueError("cov must be finite and symmetric positive definite; it contains NaN or inf")
            if not np.allclose(c, c.T, rtol=1e-8, atol=1e-12 * max(1.0, np.abs(c).max())):
                raise ValueError(f"cov must be symmetric; max |cov - cov.T| = {np.abs(c - c.T).max():.3g}")
        return cls(dim=int(cov.shape[0]), scale_tril=jnp.linalg.cholesky(cov), loc=loc)

    def _require_parameters(self) -> None:
        if self.std is None and self.scale_tril is None:
            raise ValueError("Gaussian has neither std nor scale_tril; build it with "
                             "Gaussian.isotropic, diagonal, or full")

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        self._require_parameters()
        z = jax.random.normal(key, shape + (self.dim,))
        if self.scale_tril is None:
            return self.loc + self.std * z
        return self.loc + z @ self.scale_tril.T

    def log_prob(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
        self._require_parameters()
        e = x - self.loc
        if self.scale_tril is None:
            std = jnp.broadcast_to(self.std, (self.dim,))
            return (-0.5 * (e / std) ** 2 - jnp.log(std)).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        flat = e.reshape(-1, self.dim)
        white = jax.scipy.linalg.solve_triangular(self.scale_tril, flat.T, lower=True).T.reshape(e.shape)
        log_det = jnp.log(jnp.abs(jnp.diag(self.scale_tril))).sum()
        return -0.5 * (white ** 2).sum(-1) - log_det - 0.5 * self.dim * math.log(2 * math.pi)

    def variance(self) -> Array:
        if self.scale_tril is None:
            return jnp.broadcast_to(self.std ** 2, (self.dim,))
        return (self.scale_tril ** 2).sum(-1)

    def cov(self) -> Array:
        if self.scale_tril is None:
            return jnp.diag(self.variance())
        return self.scale_tril @ self.scale_tril.T


@struct.dataclass
class StudentT:
    """Independent Student-t components with ``df`` degrees of freedom and scale ``scale``.

    Variance per component is ``scale**2 * df / (df - 2)`` for ``df > 2`` and
    undefined otherwise.

    Attributes:
        dim: Dimension.
        df: Degrees of freedom, positive.
        scale: Scale, scalar or ``(dim,)``.
        loc: Location (the mean when ``df > 1``), scalar or ``(dim,)``.
    """

    dim: int = struct.field(pytree_node=False)
    df: float = 4.0
    scale: Array = 1.0
    loc: Array = 0.0

    def __post_init__(self):
        check_positive(self.df, "df")
        per_component(self, "scale", self.dim)
        check_positive(self.scale, "scale")
        _check_loc(self)

    @classmethod
    def with_std(cls, dim: int, df: float, std: float, loc=0.0) -> "StudentT":
        if df <= 2:
            raise ValueError(f"Student-t with df={df} has no finite std")
        return cls(dim=dim, df=df, scale=std * math.sqrt((df - 2) / df), loc=loc)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.loc + self.scale * jax.random.t(key, self.df, shape + (self.dim,))

    def log_prob(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
        scale = jnp.broadcast_to(self.scale, (self.dim,))
        df = self.df
        z = (x - self.loc) / scale
        per = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * jnp.log(df * jnp.pi)
               - jnp.log(scale) - (df + 1) / 2 * jnp.log1p(z ** 2 / df))
        return per.sum(-1)

    def variance(self) -> Array:
        if is_concrete(self.df) and self.df <= 2:
            raise NotImplementedError(f"Student-t with df={self.df} has infinite variance")
        return jnp.broadcast_to(self.scale ** 2 * self.df / (self.df - 2), (self.dim,))

    def cov(self) -> Array:
        return jnp.diag(self.variance())


@struct.dataclass
class Cauchy:
    """Independent Cauchy components (Student-t with one degree of freedom).

    Attributes:
        dim: Dimension.
        scale: Scale, scalar or ``(dim,)``.
        loc: Location (the median), scalar or ``(dim,)``.
    """

    dim: int = struct.field(pytree_node=False)
    scale: Array = 1.0
    loc: Array = 0.0

    def __post_init__(self):
        per_component(self, "scale", self.dim)
        check_positive(self.scale, "scale")
        _check_loc(self)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.loc + self.scale * jax.random.cauchy(key, shape + (self.dim,))

    def log_prob(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
        scale = jnp.broadcast_to(self.scale, (self.dim,))
        return (-jnp.log(jnp.pi * scale) - jnp.log1p(((x - self.loc) / scale) ** 2)).sum(-1)

    def variance(self) -> Array:
        raise NotImplementedError("Cauchy noise has no variance")

    def cov(self) -> Array:
        raise NotImplementedError("Cauchy noise has no covariance")


@struct.dataclass
class Laplace:
    """Independent Laplace (double-exponential) components with scale ``scale``.

    Variance per component is ``2 * scale**2``.

    Attributes:
        dim: Dimension.
        scale: Scale, scalar or ``(dim,)``.
        loc: Mean, scalar or ``(dim,)``.
    """

    dim: int = struct.field(pytree_node=False)
    scale: Array = 1.0
    loc: Array = 0.0

    def __post_init__(self):
        per_component(self, "scale", self.dim)
        check_positive(self.scale, "scale")
        _check_loc(self)

    @classmethod
    def with_std(cls, dim: int, std: float, loc=0.0) -> "Laplace":
        return cls(dim=dim, scale=std / math.sqrt(2.0), loc=loc)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.loc + self.scale * jax.random.laplace(key, shape + (self.dim,))

    def log_prob(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
        scale = jnp.broadcast_to(self.scale, (self.dim,))
        return (-jnp.log(2 * scale) - jnp.abs(x - self.loc) / scale).sum(-1)

    def variance(self) -> Array:
        return jnp.broadcast_to(2 * self.scale ** 2, (self.dim,))

    def cov(self) -> Array:
        return jnp.diag(self.variance())


@struct.dataclass
class GaussianMixture:
    """Contaminated Gaussian, independently per component (Tukey's outlier model).

    Each component is ``N(0, std**2)`` with probability ``1 - outlier_prob`` and
    ``N(0, (outlier_scale * std)**2)`` with probability ``outlier_prob``.

    Attributes:
        dim: Dimension.
        std: Standard deviation of the main component, scalar or ``(dim,)``.
        outlier_prob: Contamination probability in [0, 1).
        outlier_scale: Std multiplier of the outlier component, > 0.
        loc: Mean of both components, scalar or ``(dim,)``.
    """

    dim: int = struct.field(pytree_node=False)
    std: Array = 1.0
    outlier_prob: float = 0.05
    outlier_scale: float = 10.0
    loc: Array = 0.0

    def __post_init__(self):
        per_component(self, "std", self.dim)
        check_positive(self.std, "std")
        check_positive(self.outlier_scale, "outlier_scale")
        if is_concrete(self.outlier_prob) and not 0 <= self.outlier_prob < 1:
            raise ValueError(f"outlier_prob must lie in [0, 1), got {self.outlier_prob}")
        _check_loc(self)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        k_mask, k_z = jax.random.split(key)
        full = shape + (self.dim,)
        outlier = jax.random.uniform(k_mask, full) < self.outlier_prob
        scale = self.std * jnp.where(outlier, self.outlier_scale, 1.0)
        return self.loc + scale * jax.random.normal(k_z, full)

    def log_prob(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
        std = jnp.broadcast_to(self.std, (self.dim,))
        e = x - self.loc

        def normal(s):
            return -0.5 * (e / s) ** 2 - jnp.log(s) - 0.5 * math.log(2 * math.pi)

        p = self.outlier_prob
        per = jnp.logaddexp(jnp.log1p(-p) + normal(std), jnp.log(p) + normal(self.outlier_scale * std))
        return per.sum(-1)

    def variance(self) -> Array:
        factor = 1 - self.outlier_prob + self.outlier_prob * self.outlier_scale ** 2
        return jnp.broadcast_to(self.std ** 2 * factor, (self.dim,))

    def cov(self) -> Array:
        return jnp.diag(self.variance())


@struct.dataclass
class PointMass:
    """All mass at ``value``: a known initial state.

    It has no density, so ``log_prob`` raises; its covariance is zero, so the
    Kalman oracle treats it as an exactly known state. For a deterministic
    transition use ``Additive(map)`` with no noise instead.

    Attributes:
        value: The point, shape ``(dim,)``.
    """

    value: Array

    def __post_init__(self):
        if is_numeric(self.value) and np.ndim(self.value) < 1:
            raise ValueError(f"PointMass value must have shape (..., dim), got shape {np.shape(self.value)}")
        check_finite(self.value, "PointMass value")

    @property
    def dim(self) -> int:
        return int(self.value.shape[-1])

    @property
    def loc(self) -> Array:
        return self.value

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return jnp.broadcast_to(self.value, shape + (self.dim,))

    def log_prob(self, x: Array) -> Array:
        raise ValueError("a point mass has no density")

    def variance(self) -> Array:
        return jnp.zeros(self.dim)

    def cov(self) -> Array:
        return jnp.zeros((self.dim, self.dim))


@struct.dataclass
class Embedded:
    """The law of ``matrix @ z`` with ``z ~ law``: noise confined to a subspace of ``R^dim``.

    Use it for process noise that enters only some directions, for example the
    velocity of a position-velocity model or a few Fourier modes of a PDE. When
    ``matrix`` has fewer columns than rows the law is singular: it has no density
    on ``R^dim``, so ``log_prob`` raises, while sampling, ``loc``, and ``cov`` work
    and the Kalman oracle accepts it when ``law`` is Gaussian. Singular noise is
    declared explicitly this way; ``Gaussian.full`` still rejects a singular
    covariance.

    Attributes:
        law: Law of ``z``, dimension ``r``.
        matrix: Shape ``(dim, r)``.
    """

    law: object
    matrix: Array

    def __post_init__(self):
        if not is_numeric(self.matrix):
            return
        if np.ndim(self.matrix) < 2:
            raise ValueError(f"matrix must have shape (dim, r), got shape {np.shape(self.matrix)}")
        check_finite(self.matrix, "matrix")
        inner_dim = getattr(self.law, "dim", None)
        if isinstance(inner_dim, int) and np.shape(self.matrix)[-1] != inner_dim:
            raise ValueError(f"matrix must have {inner_dim} columns to embed a law of dimension {inner_dim}, "
                             f"got shape {np.shape(self.matrix)}")

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[-2])

    @property
    def loc(self) -> Array:
        return jnp.broadcast_to(self.law.loc, (self.law.dim,)) @ self.matrix.T

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.law.sample(key, shape) @ self.matrix.T

    def log_prob(self, x: Array) -> Array:
        raise ValueError("an Embedded law is confined to a subspace and has no density on R^dim")

    def variance(self) -> Array:
        return jnp.diag(self.cov())

    def cov(self) -> Array:
        return self.matrix @ self.law.cov() @ self.matrix.T


@struct.dataclass
class History:
    """Law of the first ``lags + 1`` states of a trajectory, stacked newest first: ``(x_L, ..., x_1, x_0)``.

    ``x_0 ~ initial`` and ``x_k | x_{k-1} ~ transition``. It is the initial law of a
    model on stacked histories (see ``jax_da.delayed``). ``log_prob`` is the chain
    density ``log p(x_0) + sum_k log p(x_k | x_{k-1})`` and raises when a factor has
    no density.

    Attributes:
        initial: Law of ``x_0`` on ``R^D``.
        transition: Conditional law on ``R^D``.
        lags: ``L >= 1`` (static).
    """

    initial: object
    transition: object
    lags: int = struct.field(pytree_node=False)

    def __post_init__(self):
        if not isinstance(self.lags, int) or self.lags < 1:
            raise ValueError(f"lags must be a positive integer, got {self.lags!r}")
        if not placeholder_dims(self.initial, self.transition) and \
                not self.transition.in_dim == self.transition.dim == self.initial.dim:
            raise ValueError(f"transition must map R^D to R^D with D = initial.dim = {self.initial.dim}, "
                             f"got in_dim={self.transition.in_dim}, dim={self.transition.dim}")

    @property
    def block(self) -> int:
        return self.initial.dim

    @property
    def dim(self) -> int:
        return (self.lags + 1) * self.block

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        keys = jax.random.split(key, self.lags + 1)
        states = [self.initial.sample(keys[0], shape)]
        for k in range(1, self.lags + 1):
            states.append(self.transition.sample(keys[k], states[-1]))
        return jnp.concatenate(states[::-1], axis=-1)

    def log_prob(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
        D = self.block
        oldest_first = [x[..., (self.lags - k) * D:(self.lags - k + 1) * D] for k in range(self.lags + 1)]
        total = self.initial.log_prob(oldest_first[0])
        for k in range(1, self.lags + 1):
            total = total + self.transition.log_prob(oldest_first[k], oldest_first[k - 1])
        return total

    def variance(self) -> Array:
        raise NotImplementedError("History has no closed-form moments in general; "
                                  "KalmanOracle computes them for linear-Gaussian parts")

    def cov(self) -> Array:
        return self.variance()

