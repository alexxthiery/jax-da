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
from flax import struct
from jax import Array
from jax.scipy.special import gammaln

from jax_da._validation import check_event_shape, is_concrete


def _check_positive(name, value):
    if is_concrete(value) and not bool(jnp.all(jnp.asarray(value) > 0)):
        raise ValueError(f"{name} must be positive, got {value}")


@struct.dataclass
class Gaussian:
    """Zero-mean Gaussian, iid-scalar, diagonal, or full covariance.

    Build with ``Gaussian.isotropic``, ``Gaussian.diagonal``, or ``Gaussian.full``,
    each taking an optional ``loc``.
    For iid or diagonal noise ``scale_tril`` is None and ``std`` holds the
    per-component standard deviation (shape ``()`` or ``(dim,)``); for full
    covariance ``std`` is None and ``scale_tril`` is the Cholesky factor.

    Attributes:
        dim: Dimension.
        std: Per-component standard deviation, or None.
        scale_tril: Lower Cholesky factor of the covariance, ``(dim, dim)``, or None.
        loc: Mean, scalar or ``(dim,)``.
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
            _check_positive("std", self.std)

    @classmethod
    def isotropic(cls, dim: int, std: float, loc=0.0) -> "Gaussian":
        return cls(dim=dim, std=jnp.asarray(std, dtype=float), loc=jnp.asarray(loc, dtype=float))

    @classmethod
    def diagonal(cls, std: Array, loc=0.0) -> "Gaussian":
        std = jnp.asarray(std, dtype=float)
        return cls(dim=int(std.shape[0]), std=std, loc=jnp.asarray(loc, dtype=float))

    @classmethod
    def full(cls, cov: Array, loc=0.0) -> "Gaussian":
        cov = jnp.asarray(cov, dtype=float)
        return cls(dim=int(cov.shape[0]), scale_tril=jnp.linalg.cholesky(cov), loc=jnp.asarray(loc, dtype=float))

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        z = jax.random.normal(key, shape + (self.dim,))
        if self.scale_tril is None:
            return self.loc + self.std * z
        return self.loc + z @ self.scale_tril.T

    def log_prob(self, x: Array) -> Array:
        check_event_shape(x, (self.dim,))
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
        _check_positive("df", self.df)
        _check_positive("scale", self.scale)

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
        _check_positive("scale", self.scale)

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
        _check_positive("scale", self.scale)

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
        _check_positive("std", self.std)
        _check_positive("outlier_scale", self.outlier_scale)
        if is_concrete(self.outlier_prob) and not 0 <= self.outlier_prob < 1:
            raise ValueError(f"outlier_prob must lie in [0, 1), got {self.outlier_prob}")

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
