"""Additive noise laws for model error and observation error.

Every law is zero-mean on R^d with:

- ``sample(key, shape=())`` returning draws of shape ``shape + (dim,)``;
- ``log_prob(e)`` mapping ``(..., dim)`` to ``(...)``;
- ``variance()`` (per component, shape ``(dim,)``) and ``cov()`` (``(dim, dim)``),
  which raise ``NotImplementedError`` when the second moment does not exist.

Heavy-tailed laws use the textbook ``scale`` parameter; ``with_std`` builds one
with a given standard deviation, to compare against a Gaussian of equal spread.
"""

import math

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jax.scipy.special import gammaln, logsumexp

from jax_da._validation import check_event_shape, is_concrete


def _check_positive(name, value):
    if is_concrete(value) and not bool(jnp.all(jnp.asarray(value) > 0)):
        raise ValueError(f"{name} must be positive, got {value}")


@struct.dataclass
class Gaussian:
    """Zero-mean Gaussian, iid-scalar, diagonal, or full covariance.

    Build with ``Gaussian.isotropic``, ``Gaussian.diagonal``, or ``Gaussian.full``.
    For iid or diagonal noise ``scale_tril`` is None and ``std`` holds the
    per-component standard deviation (shape ``()`` or ``(dim,)``); for full
    covariance ``std`` is None and ``scale_tril`` is the Cholesky factor.

    Attributes:
        dim: Dimension.
        std: Per-component standard deviation, or None.
        scale_tril: Lower Cholesky factor of the covariance, ``(dim, dim)``, or None.
    """

    dim: int = struct.field(pytree_node=False)
    std: Array | None = None
    scale_tril: Array | None = None

    def __post_init__(self):
        if (self.std is None) == (self.scale_tril is None):
            raise ValueError("Gaussian needs exactly one of std or scale_tril")
        if self.std is not None:
            _check_positive("std", self.std)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        return cls(dim=dim, std=jnp.asarray(std, dtype=float))

    @classmethod
    def diagonal(cls, std: Array) -> "Gaussian":
        std = jnp.asarray(std, dtype=float)
        return cls(dim=int(std.shape[0]), std=std)

    @classmethod
    def full(cls, cov: Array) -> "Gaussian":
        cov = jnp.asarray(cov, dtype=float)
        return cls(dim=int(cov.shape[0]), scale_tril=jnp.linalg.cholesky(cov))

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        z = jax.random.normal(key, shape + (self.dim,))
        if self.scale_tril is None:
            return self.std * z
        return z @ self.scale_tril.T

    def log_prob(self, e: Array) -> Array:
        check_event_shape(e, (self.dim,), "e")
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
    """

    dim: int = struct.field(pytree_node=False)
    df: float = 4.0
    scale: Array = 1.0

    def __post_init__(self):
        _check_positive("df", self.df)
        _check_positive("scale", self.scale)

    @classmethod
    def with_std(cls, dim: int, df: float, std: float) -> "StudentT":
        if df <= 2:
            raise ValueError(f"Student-t with df={df} has no finite std")
        return cls(dim=dim, df=df, scale=std * math.sqrt((df - 2) / df))

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.scale * jax.random.t(key, self.df, shape + (self.dim,))

    def log_prob(self, e: Array) -> Array:
        check_event_shape(e, (self.dim,), "e")
        scale = jnp.broadcast_to(self.scale, (self.dim,))
        df = self.df
        z = e / scale
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
    """

    dim: int = struct.field(pytree_node=False)
    scale: Array = 1.0

    def __post_init__(self):
        _check_positive("scale", self.scale)

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.scale * jax.random.cauchy(key, shape + (self.dim,))

    def log_prob(self, e: Array) -> Array:
        check_event_shape(e, (self.dim,), "e")
        scale = jnp.broadcast_to(self.scale, (self.dim,))
        return (-jnp.log(jnp.pi * scale) - jnp.log1p((e / scale) ** 2)).sum(-1)

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
    """

    dim: int = struct.field(pytree_node=False)
    scale: Array = 1.0

    def __post_init__(self):
        _check_positive("scale", self.scale)

    @classmethod
    def with_std(cls, dim: int, std: float) -> "Laplace":
        return cls(dim=dim, scale=std / math.sqrt(2.0))

    def sample(self, key: Array, shape: tuple[int, ...] = ()) -> Array:
        return self.scale * jax.random.laplace(key, shape + (self.dim,))

    def log_prob(self, e: Array) -> Array:
        check_event_shape(e, (self.dim,), "e")
        scale = jnp.broadcast_to(self.scale, (self.dim,))
        return (-jnp.log(2 * scale) - jnp.abs(e) / scale).sum(-1)

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
    """

    dim: int = struct.field(pytree_node=False)
    std: Array = 1.0
    outlier_prob: float = 0.05
    outlier_scale: float = 10.0

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
        return scale * jax.random.normal(k_z, full)

    def log_prob(self, e: Array) -> Array:
        check_event_shape(e, (self.dim,), "e")
        std = jnp.broadcast_to(self.std, (self.dim,))

        def normal(s):
            return -0.5 * (e / s) ** 2 - jnp.log(s) - 0.5 * math.log(2 * math.pi)

        per = logsumexp(
            jnp.stack([normal(std), normal(self.outlier_scale * std)]),
            b=jnp.array([1 - self.outlier_prob, self.outlier_prob]).reshape((2,) + (1,) * e.ndim),
            axis=0)
        return per.sum(-1)

    def variance(self) -> Array:
        factor = 1 - self.outlier_prob + self.outlier_prob * self.outlier_scale ** 2
        return jnp.broadcast_to(self.std ** 2 * factor, (self.dim,))

    def cov(self) -> Array:
        return jnp.diag(self.variance())
