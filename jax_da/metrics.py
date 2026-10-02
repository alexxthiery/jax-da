"""Scores of an ensemble forecast against the truth.

Conventions: an ensemble has shape ``(..., N, D)`` and the truth ``(..., D)``,
where ``...`` is usually time. Per-time scores return shape ``(...)``;
average them over the window you care about. Calibration summaries
(``spread_skill_ratio``, ``rank_histogram``) aggregate over all leading axes.
All functions are pure JAX and trace under ``jit`` and ``vmap``.
"""

import math

import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from jax_da._validation import check_finite, check_positive


def _check(ensemble: Array, truth: Array) -> None:
    if ensemble.ndim < 2 or ensemble.shape[:-2] + ensemble.shape[-1:] != truth.shape:
        raise ValueError(
            f"Expected ensemble (..., N, D) and truth (..., D) with matching ... and D, "
            f"got {tuple(ensemble.shape)} and {tuple(truth.shape)}.")
    check_finite(ensemble, "ensemble")
    check_finite(truth, "truth")


def rmse(ensemble: Array, truth: Array) -> Array:
    """Root mean square error of the ensemble mean over components, shape ``(...)``."""
    _check(ensemble, truth)
    return jnp.sqrt(jnp.mean((ensemble.mean(-2) - truth) ** 2, axis=-1))


def spread(ensemble: Array) -> Array:
    """``sqrt(mean_d Var_n x)`` with the unbiased (``ddof=1``) variance, shape ``(...)``."""
    if ensemble.ndim < 2 or ensemble.shape[-2] < 2:
        raise ValueError(f"Expected ensemble (..., N, D) with N >= 2, got {tuple(ensemble.shape)}.")
    check_finite(ensemble, "ensemble")
    return jnp.sqrt(jnp.mean(jnp.var(ensemble, axis=-2, ddof=1), axis=-1))


def crps(ensemble: Array, truth: Array, fair: bool = False) -> Array:
    """Continuous ranked probability score, averaged over components, shape ``(...)``.

    ``CRPS = E|X - y| - 1/2 E|X - X'|`` for ensemble members ``X, X'``. The
    default treats the ensemble as the forecast distribution; ``fair=True``
    divides the second term by ``N (N - 1)`` instead of ``N**2``, giving an
    unbiased estimate of the CRPS of the distribution the members are drawn
    from (Ferro 2014), so ensembles of different sizes compare fairly.
    Computed in ``O(N log N)`` by sorting.
    """
    _check(ensemble, truth)
    n = ensemble.shape[-2]
    if fair and n < 2:
        raise ValueError("fair CRPS needs at least two members")
    term1 = jnp.mean(jnp.abs(ensemble - truth[..., None, :]), axis=-2)
    # sum_{i,j} |x_i - x_j| = 2 sum_k (2k - n - 1) x_(k) for sorted members x_(1..n).
    weights = (2 * jnp.arange(1, n + 1) - n - 1).astype(ensemble.dtype)
    pair_sum = 2 * jnp.einsum("k,...kd->...d", weights, jnp.sort(ensemble, axis=-2))
    term2 = pair_sum / (2 * n * (n - 1) if fair else 2 * n * n)
    return jnp.mean(term1 - term2, axis=-1)


def crps_gaussian(mean: Array, std: Array, truth: Array) -> Array:
    """Closed-form CRPS of independent Gaussian forecasts, averaged over components, shape ``(...)``.

    ``CRPS(N(mu, s^2), y) = s [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]`` with
    ``z = (y - mu) / s`` (Gneiting et al. 2005).
    """
    if not (jnp.shape(mean) == jnp.shape(std) == jnp.shape(truth)):
        raise ValueError(f"mean, std, and truth must have the same shape, got "
                         f"{jnp.shape(mean)}, {jnp.shape(std)}, {jnp.shape(truth)}")
    check_positive(std, "std")
    check_finite(mean, "mean")
    check_finite(truth, "truth")
    z = (truth - mean) / std
    per = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
    return jnp.mean(per, axis=-1)


def energy_score(ensemble: Array, truth: Array) -> Array:
    """Energy score ``E||X - y|| - 1/2 E||X - X'||``, shape ``(...)``.

    The multivariate generalization of CRPS (Gneiting and Raftery 2007);
    it equals CRPS when ``D = 1``. Memory is ``O(N^2 D)`` per leading index.
    """
    _check(ensemble, truth)
    term1 = jnp.linalg.norm(ensemble - truth[..., None, :], axis=-1).mean(-1)
    diff = ensemble[..., :, None, :] - ensemble[..., None, :, :]
    term2 = jnp.sqrt(jnp.sum(diff ** 2, axis=-1)).mean((-1, -2))
    return term1 - 0.5 * term2


def coverage(ensemble: Array, truth: Array, level: float = 0.9) -> Array:
    """Fraction of components inside the central ``level`` interval of the ensemble, shape ``(...)``.

    Interval bounds are the empirical ``(1 - level) / 2`` and ``(1 + level) / 2``
    quantiles of the members, with linear interpolation.
    """
    _check(ensemble, truth)
    if not 0 < level < 1:
        raise ValueError(f"level must lie in (0, 1), got {level}")
    lower = jnp.quantile(ensemble, (1 - level) / 2, axis=-2)
    upper = jnp.quantile(ensemble, (1 + level) / 2, axis=-2)
    return jnp.mean((truth >= lower) & (truth <= upper), axis=-1)


def spread_skill_ratio(ensemble: Array, truth: Array) -> Array:
    """``sqrt((N + 1) / N * mean Var_n x / mean (mean_n x - y)^2)``, aggregated over all leading axes.

    Variance uses ``ddof=1``. With the ``(N + 1) / N`` factor a calibrated
    ensemble (truth exchangeable with the members) has ratio 1 for every
    ``N`` (Fortin et al. 2014). Below 1: under-dispersed; above 1: over-dispersed.
    """
    _check(ensemble, truth)
    n = ensemble.shape[-2]
    var = jnp.mean(jnp.var(ensemble, axis=-2, ddof=1))
    mse = jnp.mean((ensemble.mean(-2) - truth) ** 2)
    return jnp.sqrt((n + 1) / n * var / mse)


def rank_histogram(ensemble: Array, truth: Array) -> Array:
    """Counts of the truth's rank among the members, shape ``(N + 1,)``, over all leading axes and components.

    Rank ``k`` means ``k`` members lie strictly below the truth. A calibrated
    ensemble gives a flat histogram; U-shaped means under-dispersed, dome-shaped
    over-dispersed, sloped biased. Ties have probability zero for continuous
    ensembles and count as "not below".
    """
    _check(ensemble, truth)
    n = ensemble.shape[-2]
    ranks = jnp.sum(ensemble < truth[..., None, :], axis=-2)
    return jnp.bincount(ranks.ravel(), length=n + 1)
