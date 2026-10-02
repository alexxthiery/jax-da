"""Noise laws: log_prob against SciPy, sampling moments, covariance contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from jax_da.laws import Cauchy, Gaussian, GaussianMixture, Laplace, StudentT

D = 3
LOC = jnp.array([0.4, -1.2, 2.0])  # nonzero, distinct locations so a dropped loc cannot cancel
LAWS = {
    "gaussian_iso": Gaussian.isotropic(D, 0.7, loc=0.3),
    "gaussian_diag": Gaussian.diagonal(jnp.array([0.5, 1.0, 2.0]), loc=LOC),
    "gaussian_full": Gaussian.full(jnp.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.4], [0.0, 0.4, 0.5]]), loc=LOC),
    "student_t": StudentT(D, df=5.0, scale=0.8, loc=LOC),
    "cauchy": Cauchy(D, scale=0.6, loc=LOC),
    "laplace": Laplace(D, scale=0.9, loc=LOC),
    "mixture": GaussianMixture(D, std=0.5, outlier_prob=0.1, outlier_scale=6.0, loc=LOC),
}


def scipy_logpdf(name, law, x):
    x, loc = np.asarray(x), np.asarray(law.loc)
    if name == "gaussian_full":
        return stats.multivariate_normal(loc, np.asarray(law.cov())).logpdf(x)
    if name.startswith("gaussian"):
        return stats.norm(loc, np.asarray(law.std)).logpdf(x).sum(-1)
    if name == "student_t":
        return stats.t(law.df, loc=loc, scale=np.asarray(law.scale)).logpdf(x).sum(-1)
    if name == "cauchy":
        return stats.cauchy(loc=loc, scale=np.asarray(law.scale)).logpdf(x).sum(-1)
    if name == "laplace":
        return stats.laplace(loc=loc, scale=np.asarray(law.scale)).logpdf(x).sum(-1)
    p, s = law.outlier_prob, np.asarray(law.std)
    density = (1 - p) * stats.norm(loc, s).pdf(x) + p * stats.norm(loc, law.outlier_scale * s).pdf(x)
    return np.log(density).sum(-1)


@pytest.mark.parametrize("name", LAWS)
def test_log_prob_matches_scipy_on_batches(name):
    law = LAWS[name]
    x = LOC + jax.random.normal(jax.random.PRNGKey(0), (4, 5, D)) * 1.5
    np.testing.assert_allclose(law.log_prob(x), scipy_logpdf(name, law, x), rtol=1e-10, atol=1e-10)
    assert law.log_prob(x).shape == (4, 5)


LAWS_1D = {
    "gaussian": Gaussian.isotropic(1, 0.7),
    "student_t": StudentT(1, df=5.0, scale=0.8),
    "cauchy": Cauchy(1, scale=0.6),
    "laplace": Laplace(1, scale=0.9),
    "mixture": GaussianMixture(1, std=0.5, outlier_prob=0.1, outlier_scale=6.0),
}


@pytest.mark.parametrize("name", LAWS_1D)
def test_one_dimensional_density_integrates_to_one(name):
    law = LAWS_1D[name]
    total, _ = integrate.quad(lambda v: float(jnp.exp(law.log_prob(jnp.array([v])))), -np.inf, np.inf, limit=200)
    assert abs(total - 1.0) < 1e-6


@pytest.mark.parametrize("name", [n for n in LAWS if n != "cauchy"])
def test_sample_moments_match_cov(name):
    law = LAWS[name]
    n = 200_000
    x = law.sample(jax.random.PRNGKey(1), (n,))
    assert x.shape == (n, D)
    # The sample mean has standard error sqrt(var / n); allow five.
    np.testing.assert_allclose(x.mean(0), law.loc, atol=5 * float(np.sqrt(np.max(law.variance()) / n)))
    np.testing.assert_allclose(np.cov(np.asarray(x).T), law.cov(), rtol=0.1, atol=0.02)


def test_cauchy_sample_median_and_iqr():
    x = np.asarray(Cauchy(1, scale=0.6, loc=-0.7).sample(jax.random.PRNGKey(2), (200_000,)))[:, 0]
    np.testing.assert_allclose(np.median(x), -0.7, atol=0.01)
    np.testing.assert_allclose(np.subtract(*np.percentile(x, [75, 25])), 2 * 0.6, rtol=0.02)


def test_undefined_second_moments_raise():
    with pytest.raises(NotImplementedError):
        Cauchy(D).cov()
    with pytest.raises(NotImplementedError):
        StudentT(D, df=2.0).variance()


def test_with_std_constructors_set_the_standard_deviation():
    np.testing.assert_allclose(StudentT.with_std(D, 5.0, 0.3).variance(), 0.09)
    np.testing.assert_allclose(Laplace.with_std(D, 0.3).variance(), 0.09)


def test_invalid_parameters_and_shapes_rejected():
    with pytest.raises(ValueError):
        Gaussian.isotropic(D, -1.0)
    with pytest.raises(ValueError):
        GaussianMixture(D, outlier_prob=1.0)
    with pytest.raises(ValueError):
        LAWS["laplace"].log_prob(jnp.zeros((2, D + 1)))


def test_laws_trace_under_jit_vmap_grad():
    law = LAWS["student_t"]
    x = jnp.ones((7, D))
    np.testing.assert_allclose(jax.jit(lambda l, v: l.log_prob(v))(law, x), law.log_prob(x))
    # Gradient with respect to a parameter equals a central finite difference.
    f = lambda df: StudentT(D, df=df, scale=0.8).log_prob(x[0])
    eps = 1e-6
    np.testing.assert_allclose(jax.grad(f)(5.0), (f(5.0 + eps) - f(5.0 - eps)) / (2 * eps), rtol=1e-6)
    # vmap over a parameter equals evaluating each value separately.
    scales = jnp.array([0.5, 1.0, 2.0])
    batched = jax.vmap(lambda s: Laplace(D, scale=s).log_prob(x[0]))(scales)
    np.testing.assert_allclose(batched, [Laplace(D, scale=float(s)).log_prob(x[0]) for s in scales], rtol=1e-12)
