"""Noise laws: log_prob against SciPy, sampling moments, covariance contracts."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from jax_da.noise import Cauchy, Gaussian, GaussianMixture, Laplace, StudentT

D = 3
LAWS = {
    "gaussian_iso": Gaussian.isotropic(D, 0.7),
    "gaussian_diag": Gaussian.diagonal(jnp.array([0.5, 1.0, 2.0])),
    "gaussian_full": Gaussian.full(jnp.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.4], [0.0, 0.4, 0.5]])),
    "student_t": StudentT(D, df=5.0, scale=0.8),
    "cauchy": Cauchy(D, scale=0.6),
    "laplace": Laplace(D, scale=0.9),
    "mixture": GaussianMixture(D, std=0.5, outlier_prob=0.1, outlier_scale=6.0),
}


def scipy_logpdf(name, law, e):
    e = np.asarray(e)
    if name == "gaussian_full":
        return stats.multivariate_normal(np.zeros(D), np.asarray(law.cov())).logpdf(e)
    if name.startswith("gaussian"):
        return stats.norm(0, np.broadcast_to(np.asarray(law.std), (D,))).logpdf(e).sum(-1)
    if name == "student_t":
        return stats.t(law.df, scale=law.scale).logpdf(e).sum(-1)
    if name == "cauchy":
        return stats.cauchy(scale=law.scale).logpdf(e).sum(-1)
    if name == "laplace":
        return stats.laplace(scale=law.scale).logpdf(e).sum(-1)
    p, s = law.outlier_prob, law.std
    return np.log((1 - p) * stats.norm(0, s).pdf(e) + p * stats.norm(0, law.outlier_scale * s).pdf(e)).sum(-1)


@pytest.mark.parametrize("name", LAWS)
def test_log_prob_matches_scipy_on_batches(name):
    law = LAWS[name]
    e = jax.random.normal(jax.random.PRNGKey(0), (4, 5, D)) * 1.5
    np.testing.assert_allclose(law.log_prob(e), scipy_logpdf(name, law, e), rtol=1e-10, atol=1e-10)
    assert law.log_prob(e).shape == (4, 5)


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
    x = law.sample(jax.random.PRNGKey(1), (200_000,))
    assert x.shape == (200_000, D)
    np.testing.assert_allclose(x.mean(0), 0.0, atol=0.05)
    np.testing.assert_allclose(np.cov(np.asarray(x).T), law.cov(), rtol=0.1, atol=0.02)


def test_cauchy_sample_median_and_iqr():
    x = np.asarray(Cauchy(1, scale=0.6).sample(jax.random.PRNGKey(2), (200_000,)))[:, 0]
    np.testing.assert_allclose(np.median(x), 0.0, atol=0.01)
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
    e = jnp.ones((7, D))
    np.testing.assert_allclose(jax.jit(lambda l, v: l.log_prob(v))(law, e), law.log_prob(e))
    df_grad = jax.grad(lambda df: StudentT(D, df=df, scale=0.8).log_prob(e[0]))(5.0)
    assert np.isfinite(df_grad)
    scales = jnp.array([0.5, 1.0])
    out = jax.vmap(lambda s: Laplace(D, scale=s).log_prob(e[0]))(scales)
    assert out.shape == (2,)
