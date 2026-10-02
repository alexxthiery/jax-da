"""Conditional laws against SciPy densities and sampling moments."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import jax_da as jd

L = jd.Linear(jnp.array([[0.5, 0.0], [0.2, -0.4], [0.0, 1.0]]), offset=jnp.array([0.1, -0.3, 0.2]))
X = jnp.array([[0.3, -1.2], [2.0, 0.5]])


def test_additive_with_located_heavy_tailed_noise():
    law = jd.Additive(L, jd.StudentT(3, df=5.0, scale=0.7, loc=0.4))
    out = jnp.array([[0.0, 1.0, -2.0], [3.0, 0.5, 0.1]])
    ref = stats.t(5.0, loc=np.asarray(L(X)) + 0.4, scale=0.7).logpdf(np.asarray(out)).sum(-1)
    np.testing.assert_allclose(law.log_prob(out, X), ref, rtol=1e-10)
    np.testing.assert_allclose(law.mean(X), L(X) + 0.4)
    draws = law.sample(jax.random.PRNGKey(0), jnp.broadcast_to(X[0], (200_000, 2)))
    np.testing.assert_allclose(draws.mean(0), L(X[0]) + 0.4, atol=0.01)


@pytest.mark.parametrize("noise, ref", [
    (jd.Gaussian.isotropic(3, 1.0), lambda s: stats.norm(0, s)),
    (jd.StudentT(3, df=4.0, scale=1.0), lambda s: stats.t(4.0, scale=s)),
])
def test_multiplicative_density_is_the_scaled_noise_density(noise, ref):
    law = jd.Multiplicative(L, noise)
    out = jnp.array([[0.4, -1.0, 2.0], [0.1, 0.3, -0.5]])
    scale = np.exp(np.asarray(L(X)))
    np.testing.assert_allclose(law.log_prob(out, X), ref(scale).logpdf(np.asarray(out)).sum(-1), rtol=1e-10)


def test_multiplicative_sampling_scale_and_extreme_log_scales_stay_finite():
    law = jd.Multiplicative(L, jd.Gaussian.isotropic(3, 1.0))
    draws = law.sample(jax.random.PRNGKey(1), jnp.broadcast_to(X[1], (200_000, 2)))
    np.testing.assert_allclose(draws.std(0), np.exp(np.asarray(L(X[1]))), rtol=0.01)
    tiny = jd.Multiplicative(jd.Linear(jnp.eye(1), offset=-60.0), jd.Gaussian.isotropic(1, 1.0))
    for x in (jnp.array([[0.0]]), jnp.array([[-30.0]])):
        value = tiny.log_prob(jnp.array([[0.1]]), x)
        assert not bool(jnp.isnan(value).any())
        assert not bool(jnp.isnan(jax.grad(lambda v: tiny.log_prob(jnp.array([[0.1]]), v).sum())(x)).any())


def test_poisson_matches_scipy_and_has_poisson_moments():
    law = jd.Poisson(L)
    counts = jnp.array([[0.0, 2.0, 1.0], [5.0, 0.0, 3.0]])
    rate = np.exp(np.asarray(L(X)))
    np.testing.assert_allclose(law.log_prob(counts, X), stats.poisson(rate).logpmf(np.asarray(counts)).sum(-1), rtol=1e-10)
    draws = np.asarray(law.sample(jax.random.PRNGKey(2), jnp.broadcast_to(X[1], (200_000, 2))))
    assert np.all(draws == np.round(draws)) and np.all(draws >= 0)
    np.testing.assert_allclose(draws.mean(0), rate[1], rtol=0.02)
    np.testing.assert_allclose(draws.var(0), rate[1], rtol=0.03)
    np.testing.assert_allclose(law.mean(X), rate, rtol=1e-12)


def test_dimension_mismatches_rejected():
    with pytest.raises(ValueError, match="noise.dim"):
        jd.Multiplicative(L, jd.Gaussian.isotropic(2, 1.0))
    with pytest.raises(ValueError):
        jd.Poisson(L).log_prob(jnp.zeros((2, 4)), X)
