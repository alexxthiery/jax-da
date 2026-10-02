"""Scores against definitions, closed forms, and calibrated ensembles."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

from jax_da import metrics


def calibrated(key, n_time, n_members, dim):
    """Truth and members drawn iid from the same per-time Gaussian: exchangeable by construction."""
    k1, k2, k3 = jax.random.split(key, 3)
    center = 3 * jax.random.normal(k1, (n_time, 1, dim))
    members = center + jax.random.normal(k2, (n_time, n_members, dim))
    truth = center[:, 0] + jax.random.normal(k3, (n_time, dim))
    return members, truth


def test_rmse_and_spread_hand_values():
    ens = jnp.array([[[1.0, 2.0], [3.0, 6.0]]])  # one time, N=2, D=2
    truth = jnp.array([[0.0, 0.0]])
    np.testing.assert_allclose(metrics.rmse(ens, truth), [np.sqrt((4 + 16) / 2)])
    np.testing.assert_allclose(metrics.spread(ens), [np.sqrt((2 + 8) / 2)])


@pytest.mark.parametrize("fair", [False, True])
def test_crps_matches_pairwise_definition(fair):
    ens = jax.random.normal(jax.random.PRNGKey(0), (3, 7, 4))
    truth = jax.random.normal(jax.random.PRNGKey(1), (3, 4))
    x, y = np.asarray(ens), np.asarray(truth)
    term1 = np.abs(x - y[:, None]).mean(1)
    pair = np.abs(x[:, :, None] - x[:, None, :]).sum((1, 2))
    term2 = pair / (2 * 7 * (6 if fair else 7))
    np.testing.assert_allclose(metrics.crps(ens, truth, fair=fair), (term1 - term2).mean(-1), rtol=1e-12)


def test_crps_gaussian_matches_integral_definition():
    mu, s, y = 0.3, 1.7, -0.8
    integrand = lambda z: (stats.norm(mu, s).cdf(z) - (z >= y)) ** 2
    ref = integrate.quad(integrand, -np.inf, y)[0] + integrate.quad(integrand, y, np.inf)[0]
    assert float(metrics.crps_gaussian(jnp.array([mu]), jnp.array([s]), jnp.array([y]))) == pytest.approx(ref, rel=1e-8)


def test_large_ensemble_crps_converges_to_closed_form():
    # Monte Carlo standard error of E|X - y| is about 1.2 / sqrt(2e5) = 0.003; allow three.
    ens = 0.3 + 1.7 * jax.random.normal(jax.random.PRNGKey(2), (200_000, 1))
    assert float(metrics.crps(ens, jnp.array([-0.8]))) == pytest.approx(
        float(metrics.crps_gaussian(jnp.array([0.3]), jnp.array([1.7]), jnp.array([-0.8]))), abs=0.009)


def test_fair_crps_is_unbiased_for_small_ensembles():
    # Expected over draws: fair CRPS of N=4 members equals the Gaussian CRPS; the standard one is biased up.
    ens = jax.random.normal(jax.random.PRNGKey(3), (100_000, 4, 1))
    truth = jnp.full((100_000, 1), 0.5)
    exact = float(metrics.crps_gaussian(jnp.zeros(1), jnp.ones(1), jnp.array([0.5])))
    assert float(metrics.crps(ens, truth, fair=True).mean()) == pytest.approx(exact, rel=5e-3)
    assert float(metrics.crps(ens, truth).mean()) > exact * 1.05


def test_energy_score_matches_pairwise_euclidean_definition():
    ens = jax.random.normal(jax.random.PRNGKey(9), (2, 6, 3))
    truth = jax.random.normal(jax.random.PRNGKey(10), (2, 3))
    x, y = np.asarray(ens), np.asarray(truth)
    term1 = np.linalg.norm(x - y[:, None], axis=-1).mean(-1)
    term2 = np.linalg.norm(x[:, :, None] - x[:, None, :], axis=-1).mean((-1, -2))
    np.testing.assert_allclose(metrics.energy_score(ens, truth), term1 - 0.5 * term2, rtol=1e-12)


def test_energy_score_equals_crps_in_one_dimension():
    ens = jax.random.normal(jax.random.PRNGKey(4), (5, 9, 1))
    truth = jax.random.normal(jax.random.PRNGKey(5), (5, 1))
    np.testing.assert_allclose(metrics.energy_score(ens, truth), metrics.crps(ens, truth), rtol=1e-12)


@pytest.mark.parametrize("n_members", [3, 10, 40])
def test_calibrated_ensembles_score_as_calibrated(n_members):
    ens, truth = calibrated(jax.random.PRNGKey(n_members), 20_000, n_members, 2)
    assert float(metrics.spread_skill_ratio(ens, truth)) == pytest.approx(1.0, abs=0.02)
    counts = np.asarray(metrics.rank_histogram(ens, truth))
    assert counts.sum() == 40_000
    assert stats.chisquare(counts).pvalue > 1e-3


def test_coverage_is_nominal_for_large_calibrated_ensembles():
    ens, truth = calibrated(jax.random.PRNGKey(6), 4000, 400, 3)
    for level in (0.5, 0.9):
        assert float(metrics.coverage(ens, truth, level).mean()) == pytest.approx(level, abs=0.015)


def test_underdispersed_ensemble_detected():
    ens, truth = calibrated(jax.random.PRNGKey(7), 5000, 10, 2)
    narrow = ens.mean(-2, keepdims=True) + 0.3 * (ens - ens.mean(-2, keepdims=True))
    assert float(metrics.spread_skill_ratio(narrow, truth)) < 0.5
    counts = np.asarray(metrics.rank_histogram(narrow, truth))
    assert counts[0] + counts[-1] > 3 * counts[1:-1].mean() * 2


def test_shape_mismatch_rejected_and_jit_traces():
    with pytest.raises(ValueError):
        metrics.crps(jnp.zeros((5, 3, 2)), jnp.zeros((5, 3)))
    ens, truth = calibrated(jax.random.PRNGKey(8), 4, 6, 3)
    np.testing.assert_allclose(jax.jit(metrics.crps)(ens, truth), metrics.crps(ens, truth))
