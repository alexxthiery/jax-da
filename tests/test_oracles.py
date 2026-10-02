"""KalmanOracle against brute-force conditioning of the joint Gaussian."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from jax_da.dynamics.linear import LinearDynamics
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.noise import Gaussian, Laplace
from jax_da.observations import Linear, Selector
from jax_da.oracles import KalmanOracle
from jax_da.ssm import StateSpaceModel

T, D, P = 4, 3, 2


def model(model_error=True):
    return StateSpaceModel(
        dynamics=LinearDynamics(jnp.array([[0.9, 0.2, 0.0], [-0.1, 0.8, 0.3], [0.0, 0.1, 0.95]])),
        obs_operator=Linear(jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])),
        obs_noise=Gaussian.full(jnp.array([[0.3, 0.05], [0.05, 0.2]])),
        initial_mean=jnp.array([0.5, -1.0, 2.0]),
        initial_noise=Gaussian.diagonal(jnp.array([1.0, 0.5, 2.0])),
        model_error=Gaussian.isotropic(D, 0.4) if model_error else None,
    )


def joint_gaussian(o):
    """Mean and covariance of (x_1..x_T, y_1..y_T), built from x_t = A^t x_0 + sum A^{t-s} eta_s."""
    A, H, Q, R, m0, P0 = (np.asarray(v) for v in (o.A, o.H, o.Q, o.R, o.m0, o.P0))
    powers = [np.linalg.matrix_power(A, k) for k in range(T + 1)]
    mx = np.concatenate([powers[t] @ m0 for t in range(1, T + 1)])
    Cx = np.zeros((T * D, T * D))
    for t in range(1, T + 1):
        for s in range(1, T + 1):
            block = powers[t] @ P0 @ powers[s].T
            for k in range(1, min(t, s) + 1):
                block = block + powers[t - k] @ Q @ powers[s - k].T
            Cx[(t - 1) * D:t * D, (s - 1) * D:s * D] = block
    Hb = np.kron(np.eye(T), H)
    Cxy, Cy = Cx @ Hb.T, Hb @ Cx @ Hb.T + np.kron(np.eye(T), R)
    return mx, Hb @ mx, Cx, Cxy, Cy


def condition(mx, my, Cx, Cxy, Cy, y, rows):
    gain = Cxy[rows][:, :len(my)] @ np.linalg.inv(Cy)
    return mx[rows] + gain @ (y - my), Cx[np.ix_(rows, rows)] - gain @ Cxy[rows].T


@pytest.mark.parametrize("model_error", [True, False])
def test_filter_smoother_and_evidence_match_joint_conditioning(model_error):
    ssm = model(model_error)
    oracle = KalmanOracle.from_ssm(ssm)
    y = np.asarray(ssm.simulate(jax.random.PRNGKey(0), T).observations)
    mx, my, Cx, Cxy, Cy = joint_gaussian(oracle)
    f, s = oracle.filter(jnp.asarray(y)), oracle.smooth(jnp.asarray(y))
    for t in range(T):
        rows = np.arange(t * D, (t + 1) * D)
        obs = np.arange((t + 1) * P)  # filtering conditions on y_1..y_{t+1} only
        m_f, C_f = condition(mx, my[obs], Cx, Cxy[:, obs], Cy[np.ix_(obs, obs)], y.ravel()[obs], rows)
        np.testing.assert_allclose(f.means[t], m_f, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(f.covs[t], C_f, rtol=1e-9, atol=1e-10)
        m_s, C_s = condition(mx, my, Cx, Cxy, Cy, y.ravel(), rows)
        np.testing.assert_allclose(s.means[t], m_s, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(s.covs[t], C_s, rtol=1e-8, atol=1e-10)
    ref = stats.multivariate_normal(my, Cy).logpdf(y.ravel())
    assert float(f.log_evidence) == pytest.approx(ref, rel=1e-10)


def test_selector_operator_supported():
    ssm = StateSpaceModel(LinearDynamics.damped_rotation(4), Selector.every(4, 2), Gaussian.isotropic(2, 0.5),
                          jnp.zeros(4), Gaussian.isotropic(4, 1.0), Gaussian.isotropic(4, 0.1))
    y = ssm.simulate(jax.random.PRNGKey(1), 10).observations
    f = KalmanOracle.from_ssm(ssm).filter(y)
    assert f.means.shape == (10, 4) and np.all(np.isfinite(f.covs))


def test_non_linear_gaussian_models_rejected():
    with pytest.raises(ValueError, match="linear"):
        KalmanOracle.from_ssm(StateSpaceModel(Lorenz96(dim=8), Selector.every(8, 2), Gaussian.isotropic(4, 1.0), jnp.zeros(8)))
    bad = model().replace(obs_noise=Laplace(P, scale=0.5))
    with pytest.raises(ValueError, match="Gaussian"):
        KalmanOracle.from_ssm(bad)
