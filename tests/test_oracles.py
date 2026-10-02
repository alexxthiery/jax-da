"""KalmanOracle against brute-force conditioning of the joint Gaussian."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from jax_da.conditional import Additive
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.laws import Gaussian, Laplace, PointMass
from jax_da.maps import Linear, Selector
from jax_da.oracles import KalmanOracle
from jax_da.ssm import StateSpaceModel

T, D, P = 4, 3, 2
# Raw model constants; the brute-force reference uses these, never the oracle's extracted parameters.
A = np.array([[0.9, 0.2, 0.0], [-0.1, 0.8, 0.3], [0.0, 0.1, 0.95]])
B_MAP, B_NOISE = np.array([0.3, -0.2, 0.1]), np.array([0.05, 0.0, -0.1])  # map offset and noise loc
H = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]])
C = np.array([1.0, -0.5])
Q = np.array([[0.16, 0.06, 0.0], [0.06, 0.2, -0.05], [0.0, -0.05, 0.1]])
R = np.array([[0.3, 0.05], [0.05, 0.2]])
M0 = np.array([0.5, -1.0, 2.0])
P0 = np.array([[1.0, 0.3, -0.2], [0.3, 0.5, 0.1], [-0.2, 0.1, 2.0]])


def model(model_error=True):
    # Full covariances everywhere, so cross-covariance terms of Q, R, and P0 are exercised.
    return StateSpaceModel(
        initial=Gaussian.full(jnp.asarray(P0), loc=jnp.asarray(M0)),
        transition=Additive(Linear(jnp.asarray(A), offset=jnp.asarray(B_MAP)),
                            Gaussian.full(jnp.asarray(Q), loc=jnp.asarray(B_NOISE)) if model_error else None),
        observation=Additive(Linear(jnp.asarray(H), offset=jnp.asarray(C)), Gaussian.full(jnp.asarray(R))),
    )


def joint_gaussian(model_error):
    """Mean and covariance of (x_1..x_T, y_1..y_T) from x_t = A^t x_0 + sum A^{t-s} (b + eta_s)."""
    Qm, b = (Q, B_MAP + B_NOISE) if model_error else (np.zeros((D, D)), B_MAP)
    powers = [np.linalg.matrix_power(A, k) for k in range(T + 1)]
    means, m = [], M0
    for _ in range(T):
        m = A @ m + b
        means.append(m)
    mx = np.concatenate(means)
    Cx = np.zeros((T * D, T * D))
    for t in range(1, T + 1):
        for s in range(1, T + 1):
            block = powers[t] @ P0 @ powers[s].T
            for k in range(1, min(t, s) + 1):
                block = block + powers[t - k] @ Qm @ powers[s - k].T
            Cx[(t - 1) * D:t * D, (s - 1) * D:s * D] = block
    Hb = np.kron(np.eye(T), H)
    Cxy, Cy = Cx @ Hb.T, Hb @ Cx @ Hb.T + np.kron(np.eye(T), R)
    return mx, Hb @ mx + np.tile(C, T), Cx, Cxy, Cy


def condition(mx, my, Cx, Cxy, Cy, y, rows):
    gain = Cxy[rows][:, :len(my)] @ np.linalg.inv(Cy)
    return mx[rows] + gain @ (y - my), Cx[np.ix_(rows, rows)] - gain @ Cxy[rows].T


@pytest.mark.parametrize("model_error", [True, False])
def test_filter_smoother_and_evidence_match_joint_conditioning(model_error):
    ssm = model(model_error)
    oracle = KalmanOracle.from_ssm(ssm)
    y = np.asarray(ssm.simulate(jax.random.PRNGKey(0), T).observations)
    mx, my, Cx, Cxy, Cy = joint_gaussian(model_error)
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


def test_selector_and_point_mass_supported():
    ssm = StateSpaceModel(PointMass(jnp.ones(4)), Additive(Linear.damped_rotation(4), Gaussian.isotropic(4, 0.1)),
                          Additive(Selector.every(4, 2), Gaussian.isotropic(2, 0.5)))
    f = KalmanOracle.from_ssm(ssm).filter(ssm.simulate(jax.random.PRNGKey(1), 10).observations)
    assert f.means.shape == (10, 4) and np.all(np.isfinite(f.covs))


def test_non_linear_gaussian_models_rejected():
    observe = Additive(Selector.every(8, 2), Gaussian.isotropic(4, 1.0))
    with pytest.raises(ValueError, match="Linear or Selector"):
        KalmanOracle.from_ssm(StateSpaceModel(Gaussian.isotropic(8, 1.0), Additive(Lorenz96(dim=8)), observe))
    with pytest.raises(ValueError, match="Gaussian"):
        KalmanOracle.from_ssm(model().replace(observation=Additive(Linear(jnp.asarray(H)), Laplace(P, scale=0.5))))
