"""Exact answers for linear-Gaussian state-space models.

A ``StateSpaceModel`` is linear-Gaussian when its transition and observation
are ``Additive`` with a ``Linear`` or ``Selector`` map and ``Gaussian`` (or
``Embedded`` Gaussian, or no) noise, and its initial law is ``Gaussian`` or
``PointMass``. ``Lagged`` transitions, ``Precomposed`` observations with a linear
map, and ``History`` initial laws built from such parts are accepted too, so
``delayed`` linear-Gaussian models are solved exactly. Affine
offsets (``Linear.offset`` and a noise ``loc``) are allowed. For such a model
the filtering and smoothing distributions are Gaussian and the Kalman
recursions compute them exactly.
Use float64 (``jax_enable_x64``) when comparing a method against the oracle.
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jax.scipy.linalg import cho_factor, cho_solve

from jax_da._validation import check_finite, is_concrete
from jax_da.conditional import Additive, Lagged, Precomposed
from jax_da.laws import Embedded, Gaussian, History, PointMass
from jax_da.maps import Linear, Selector
from jax_da.ssm import StateSpaceModel


@struct.dataclass
class FilterResult:
    """Kalman filter output for ``t = 1..T``.

    Attributes:
        means: ``E[x_t | y_1..y_t]``, shape ``(T, D)``.
        covs: ``Cov[x_t | y_1..y_t]``, shape ``(T, D, D)``.
        predicted_means: ``E[x_t | y_1..y_{t-1}]``, shape ``(T, D)``.
        predicted_covs: ``Cov[x_t | y_1..y_{t-1}]``, shape ``(T, D, D)``.
        log_evidence: ``log p(y_1..y_T)``, scalar.
    """

    means: Array
    covs: Array
    predicted_means: Array
    predicted_covs: Array
    log_evidence: Array


@struct.dataclass
class SmootherResult:
    """Rauch-Tung-Striebel smoother output for ``t = 1..T``.

    Attributes:
        means: ``E[x_t | y_1..y_T]``, shape ``(T, D)``.
        covs: ``Cov[x_t | y_1..y_T]``, shape ``(T, D, D)``.
    """

    means: Array
    covs: Array


def _linear(map, name):
    """``(matrix, offset)`` of a ``Linear`` or ``Selector`` map."""
    if not isinstance(map, (Linear, Selector)):
        raise ValueError(f"KalmanOracle needs the {name} to be Linear or Selector, got {type(map).__name__}")
    return map.matrix, jnp.broadcast_to(getattr(map, "offset", 0.0), (map.dim,))


def _is_gaussian(law) -> bool:
    return isinstance(law, Gaussian) or (isinstance(law, Embedded) and isinstance(law.law, Gaussian))


def _transition_params(law):
    """``(A, b, Q)``; a ``Lagged`` transition becomes its block companion form."""
    if isinstance(law, Lagged):
        A, b, Q = _transition_params(law.transition)
        D, n = law.block, law.dim
        A_big = jnp.zeros((n, n)).at[:D, :D].set(A).at[D:, : n - D].set(jnp.eye(n - D))
        return A_big, jnp.zeros(n).at[:D].set(b), jnp.zeros((n, n)).at[:D, :D].set(Q)
    return _affine_gaussian(law, "transition")


def _observation_params(law):
    """``(H, c, R)``; a ``Precomposed`` observation composes its linear map."""
    if isinstance(law, Precomposed):
        H, c, R = _observation_params(law.law)
        M, offset = _linear(law.map, "observation map")
        return H @ M, H @ offset + c, R
    return _affine_gaussian(law, "observation")


def _initial_moments(law):
    """``(m0, P0)``; a ``History`` gets the exact stacked moments of its trajectory."""
    if isinstance(law, History):
        m, P = _initial_moments(law.initial)
        A, b, Q = _transition_params(law.transition)
        means, covs = [m], [P]  # oldest first: x_0, x_1, ..., x_L
        for _ in range(law.lags):
            means.append(A @ means[-1] + b)
            covs.append(A @ covs[-1] @ A.T + Q)
        L, D = law.lags, law.block
        big = jnp.zeros((law.dim, law.dim))
        for j in range(L + 1):
            for k in range(j + 1):
                block = jnp.linalg.matrix_power(A, j - k) @ covs[k]  # Cov(x_j, x_k), j >= k
                rj, rk = (L - j) * D, (L - k) * D  # newest-first placement
                big = big.at[rj:rj + D, rk:rk + D].set(block).at[rk:rk + D, rj:rj + D].set(block.T)
        return jnp.concatenate(means[::-1]), big
    if not (_is_gaussian(law) or isinstance(law, PointMass)):
        raise ValueError(f"KalmanOracle needs a Gaussian, Embedded Gaussian, PointMass, or History initial law, "
                         f"got {type(law).__name__}")
    return jnp.broadcast_to(law.loc, (law.dim,)), law.cov()


def _affine_gaussian(law, name):
    """``(matrix, offset, cov)`` of an ``Additive`` law with a linear map and Gaussian (or no) noise."""
    if not isinstance(law, Additive):
        raise ValueError(f"KalmanOracle needs the {name} to be Additive with a Linear or Selector map")
    matrix, offset = _linear(law.map, f"{name} map")
    if law.noise is None:
        return matrix, offset, jnp.zeros((law.dim, law.dim))
    if not _is_gaussian(law.noise):
        raise ValueError(f"KalmanOracle needs Gaussian {name} noise, got {type(law.noise).__name__}")
    return matrix, offset + law.noise.loc, law.noise.cov()


@struct.dataclass
class KalmanOracle:
    """Exact filtering, smoothing, and evidence for a linear-Gaussian model.

    Build with ``KalmanOracle.from_ssm(ssm)``.

    The model is ``x_t = A x_{t-1} + b + eta_t``, ``y_t = H x_t + c + eps_t``,
    ``eta_t ~ N(0, Q)``, ``eps_t ~ N(0, R)``, ``x_0 ~ N(m0, P0)``.

    Attributes:
        A, b: Transition matrix ``(D, D)`` and offset ``(D,)``.
        H, c: Observation matrix ``(p, D)`` and offset ``(p,)``.
        Q: Model-error covariance ``(D, D)`` (zero if deterministic).
        R: Observation-noise covariance ``(p, p)``.
        m0: Initial mean ``(D,)``.
        P0: Initial covariance ``(D, D)`` (zero if ``x_0`` is known).
    """

    A: Array
    b: Array
    H: Array
    c: Array
    Q: Array
    R: Array
    m0: Array
    P0: Array

    @classmethod
    def from_ssm(cls, ssm: StateSpaceModel) -> "KalmanOracle":
        """Extract the linear-Gaussian parameters of ``ssm``.

        Raises:
            ValueError: If ``ssm`` is not linear-Gaussian.
        """
        A, b, Q = _transition_params(ssm.transition)
        H, c, R = _observation_params(ssm.observation)
        m0, P0 = _initial_moments(ssm.initial)
        return cls(A=A, b=b, H=H, c=c, Q=Q, R=R, m0=m0, P0=P0)

    def filter(self, observations: Array) -> FilterResult:
        """Kalman filter over ``y_1..y_T``, shape ``(T, p)``.

        Raises:
            ValueError: If ``observations`` is not ``(T, p)`` or (outside ``jit``) not finite.
        """
        p = self.H.shape[-2]
        if observations.ndim != 2 or observations.shape[-1] != p:
            raise ValueError(f"observations must have shape (T, {p}), got {tuple(observations.shape)}")
        check_finite(observations, "observations")
        A, b, H, c, Q, R = self.A, self.b, self.H, self.c, self.Q, self.R

        def step(carry, y):
            m, P, log_ev = carry
            m_pred = A @ m + b
            P_pred = A @ P @ A.T + Q
            S = H @ P_pred @ H.T + R
            chol = cho_factor(S, lower=True)
            innovation = y - H @ m_pred - c
            K = cho_solve(chol, H @ P_pred).T
            m_new = m_pred + K @ innovation
            P_new = P_pred - K @ S @ K.T
            P_new = 0.5 * (P_new + P_new.T)
            log_det = 2.0 * jnp.log(jnp.diag(chol[0])).sum()
            log_ev = log_ev - 0.5 * (innovation @ cho_solve(chol, innovation) + log_det + y.shape[0] * jnp.log(2 * jnp.pi))
            return (m_new, P_new, log_ev), (m_new, P_new, m_pred, P_pred)

        init = (self.m0, self.P0, jnp.zeros((), dtype=self.m0.dtype))
        (_, _, log_ev), (means, covs, pm, pc) = jax.lax.scan(step, init, observations)
        return FilterResult(means=means, covs=covs, predicted_means=pm, predicted_covs=pc, log_evidence=log_ev)

    def smooth(self, observations: Array) -> SmootherResult:
        """Rauch-Tung-Striebel smoother over ``y_1..y_T``, shape ``(T, p)``.

        Needs invertible predicted covariances. With no model error and a known
        (or degenerate) initial state they are singular; outside ``jit`` this
        raises instead of returning NaN.
        """
        f = self.filter(observations)
        A = self.A

        def step(carry, inputs):
            m_next_s, P_next_s = carry
            m, P, m_pred_next, P_pred_next = inputs
            G = jnp.linalg.solve(P_pred_next, A @ P).T
            m_s = m + G @ (m_next_s - m_pred_next)
            P_s = P + G @ (P_next_s - P_pred_next) @ G.T
            return (m_s, 0.5 * (P_s + P_s.T)), (m_s, P_s)

        inputs = (f.means[:-1], f.covs[:-1], f.predicted_means[1:], f.predicted_covs[1:])
        _, (means, covs) = jax.lax.scan(step, (f.means[-1], f.covs[-1]), inputs, reverse=True)
        result = SmootherResult(
            means=jnp.concatenate([means, f.means[-1:]]),
            covs=jnp.concatenate([covs, f.covs[-1:]]),
        )
        if is_concrete(result.covs) and not bool(jnp.all(jnp.isfinite(result.covs))):
            raise ValueError("smoother failed: a predicted covariance is singular (no model error and a "
                             "degenerate initial covariance?); add model error or initial uncertainty")
        return result
