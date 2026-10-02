"""Exact answers for linear-Gaussian state-space models.

A ``StateSpaceModel`` is linear-Gaussian when its transition and observation
are ``Additive`` with a ``Linear`` or ``Selector`` map and ``Gaussian`` noise
(or no noise), and its initial law is ``Gaussian`` or ``PointMass``. Affine
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

from jax_da.conditional import Additive
from jax_da.laws import Gaussian, PointMass
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


def _affine_gaussian(law, name):
    """``(matrix, offset, cov)`` of an ``Additive`` law with a linear map and Gaussian (or no) noise."""
    if not isinstance(law, Additive) or not isinstance(law.map, (Linear, Selector)):
        raise ValueError(f"KalmanOracle needs the {name} to be Additive with a Linear or Selector map")
    offset = jnp.broadcast_to(getattr(law.map, "offset", 0.0), (law.dim,))
    if law.noise is None:
        return law.map.matrix, offset, jnp.zeros((law.dim, law.dim))
    if not isinstance(law.noise, Gaussian):
        raise ValueError(f"KalmanOracle needs Gaussian {name} noise, got {type(law.noise).__name__}")
    return law.map.matrix, offset + law.noise.loc, law.noise.cov()


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
        A, b, Q = _affine_gaussian(ssm.transition, "transition")
        H, c, R = _affine_gaussian(ssm.observation, "observation")
        if not isinstance(ssm.initial, (Gaussian, PointMass)):
            raise ValueError(f"KalmanOracle needs a Gaussian or PointMass initial law, got {type(ssm.initial).__name__}")
        m0 = jnp.broadcast_to(ssm.initial.loc, (ssm.state_dim,))
        return cls(A=A, b=b, H=H, c=c, Q=Q, R=R, m0=m0, P0=ssm.initial.cov())

    def filter(self, observations: Array) -> FilterResult:
        """Kalman filter over ``y_1..y_T``, shape ``(T, p)``."""
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
        """Rauch-Tung-Striebel smoother over ``y_1..y_T``, shape ``(T, p)``."""
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
        return SmootherResult(
            means=jnp.concatenate([means, f.means[-1:]]),
            covs=jnp.concatenate([covs, f.covs[-1:]]),
        )
