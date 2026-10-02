"""jax-da: state-space models, scoring, and oracles for testing data assimilation in JAX."""

from jax_da import metrics, problems
from jax_da.conditional import Additive, Multiplicative, Poisson
from jax_da.dynamics import (
    KolmogorovFlow,
    KuramotoSivashinsky,
    Lorenz63,
    Lorenz96,
    Lorenz96TwoScale,
    TanhSquared,
)
from jax_da.geometry import Ring, Torus2D, Unstructured
from jax_da.laws import Cauchy, Gaussian, GaussianMixture, Laplace, PointMass, StudentT
from jax_da.maps import Elementwise, Function, Linear, Selector
from jax_da.oracles import FilterResult, KalmanOracle, SmootherResult
from jax_da.ssm import StateSpaceModel, Trajectory

__version__ = "0.1.0"
__all__ = [
    "Additive",
    "Cauchy",
    "Elementwise",
    "FilterResult",
    "Function",
    "Gaussian",
    "GaussianMixture",
    "KalmanOracle",
    "KolmogorovFlow",
    "KuramotoSivashinsky",
    "Laplace",
    "Linear",
    "Lorenz63",
    "Lorenz96",
    "Lorenz96TwoScale",
    "Multiplicative",
    "PointMass",
    "Poisson",
    "Ring",
    "Selector",
    "SmootherResult",
    "StateSpaceModel",
    "StudentT",
    "TanhSquared",
    "Torus2D",
    "Trajectory",
    "Unstructured",
    "metrics",
    "problems",
]
