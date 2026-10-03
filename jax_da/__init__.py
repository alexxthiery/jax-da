"""jax-da: state-space models, scoring, and oracles for testing data assimilation in JAX."""

from jax_da import metrics, problems
from jax_da.conditional import Additive, Lagged, Multiplicative, Poisson, Precomposed
from jax_da.dynamics import (
    KolmogorovFlow,
    KuramotoSivashinsky,
    Lorenz63,
    Lorenz96,
    Lorenz96TwoScale,
    TanhSquared,
)
from jax_da.geometry import Ring, Torus2D, Unstructured
from jax_da.laws import Cauchy, Embedded, Gaussian, GaussianMixture, History, Laplace, PointMass, StudentT
from jax_da.maps import Elementwise, Function, Linear, Selector
from jax_da.oracles import FilterResult, KalmanOracle, SmootherResult
from jax_da.ssm import StateSpaceModel, Trajectory
from jax_da.transforms import delayed

__version__ = "0.1.0"
__all__ = [
    "Additive",
    "Cauchy",
    "Elementwise",
    "Embedded",
    "FilterResult",
    "Function",
    "Gaussian",
    "GaussianMixture",
    "History",
    "KalmanOracle",
    "KolmogorovFlow",
    "KuramotoSivashinsky",
    "Lagged",
    "Laplace",
    "Linear",
    "Lorenz63",
    "Lorenz96",
    "Lorenz96TwoScale",
    "Multiplicative",
    "PointMass",
    "Poisson",
    "Precomposed",
    "Ring",
    "Selector",
    "SmootherResult",
    "StateSpaceModel",
    "StudentT",
    "TanhSquared",
    "Torus2D",
    "Trajectory",
    "Unstructured",
    "delayed",
    "metrics",
    "problems",
]
