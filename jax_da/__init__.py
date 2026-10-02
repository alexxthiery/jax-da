"""jax-da: state-space models, scoring, and oracles for testing data assimilation in JAX."""

from jax_da.dynamics import LinearDynamics, Lorenz63, Lorenz96
from jax_da.geometry import Ring, Torus2D, Unstructured
from jax_da.noise import Cauchy, Gaussian, GaussianMixture, Laplace, StudentT
from jax_da.observations import Elementwise, Linear, Selector
from jax_da.ssm import StateSpaceModel, Trajectory

__version__ = "0.1.0"
__all__ = [
    "Cauchy",
    "Elementwise",
    "Gaussian",
    "GaussianMixture",
    "Laplace",
    "Linear",
    "LinearDynamics",
    "Lorenz63",
    "Lorenz96",
    "Ring",
    "Selector",
    "StateSpaceModel",
    "StudentT",
    "Torus2D",
    "Trajectory",
    "Unstructured",
]
