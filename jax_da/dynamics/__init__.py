"""Dynamical systems: deterministic flows over one assimilation interval on a flat state.

``KuramotoSivashinsky`` and ``KolmogorovFlow`` need the optional Exponax
dependency only when a model is integrated, so importing them is always safe.
"""

from jax_da.dynamics.kolmogorov import KolmogorovFlow
from jax_da.dynamics.ks import KuramotoSivashinsky
from jax_da.dynamics.linear import LinearDynamics
from jax_da.dynamics.lorenz63 import Lorenz63
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.dynamics.lorenz96_two_scale import Lorenz96TwoScale

__all__ = ["KolmogorovFlow", "KuramotoSivashinsky", "LinearDynamics", "Lorenz63", "Lorenz96", "Lorenz96TwoScale"]
