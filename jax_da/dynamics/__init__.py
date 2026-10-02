"""Nonlinear dynamics: maps over one assimilation interval on a flat state.

Each is a map (``in_dim == dim``, called as ``model(x)``) with a ``geometry``;
the chaotic systems add ``initial_condition`` and ``spinup``.
``KuramotoSivashinsky`` and ``KolmogorovFlow`` need the optional Exponax
dependency only when integrated, so importing them is always safe.
Linear and function-defined dynamics are ``jax_da.Linear`` and ``jax_da.Function``.
"""

from jax_da.dynamics.kolmogorov import KolmogorovFlow
from jax_da.dynamics.ks import KuramotoSivashinsky
from jax_da.dynamics.lorenz63 import Lorenz63
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.dynamics.lorenz96_two_scale import Lorenz96TwoScale
from jax_da.dynamics.tanh_squared import TanhSquared

__all__ = ["KolmogorovFlow", "KuramotoSivashinsky", "Lorenz63", "Lorenz96", "Lorenz96TwoScale", "TanhSquared"]
