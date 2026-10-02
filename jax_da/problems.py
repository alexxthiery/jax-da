"""Named benchmark models with settings from the literature.

Each function returns a ``StateSpaceModel``. Keyword arguments override the
defaults, so a preset is a starting point rather than a fixed configuration.
Chaotic presets place ``initial_mean`` on the attractor by spinning up from
``attractor_seed``; ``x_0`` is then ``initial_mean`` plus ``initial_std`` noise.
The same arguments always give the same model.
"""

import jax
import jax.numpy as jnp

from jax_da.dynamics.linear import LinearDynamics
from jax_da.dynamics.lorenz63 import Lorenz63
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.dynamics.lorenz96_two_scale import Lorenz96TwoScale
from jax_da.noise import Gaussian
from jax_da.observations import Selector
from jax_da.ssm import StateSpaceModel


def _attractor_point(dynamics, n_spinup: int, attractor_seed: int):
    x = dynamics.initial_condition(jax.random.PRNGKey(attractor_seed))
    return jax.jit(dynamics.spinup, static_argnums=1)(x, n_spinup)


def _chaotic(dynamics, obs_operator, obs_std, initial_std, model_error_std, n_spinup, attractor_seed):
    D = dynamics.dim
    return StateSpaceModel(
        dynamics=dynamics,
        obs_operator=obs_operator,
        obs_noise=Gaussian.isotropic(obs_operator.dim, obs_std),
        initial_mean=_attractor_point(dynamics, n_spinup, attractor_seed),
        initial_noise=Gaussian.isotropic(D, initial_std) if initial_std else None,
        model_error=Gaussian.isotropic(D, model_error_std) if model_error_std else None,
    )


def linear_gaussian(dim: int = 4, obs_every: int = 2, obs_std: float = 0.5, model_error_std: float = 0.3,
                    initial_std: float = 1.0, decay: float = 0.98, angle: float = 0.2) -> StateSpaceModel:
    """Damped rotations observed on every ``obs_every``-th component; exact answers via ``KalmanOracle``."""
    return StateSpaceModel(
        dynamics=LinearDynamics.damped_rotation(dim, decay, angle),
        obs_operator=Selector.every(dim, obs_every),
        obs_noise=Gaussian.isotropic(len(range(0, dim, obs_every)), obs_std),
        initial_mean=jnp.zeros(dim),
        initial_noise=Gaussian.isotropic(dim, initial_std),
        model_error=Gaussian.isotropic(dim, model_error_std),
    )


def lorenz63(dt: float = 0.25, obs_std: float = 2 ** 0.5, initial_std: float = 1.0, model_error_std: float = 0.0,
             n_spinup: int = 1000, attractor_seed: int = 0) -> StateSpaceModel:
    """Lorenz-63 observed in all components every 0.25 time units with ``R = 2 I`` (Sakov et al. 2012)."""
    dynamics = Lorenz63(dt=dt, substeps=max(1, round(dt / 0.01)))
    return _chaotic(dynamics, Selector(3, (0, 1, 2)), obs_std, initial_std, model_error_std, n_spinup, attractor_seed)


def lorenz96(dim: int = 40, obs_every: int = 1, dt: float = 0.05, forcing: float = 8.0, obs_std: float = 1.0,
             initial_std: float = 1.0, model_error_std: float = 0.0, n_spinup: int = 1000,
             attractor_seed: int = 0) -> StateSpaceModel:
    """Lorenz-96 with ``D = 40``, ``F = 8``, ``dt = 0.05``, every component observed with ``R = I`` (Sakov and Oke 2008).

    ``obs_every=2`` or ``4`` gives the sparse-network variants used in localization studies.
    """
    dynamics = Lorenz96(dim=dim, dt=dt, forcing=forcing)
    return _chaotic(dynamics, Selector.every(dim, obs_every), obs_std, initial_std, model_error_std,
                    n_spinup, attractor_seed)


def lorenz96_two_scale(n_slow: int = 8, n_fast: int = 32, dt: float = 0.05, forcing: float = 20.0,
                       obs_std: float = 1.0, initial_std: float = 0.0, n_spinup: int = 2000,
                       attractor_seed: int = 0) -> StateSpaceModel:
    """Two-scale Lorenz-96 truth (``K = 8, J = 32, F = 20, h = 1, c = b = 10``, Wilks 2005), slow variables observed.

    Intended as the truth of a model-error study: assimilate with
    ``Lorenz96(dim=n_slow, forcing=forcing)`` and an observation operator on
    the slow block. ``initial_std`` defaults to 0 because perturbing the fast
    variables by a common std would knock them off the attractor.
    """
    dynamics = Lorenz96TwoScale(n_slow=n_slow, n_fast=n_fast, dt=dt, forcing=forcing)
    return _chaotic(dynamics, Selector(dynamics.dim, tuple(range(n_slow))), obs_std, initial_std, 0.0,
                    n_spinup, attractor_seed)


def kuramoto_sivashinsky(num_points: int = 128, obs_every: int = 8, dt: float = 1.0, obs_std: float = 0.7,
                         initial_std: float = 0.1, model_error_std: float = 0.0, n_spinup: int = 500,
                         attractor_seed: int = 0) -> StateSpaceModel:
    """KS on ``L = 32 pi`` with 128 points, every 8th point observed every time unit (Bach et al. 2025). Needs Exponax."""
    from jax_da.dynamics.ks import KuramotoSivashinsky

    dynamics = KuramotoSivashinsky(num_points=num_points, dt=dt)
    return _chaotic(dynamics, Selector.every(num_points, obs_every), obs_std, initial_std, model_error_std,
                    n_spinup, attractor_seed)


def kolmogorov(resolution: int = 64, obs_per_side: int = 8, dt: float = 0.2, obs_std: float = 0.1,
               initial_std: float = 0.1, model_error_std: float = 0.2, n_spinup: int = 500,
               attractor_seed: int = 0) -> StateSpaceModel:
    """Kolmogorov flow at ``Re = 100`` on a 64 x 64 grid, vorticity observed on an 8 x 8 sub-grid. Needs Exponax.

    Spin-up at full resolution takes some seconds.
    """
    from jax_da.dynamics.kolmogorov import KolmogorovFlow

    dynamics = KolmogorovFlow(resolution=resolution, dt=dt)
    side = [(k * resolution) // obs_per_side for k in range(obs_per_side)]
    indices = tuple(i * resolution + j for i in side for j in side)
    return _chaotic(dynamics, Selector(dynamics.dim, indices), obs_std, initial_std, model_error_std,
                    n_spinup, attractor_seed)
