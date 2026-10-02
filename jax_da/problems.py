"""Named benchmark models with settings from the literature.

Each function returns a ``StateSpaceModel``. Keyword arguments override the
defaults, so a preset is a starting point rather than a fixed configuration.
Chaotic presets place ``initial_mean`` on the attractor by spinning up from
``attractor_seed``; ``x_0`` is then ``initial_mean`` plus ``initial_std`` noise.
The same arguments always give the same model.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jax_da.dynamics.kolmogorov import KolmogorovFlow
from jax_da.dynamics.ks import KuramotoSivashinsky
from jax_da.dynamics.linear import LinearDynamics
from jax_da.dynamics.lorenz63 import Lorenz63
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.dynamics.lorenz96_two_scale import Lorenz96TwoScale
from jax_da.geometry import Ring, Torus2D
from jax_da.noise import Gaussian
from jax_da.observations import Linear, Selector
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


def _random_covariance(rng: np.random.Generator, dim: int, std: float) -> np.ndarray:
    """Correlated SPD matrix ``G G^T / dim + I`` rescaled so its mean variance is ``std**2``.

    The identity term keeps the condition number moderate (eigenvalues of the
    unscaled matrix lie in ``[1, (1 + 1)^2 + 1]`` asymptotically).
    """
    g = rng.standard_normal((dim, dim))
    cov = g @ g.T / dim + np.eye(dim)
    return cov * (std ** 2 / np.mean(np.diag(cov)))


def linear_gaussian_full(state_dim: int = 6, obs_dim: int = 3, spectral_radius: float = 0.95,
                         model_error_std: float = 0.3, obs_std: float = 0.5, initial_std: float = 1.0,
                         seed: int = 0) -> StateSpaceModel:
    """General linear-Gaussian model with dense ``A``, dense ``H``, and full ``Q``, ``R``, ``P0``.

        x_0 ~ N(0, P0),  x_t = A x_{t-1} + eta_t,  eta_t ~ N(0, Q),  y_t = H x_t + eps_t,  eps_t ~ N(0, R)

    ``A`` is a Gaussian matrix rescaled to ``spectral_radius``; ``H`` is Gaussian
    with entries of variance ``1 / state_dim``; ``Q``, ``R``, ``P0`` are correlated
    SPD matrices with mean variance ``model_error_std**2``, ``obs_std**2``, and
    ``initial_std**2``. Matrices come from ``numpy.random.default_rng(seed)``, so
    the model is identical across JAX versions and devices. Exact answers via
    ``KalmanOracle``.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((state_dim, state_dim))
    a *= spectral_radius / np.max(np.abs(np.linalg.eigvals(a)))
    h = rng.standard_normal((obs_dim, state_dim)) / np.sqrt(state_dim)
    q = _random_covariance(rng, state_dim, model_error_std)
    r = _random_covariance(rng, obs_dim, obs_std)
    p0 = _random_covariance(rng, state_dim, initial_std)
    return StateSpaceModel(
        dynamics=LinearDynamics(jnp.asarray(a)),
        obs_operator=Linear(jnp.asarray(h)),
        obs_noise=Gaussian.full(jnp.asarray(r)),
        initial_mean=jnp.zeros(state_dim),
        initial_noise=Gaussian.full(jnp.asarray(p0)),
        model_error=Gaussian.full(jnp.asarray(q)),
    )


def _fourier_operator(multiplier: np.ndarray) -> np.ndarray:
    """Dense real matrix of the map ``x -> ifftn(multiplier * fftn(x))`` on a periodic grid.

    ``multiplier`` has the grid shape and must be Hermitian, ``m(-k) = conj(m(k))``,
    so the map is real. Column ``j`` is the image of the ``j``-th row-major basis field.
    """
    shape = multiplier.shape
    dim = int(np.prod(shape))
    axes = tuple(range(1, len(shape) + 1))
    basis = np.eye(dim).reshape((dim,) + shape)
    images = np.fft.ifftn(multiplier * np.fft.fftn(basis, axes=axes), axes=axes)
    if np.abs(images.imag).max() > 1e-10 * np.abs(images.real).max():
        raise ValueError("Fourier multiplier is not Hermitian; the operator would be complex")
    return images.real.reshape(dim, dim).T


def advection_diffusion(grid_shape: tuple[int, ...] = (256,), dt: float = 1.0, velocity=1.0,
                        diffusivity: float = 0.1, damping: float = 0.02, noise_length: float = 10.0,
                        model_error_std: float = 0.2, obs_every: int = 8, obs_std: float = 0.5,
                        nugget: float = 0.01) -> StateSpaceModel:
    """Stochastic advection-diffusion on a periodic ring or torus, in statistical equilibrium. Exact via ``KalmanOracle``.

        du/dt + c . grad u = kappa Laplacian(u) - lambda u + noise

    over one interval ``dt`` is applied exactly in Fourier space: mode ``k`` is
    multiplied by ``exp(-(kappa |k|^2 + lambda) dt) exp(-i c . k dt)``, so there is
    no discretization error and fractional-cell shifts are exact. Lengths are in
    grid cells. The model error is Gaussian, translation invariant, with a
    squared-exponential correlation of length ``noise_length`` plus a white
    ``nugget`` fraction that keeps ``Q`` well conditioned, and marginal std
    ``model_error_std``. ``x_0`` is drawn from the stationary law ``N(0, Sigma)``,
    ``Sigma = A Sigma A^T + Q``, whose spectrum is ``q_k / (1 - |a_k|^2)``; this
    needs ``damping > 0``. Every ``obs_every``-th grid point along each axis is
    observed with noise std ``obs_std``.

    Args:
        grid_shape: ``(n,)`` for a ring or ``(height, width)`` for a torus.
        velocity: Advection speed in cells per time unit, scalar or one per axis.

    Returns:
        A linear-Gaussian ``StateSpaceModel`` with ``Ring`` or ``Torus2D`` geometry.
    """
    shape = tuple(int(n) for n in grid_shape)
    if len(shape) not in (1, 2):
        raise ValueError(f"grid_shape must be (n,) or (height, width), got {grid_shape}")
    if damping <= 0:
        raise ValueError(f"damping must be positive for a stationary climatology, got {damping}")
    velocity = np.broadcast_to(np.asarray(velocity, dtype=float), (len(shape),))
    k = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n) for n in shape], indexing="ij")
    k_squared = sum(ki ** 2 for ki in k)
    # Translation by c dt along each axis. On the grid the Nyquist mode cos(pi i) has no sine
    # partner, so translation acts on it as the real factor cos(pi c dt): exact for whole-cell shifts.
    translation = np.ones(shape, dtype=complex)
    for c, ki in zip(velocity, k):
        nyquist = np.isclose(np.abs(ki), np.pi)
        translation *= np.where(nyquist, np.cos(c * ki * dt), np.exp(-1j * c * ki * dt))
    a = np.exp(-(diffusivity * k_squared + damping) * dt) * translation
    q = np.exp(-0.5 * k_squared * noise_length ** 2)
    q = model_error_std ** 2 * ((1 - nugget) * q / q.mean() + nugget)  # spectrum mean = marginal variance
    sigma = q / (1 - np.abs(a) ** 2)

    dim = int(np.prod(shape))
    layout = Ring(shape[0]) if len(shape) == 1 else Torus2D(*shape)
    observed = np.zeros(shape, dtype=bool)
    observed[tuple(slice(0, None, obs_every) for _ in shape)] = True
    indices = tuple(int(i) for i in np.flatnonzero(observed))  # row-major flat indices
    return StateSpaceModel(
        dynamics=LinearDynamics(jnp.asarray(_fourier_operator(a)), layout=layout),
        obs_operator=Selector(dim, indices),
        obs_noise=Gaussian.isotropic(len(indices), obs_std),
        initial_mean=jnp.zeros(dim),
        initial_noise=Gaussian.full(jnp.asarray(_fourier_operator(sigma))),
        model_error=Gaussian.full(jnp.asarray(_fourier_operator(q))),
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
    dynamics = KuramotoSivashinsky(num_points=num_points, dt=dt)
    return _chaotic(dynamics, Selector.every(num_points, obs_every), obs_std, initial_std, model_error_std,
                    n_spinup, attractor_seed)


def kolmogorov(resolution: int = 64, obs_per_side: int = 8, dt: float = 0.2, obs_std: float = 0.1,
               initial_std: float = 0.1, model_error_std: float = 0.2, n_spinup: int = 500,
               attractor_seed: int = 0) -> StateSpaceModel:
    """Kolmogorov flow at ``Re = 100`` on a 64 x 64 grid, vorticity observed on an 8 x 8 sub-grid. Needs Exponax.

    Spin-up at full resolution takes some seconds.
    """
    dynamics = KolmogorovFlow(resolution=resolution, dt=dt)
    side = [(k * resolution) // obs_per_side for k in range(obs_per_side)]
    indices = tuple(i * resolution + j for i in side for j in side)
    return _chaotic(dynamics, Selector(dynamics.dim, indices), obs_std, initial_std, model_error_std,
                    n_spinup, attractor_seed)
