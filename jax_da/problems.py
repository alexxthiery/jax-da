"""Named benchmark models with settings from the literature.

Each function returns a ``StateSpaceModel``. Keyword arguments override the
defaults, so a preset is a starting point rather than a fixed configuration.
Chaotic presets spin up from ``attractor_seed`` to a point on the attractor;
``x_0`` is that point plus Gaussian noise of std ``initial_std`` (a ``PointMass``
when ``initial_std`` is 0).
The same arguments always give the same model.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np

from jax_da.conditional import Additive, Multiplicative, Poisson
from jax_da.dynamics.kolmogorov import KolmogorovFlow
from jax_da.dynamics.ks import KuramotoSivashinsky
from jax_da.dynamics.lorenz63 import Lorenz63
from jax_da.dynamics.lorenz96 import Lorenz96
from jax_da.dynamics.lorenz96_two_scale import Lorenz96TwoScale
from jax_da.dynamics.tanh_squared import TanhSquared
from jax_da.geometry import Ring, Torus2D
from jax_da.laws import Embedded, Gaussian, PointMass
from jax_da.maps import Linear, Selector
from jax_da.ssm import StateSpaceModel


def _attractor_point(dynamics, n_spinup: int, attractor_seed: int):
    x = dynamics.initial_condition(jax.random.PRNGKey(attractor_seed))
    return jax.jit(dynamics.spinup, static_argnums=1)(x, n_spinup)


def _chaotic(dynamics, obs_operator, obs_std, initial_std, model_error_std, n_spinup, attractor_seed):
    D = dynamics.dim
    x_star = _attractor_point(dynamics, n_spinup, attractor_seed)
    return StateSpaceModel(
        initial=Gaussian.isotropic(D, initial_std, loc=x_star) if initial_std else PointMass(x_star),
        transition=Additive(dynamics, Gaussian.isotropic(D, model_error_std) if model_error_std else None),
        observation=Additive(obs_operator, Gaussian.isotropic(obs_operator.dim, obs_std)),
        geometry=dynamics.geometry,
    )


def linear_gaussian(dim: int = 4, obs_every: int = 2, obs_std: float = 0.5, model_error_std: float = 0.3,
                    initial_std: float = 1.0, decay: float = 0.98, angle: float = 0.2) -> StateSpaceModel:
    """Damped rotations observed on every ``obs_every``-th component; exact answers via ``KalmanOracle``."""
    observe = Selector.every(dim, obs_every)
    return StateSpaceModel(
        initial=Gaussian.isotropic(dim, initial_std),
        transition=Additive(Linear.damped_rotation(dim, decay, angle), Gaussian.isotropic(dim, model_error_std)),
        observation=Additive(observe, Gaussian.isotropic(observe.dim, obs_std)),
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
    if state_dim < 1 or obs_dim < 1:
        raise ValueError(f"state_dim and obs_dim must be positive, got {state_dim} and {obs_dim}")
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((state_dim, state_dim))
    a *= spectral_radius / np.max(np.abs(np.linalg.eigvals(a)))
    h = rng.standard_normal((obs_dim, state_dim)) / np.sqrt(state_dim)
    q = _random_covariance(rng, state_dim, model_error_std)
    r = _random_covariance(rng, obs_dim, obs_std)
    p0 = _random_covariance(rng, state_dim, initial_std)
    return StateSpaceModel(
        initial=Gaussian.full(jnp.asarray(p0)),
        transition=Additive(Linear(jnp.asarray(a)), Gaussian.full(jnp.asarray(q))),
        observation=Additive(Linear(jnp.asarray(h)), Gaussian.full(jnp.asarray(r))),
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


def _stationary_covariance(A: np.ndarray, Q: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    """``sum_k A^k Q A^kT`` by doubling: ``S <- S + M S M^T``, ``M <- M M``; needs a contractive ``A``."""
    S, M = Q.copy(), A.copy()
    for _ in range(64):
        increment = M @ S @ M.T
        S = S + increment
        M = M @ M
        if np.abs(increment).max() <= tol * np.abs(S).max():
            return 0.5 * (S + S.T)
    raise ValueError("stationary covariance did not converge; the dynamics must be contractive (damping > 0)")


def _low_rank_gaussian(cov: np.ndarray, rtol: float = 1e-12) -> Embedded:
    """Zero-mean Gaussian with covariance ``cov`` (PSD, possibly singular) as an ``Embedded`` law.

    Eigenvalues below ``rtol`` times the largest are dropped, a relative change of at
    most ``rtol`` in the covariance.
    """
    values, vectors = np.linalg.eigh(cov)
    keep = values > rtol * values.max()
    factor = vectors[:, keep] * np.sqrt(values[keep])
    return Embedded(Gaussian.isotropic(int(keep.sum()), 1.0), jnp.asarray(factor))


def advection_diffusion(grid_shape: tuple[int, ...] = (256,), dt: float = 1.0, velocity=1.0,
                        diffusivity: float = 0.1, damping: float = 0.02, noise_length: float = 10.0,
                        model_error_std: float = 0.2, obs_every: int = 8, obs_std: float = 0.5,
                        nugget: float = 0.01, forcing_mask=None) -> StateSpaceModel:
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

    With ``forcing_mask`` (a boolean array of shape ``grid_shape``), the model error
    is the same correlated noise restricted to the masked cells and zero elsewhere (an
    ``Embedded`` law), so noise injected upstream reaches downstream sensors only after
    the advection travel time. The stationary covariance is then computed by summing
    ``A^k Q A^kT`` (doubling), since it is no longer translation invariant.

    Args:
        grid_shape: ``(n,)`` for a ring or ``(height, width)`` for a torus.
        velocity: Advection speed in cells per time unit, scalar or one per axis.
        forcing_mask: Cells that receive model error; all cells if None.

    Returns:
        A linear-Gaussian ``StateSpaceModel`` with ``Ring`` or ``Torus2D`` geometry.
    """
    shape = tuple(int(n) for n in grid_shape)
    if len(shape) not in (1, 2):
        raise ValueError(f"grid_shape must be (n,) or (height, width), got {grid_shape}")
    if damping <= 0:
        raise ValueError(f"damping must be positive for a stationary climatology, got {damping}")
    if obs_every < 1:
        raise ValueError(f"obs_every must be a positive integer, got {obs_every}")
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
    A, Q = _fourier_operator(a), _fourier_operator(q)
    noise, initial = Gaussian.full(jnp.asarray(Q)), Gaussian.full(jnp.asarray(_fourier_operator(sigma)))
    if forcing_mask is not None:
        mask = np.asarray(forcing_mask)
        if mask.shape != shape or mask.dtype != bool or not mask.any():
            raise ValueError(f"forcing_mask must be a boolean array of shape {shape} with at least one True cell")
        forced = np.flatnonzero(mask)  # row-major flat indices
        embed, Q_forced = np.eye(dim)[:, forced], Q[np.ix_(forced, forced)]
        noise = Embedded(Gaussian.full(jnp.asarray(Q_forced)), jnp.asarray(embed))
        # With diffusion, cells far downstream of the forcing receive only heavily damped
        # high-wavenumber content, so the stationary covariance is singular to machine
        # precision. Keep it exactly as a low-rank law instead of regularizing it.
        initial = _low_rank_gaussian(_stationary_covariance(A, embed @ Q_forced @ embed.T))
    layout = Ring(shape[0]) if len(shape) == 1 else Torus2D(*shape)
    observed = np.zeros(shape, dtype=bool)
    observed[tuple(slice(0, None, obs_every) for _ in shape)] = True
    indices = tuple(int(i) for i in np.flatnonzero(observed))  # row-major flat indices
    return StateSpaceModel(
        initial=initial,
        transition=Additive(Linear(jnp.asarray(A)), noise),
        observation=Additive(Selector(dim, indices), Gaussian.isotropic(len(indices), obs_std)),
        geometry=layout,
    )


def integrated_random_walk(order: int = 2, dt: float = 1.0, noise_std: float = 1.0, obs_std: float = 1.0,
                           initial_std: float = 1.0, discretization: str = "exact") -> StateSpaceModel:
    """A Brownian motion integrated ``order - 1`` times, observed in its lowest coordinate.

    The state is ``(position, velocity, ...)`` with ``order`` components; white noise of
    intensity ``noise_std`` drives the highest derivative and only the position is
    observed. ``order=2`` is the constant-velocity tracking model, ``order=3`` constant
    acceleration. Fresh noise reaches the observation only through ``order - 1``
    integrations, so a particle filter must select on noise injected several steps ago.

    ``discretization="exact"`` integrates the linear SDE over ``dt``: ``A = expm(N dt)``
    and ``Q_ij = noise_std^2 dt^(2n-1-i-j) / ((n-1-i)! (n-1-j)! (2n-1-i-j))``, full rank
    but with ``H Q H^T = O(dt^(2n-1))``. ``discretization="euler"`` uses ``A = I + N dt``
    and noise ``noise_std * sqrt(dt)`` on the highest derivative only (an ``Embedded``
    law), so ``H A^k G = 0`` exactly for ``k < order - 1``. Both are exact for
    ``KalmanOracle``.
    """
    if not isinstance(order, int) or order < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")
    if not dt > 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if discretization not in ("exact", "euler"):
        raise ValueError(f"discretization must be 'exact' or 'euler', got {discretization!r}")
    n = order
    if discretization == "exact":
        A = np.array([[dt ** (j - i) / math.factorial(j - i) if j >= i else 0.0 for j in range(n)] for i in range(n)])
        Q = np.array([[noise_std ** 2 * dt ** (2 * n - 1 - i - j)
                       / (math.factorial(n - 1 - i) * math.factorial(n - 1 - j) * (2 * n - 1 - i - j))
                       for j in range(n)] for i in range(n)])
        noise = Gaussian.full(jnp.asarray(Q))
    else:
        A = np.eye(n) + dt * np.diag(np.ones(n - 1), 1)
        noise = Embedded(Gaussian.isotropic(1, noise_std * math.sqrt(dt)), jnp.asarray(np.eye(n)[:, -1:]))
    return StateSpaceModel(
        initial=Gaussian.isotropic(n, initial_std),
        transition=Additive(Linear(jnp.asarray(A)), noise),
        observation=Additive(Selector(n, (0,)), Gaussian.isotropic(1, obs_std)),
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


def stochastic_volatility(dim: int = 1, phi: float = 0.98, sigma: float = 0.15, beta: float = 0.8,
                          mu: float = 0.0) -> StateSpaceModel:
    """Stochastic volatility: latent log-variance ``x`` drives the scale of the observed returns.

        x_t = mu + phi (x_{t-1} - mu) + sigma eta_t,   y_t = beta exp(x_t / 2) eps_t,   eta, eps ~ N(0, I)

    independently per component, with ``x_0`` from the stationary law
    ``N(mu, sigma^2 / (1 - phi^2))``. The likelihood is far from Gaussian in
    ``x``: a classic particle-filter benchmark on which Kalman-type methods are
    not consistent (Kim, Shephard, and Chib 1998; multivariate diagonal form as
    in Chib, Omori, and Asai 2009 and the SIXO benchmark). Defaults follow the
    flowsmc example.
    """
    if not abs(phi) < 1:
        raise ValueError(f"|phi| must be < 1 for a stationary initial law, got {phi}")
    if not (sigma > 0 and beta > 0):
        raise ValueError(f"sigma and beta must be positive, got sigma={sigma}, beta={beta}")
    eye = jnp.eye(dim)
    return StateSpaceModel(
        initial=Gaussian.isotropic(dim, sigma / np.sqrt(1 - phi ** 2), loc=mu),
        transition=Additive(Linear(phi * eye, offset=(1 - phi) * mu), Gaussian.isotropic(dim, sigma)),
        observation=Multiplicative(Linear(0.5 * eye, offset=np.log(beta)), Gaussian.isotropic(dim, 1.0)),
    )


def nonlinear_poisson(state_dim: int = 4, obs_dim: int = 32, rho: float = 0.55, alpha: float = 0.55,
                      q: float = 0.45, seed: int = 0) -> StateSpaceModel:
    """Nonlinear Gaussian dynamics observed through Poisson counts (flowsmc benchmark).

        x_0 ~ N(0, I),   x_t = rho x_{t-1} + alpha (tanh(B x_{t-1})^2 - 1/4) + q eta_t,
        y_{t,i} ~ Poisson(exp(C_i x_t + b_i))

    ``B`` is a Gaussian matrix scaled to spectral norm at most 1.2 (``[[1.5]]``
    when ``state_dim == 1``), ``C`` is Gaussian with entry variance
    ``1 / state_dim`` (an even ramp in ``[-1.2, 1.2]`` when ``state_dim == 1``),
    and ``b_i = -0.2 + 0.1 cos(i)``, as in flowsmc's "quick" configuration.
    Matrices come from ``numpy.random.default_rng(seed)``. ``rho``, ``alpha``,
    and ``q`` are pytree children, so they can be learned by gradient.
    """
    rng = np.random.default_rng(seed)
    if state_dim == 1:
        B = np.array([[1.5]])
        C = (1.2 * np.linspace(-1.0, 1.0, obs_dim))[:, None]
    else:
        raw = rng.standard_normal((state_dim, state_dim))
        B = 1.2 * raw / max(np.linalg.norm(raw, 2), 1.0)
        C = rng.standard_normal((obs_dim, state_dim)) / np.sqrt(state_dim)
    b = -0.2 + 0.1 * np.cos(np.arange(obs_dim))
    return StateSpaceModel(
        initial=Gaussian.isotropic(state_dim, 1.0),
        transition=Additive(TanhSquared(jnp.asarray(B), rho, alpha), Gaussian.isotropic(state_dim, q)),
        observation=Poisson(Linear(jnp.asarray(C), offset=jnp.asarray(b))),
    )
