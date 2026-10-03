"""Noise on a subspace (Embedded), observation of a function of the state (Precomposed),
stacked histories (Lagged, History), and the delayed-observation transform."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_da as jd


def brute_force_filter(A, b, Q, H, c, R, m0, P0, y):
    """Filtering means and covariances by conditioning the joint Gaussian of (x_1..x_T, y_1..y_T).

    Built from the raw model constants, independently of KalmanOracle; Q and P0 may be singular.
    """
    T, D, P = y.shape[0], A.shape[0], H.shape[0]
    means, m = [], m0
    for _ in range(T):
        m = A @ m + b
        means.append(m)
    mx = np.concatenate(means)
    powers = [np.linalg.matrix_power(A, k) for k in range(T + 1)]
    Cx = np.zeros((T * D, T * D))
    for t in range(1, T + 1):
        for s in range(1, T + 1):
            block = powers[t] @ P0 @ powers[s].T
            for k in range(1, min(t, s) + 1):
                block = block + powers[t - k] @ Q @ powers[s - k].T
            Cx[(t - 1) * D:t * D, (s - 1) * D:s * D] = block
    Hb = np.kron(np.eye(T), H)
    my, Cxy, Cy = Hb @ mx + np.tile(c, T), Cx @ Hb.T, Hb @ Cx @ Hb.T + np.kron(np.eye(T), R)
    out_m, out_C = [], []
    for t in range(T):
        rows, obs = np.arange(t * D, (t + 1) * D), np.arange((t + 1) * P)
        gain = Cxy[np.ix_(rows, obs)] @ np.linalg.inv(Cy[np.ix_(obs, obs)])
        out_m.append(mx[rows] + gain @ (y.ravel()[obs] - my[obs]))
        out_C.append(Cx[np.ix_(rows, rows)] - gain @ Cxy[np.ix_(rows, obs)].T)
    return np.array(out_m), np.array(out_C)


G = jnp.array([[1.0, 0.0], [0.5, 1.0], [0.0, -2.0]])  # rank 2 in R^3


def test_embedded_law_moments_and_support():
    inner = jd.Gaussian.diagonal(jnp.array([0.7, 0.3]), loc=jnp.array([1.0, -1.0]))
    law = jd.Embedded(inner, G)
    assert law.dim == 3
    np.testing.assert_allclose(law.loc, G @ jnp.array([1.0, -1.0]), rtol=1e-12)
    np.testing.assert_allclose(law.cov(), G @ inner.cov() @ G.T, rtol=1e-12)
    draws = np.asarray(law.sample(jax.random.PRNGKey(0), (200_000,)))
    # Five standard errors of the sample mean, sqrt(max variance / n).
    np.testing.assert_allclose(draws.mean(0), law.loc, atol=5 * float(np.sqrt(np.max(np.diag(law.cov())) / 200_000)))
    np.testing.assert_allclose(np.cov(draws.T), law.cov(), rtol=0.02, atol=0.003)
    # Every draw lies in the range of G: the residual of its projection is zero.
    projector = np.asarray(G) @ np.linalg.pinv(np.asarray(G))
    np.testing.assert_allclose(draws[:100] @ projector.T, draws[:100], atol=1e-12)


def test_oracle_is_exact_with_singular_model_error():
    # Euler constant velocity: noise enters velocity only, position is observed.
    dt, sigma = 0.5, 0.8
    A = np.array([[1.0, dt], [0.0, 1.0]])
    e = np.array([[0.0], [1.0]])
    ssm = jd.StateSpaceModel(
        initial=jd.Gaussian.full(jnp.array([[1.0, 0.2], [0.2, 0.5]]), loc=jnp.array([0.3, -0.1])),
        transition=jd.Additive(jd.Linear(jnp.asarray(A)), jd.Embedded(jd.Gaussian.isotropic(1, sigma * np.sqrt(dt)), jnp.asarray(e))),
        observation=jd.Additive(jd.Selector(2, (0,)), jd.Gaussian.isotropic(1, 0.4)),
    )
    y = np.asarray(ssm.simulate(jax.random.PRNGKey(1), 6).observations)
    f = jd.KalmanOracle.from_ssm(ssm).filter(jnp.asarray(y))
    m, C = brute_force_filter(A, np.zeros(2), sigma ** 2 * dt * e @ e.T, np.array([[1.0, 0.0]]), np.zeros(1),
                              np.array([[0.16]]), np.array([0.3, -0.1]), np.array([[1.0, 0.2], [0.2, 0.5]]), y)
    np.testing.assert_allclose(f.means, m, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(f.covs, C, rtol=1e-9, atol=1e-10)


H2 = jnp.array([[1.0, -0.5, 0.0], [0.2, 0.0, 1.0]])


def test_precomposed_equals_the_composed_linear_model():
    select = jd.Selector(5, (4, 1, 2))
    noise = jd.Gaussian.diagonal(jnp.array([0.3, 0.6]))
    pre = jd.Precomposed(jd.Additive(jd.Linear(H2, offset=jnp.array([0.1, -0.2])), noise), select)
    flat = jd.Additive(jd.Linear(H2 @ select.matrix, offset=jnp.array([0.1, -0.2])), noise)
    x = jax.random.normal(jax.random.PRNGKey(2), (4, 5))
    y = jax.random.normal(jax.random.PRNGKey(3), (4, 2))
    assert (pre.in_dim, pre.dim) == (5, 2)
    np.testing.assert_allclose(pre.log_prob(y, x), flat.log_prob(y, x), rtol=1e-12)
    np.testing.assert_allclose(pre.mean(x), flat.mean(x), rtol=1e-12)
    np.testing.assert_allclose(pre.sample(jax.random.PRNGKey(4), x), flat.sample(jax.random.PRNGKey(4), x), rtol=1e-12)


def test_oracle_composes_a_precomposed_observation_with_its_map_offset():
    # y = H2 (M x + o) + c + noise: the oracle must see the affine map H2 M x + (H2 o + c).
    base = jd.problems.linear_gaussian_full(state_dim=3, obs_dim=2, seed=4)
    M, o, c = np.array([[0.5, 1.0, 0.0], [0.0, 1.0, -1.0], [1.0, 0.0, 2.0]]), np.array([1.0, -2.0, 0.5]), np.array([0.3, -0.4])
    noise = jd.Gaussian.diagonal(jnp.array([0.3, 0.6]))
    observation = jd.Precomposed(jd.Additive(jd.Linear(H2, offset=jnp.asarray(c)), noise),
                                 jd.Linear(jnp.asarray(M), offset=jnp.asarray(o)))
    ssm = jd.StateSpaceModel(initial=base.initial, transition=base.transition, observation=observation)
    y = np.asarray(ssm.simulate(jax.random.PRNGKey(12), 6).observations)
    f = jd.KalmanOracle.from_ssm(ssm).filter(jnp.asarray(y))
    A, Q = np.asarray(base.transition.map.matrix), np.asarray(base.transition.noise.cov())
    m, C = brute_force_filter(A, np.zeros(3), Q, np.asarray(H2) @ M, np.asarray(H2) @ o + c, np.asarray(noise.cov()),
                              np.asarray(base.initial.loc), np.asarray(base.initial.cov()), y)
    np.testing.assert_allclose(f.means, m, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(f.covs, C, rtol=1e-9, atol=1e-10)


def base_linear():
    return jd.problems.linear_gaussian_full(state_dim=3, obs_dim=2, seed=4)


def test_lagged_shifts_history_and_draws_the_newest_block_from_the_base():
    base = base_linear().transition
    lagged = jd.Lagged(base, 2)
    assert (lagged.in_dim, lagged.dim) == (9, 9)
    u = jax.random.normal(jax.random.PRNGKey(5), (4, 9))
    out = lagged.sample(jax.random.PRNGKey(6), u)
    np.testing.assert_array_equal(out[:, 3:], u[:, :6])
    np.testing.assert_allclose(out[:, :3], base.sample(jax.random.PRNGKey(6), u[:, :3]), rtol=1e-12)
    np.testing.assert_allclose(lagged.mean(u)[:, :3], base.mean(u[:, :3]), rtol=1e-12)
    np.testing.assert_allclose(lagged.log_prob(out, u), base.log_prob(out[:, :3], u[:, :3]), rtol=1e-12)
    inconsistent = out.at[:, 5].add(1.0)
    assert bool(jnp.all(jnp.isneginf(lagged.log_prob(inconsistent, u))))


def test_history_is_a_trajectory_and_its_density_is_the_chain_density():
    deterministic = jd.Additive(jd.Lorenz96(dim=4))
    x0 = jnp.array([8.0, 7.5, 8.5, 9.0])
    hist = jd.History(jd.PointMass(x0), deterministic, 2)
    u = hist.sample(jax.random.PRNGKey(7))
    f = jd.Lorenz96(dim=4)
    np.testing.assert_allclose(u, jnp.concatenate([f(f(x0)), f(x0), x0]), rtol=1e-12)  # newest block first
    base = base_linear()
    hist = jd.History(base.initial, base.transition, 2)
    v = jax.random.normal(jax.random.PRNGKey(8), (3, 9))
    x2, x1, x0 = v[:, :3], v[:, 3:6], v[:, 6:]
    expected = (base.initial.log_prob(x0) + base.transition.log_prob(x1, x0) + base.transition.log_prob(x2, x1))
    np.testing.assert_allclose(hist.log_prob(v), expected, rtol=1e-12)
    assert hist.sample(jax.random.PRNGKey(9), (5,)).shape == (5, 9)


@pytest.mark.parametrize("lags", [1, 3])
def test_delayed_filter_on_the_oldest_block_equals_the_base_filter(lags):
    # y_t of the delayed model observes the oldest block, which is the base state x_t:
    # p(oldest block | y_1..y_t) in the delayed model must equal the base filter p(x_t | y_1..y_t).
    base = base_linear()
    delayed = jd.delayed(base, lags)
    y = base.simulate(jax.random.PRNGKey(10), 8).observations
    f_base = jd.KalmanOracle.from_ssm(base).filter(y)
    f_delayed = jd.KalmanOracle.from_ssm(delayed).filter(y)
    D = base.state_dim
    old = slice(lags * D, (lags + 1) * D)
    np.testing.assert_allclose(f_delayed.means[:, old], f_base.means, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(f_delayed.covs[:, old, old], f_base.covs, rtol=1e-8, atol=1e-10)
    assert float(f_delayed.log_evidence) == pytest.approx(float(f_base.log_evidence), rel=1e-10)


def test_delayed_simulation_observes_the_oldest_block_and_shifts():
    base = jd.problems.lorenz96(dim=8, obs_every=2, n_spinup=50, model_error_std=0.1, obs_std=1e-6)
    delayed = jd.delayed(base, 3)
    traj = delayed.simulate(jax.random.PRNGKey(11), 20)
    assert delayed.state_dim == 32 and delayed.obs_dim == 4
    np.testing.assert_array_equal(traj.states[1:, 8:], traj.states[:-1, :24])
    np.testing.assert_allclose(traj.observations, traj.states[:, 24::2], atol=1e-5)


@pytest.mark.parametrize("make", [lambda: jd.problems.stochastic_volatility(dim=2), lambda: jd.problems.nonlinear_poisson()])
def test_delayed_works_for_non_additive_observations(make):
    base = make()
    delayed = jd.delayed(base, 2)
    traj = delayed.simulate(jax.random.PRNGKey(12), 10)
    D = base.state_dim
    assert traj.observations.shape == (10, base.obs_dim)
    np.testing.assert_allclose(delayed.log_likelihood(traj.observations, traj.states),
                               base.log_likelihood(traj.observations, traj.states[:, 2 * D:]), rtol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_integrated_random_walk_exact_discretization_matches_van_loan(order):
    from scipy.linalg import expm

    dt, sigma = 0.7, 1.3
    ssm = jd.problems.integrated_random_walk(order=order, dt=dt, noise_std=sigma, discretization="exact")
    N = np.diag(np.ones(order - 1), 1)
    g = np.zeros((order, 1)); g[-1] = sigma
    # Van Loan: expm([[-N, g g^T], [0, N^T]] dt) gives A = expm(N dt) and Q = A @ (top-right block).
    block = expm(np.block([[-N, g @ g.T], [np.zeros((order, order)), N.T]]) * dt)
    A = block[order:, order:].T
    np.testing.assert_allclose(ssm.transition.map.matrix, A, rtol=1e-12)
    np.testing.assert_allclose(ssm.transition.noise.cov(), A @ block[:order, order:], rtol=1e-10, atol=1e-14)
    assert ssm.observation.map.indices == (0,)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_integrated_random_walk_euler_noise_reaches_position_only_after_order_minus_one_steps(order):
    ssm = jd.problems.integrated_random_walk(order=order, dt=0.5, noise_std=2.0, discretization="euler")
    A = np.asarray(ssm.transition.map.matrix)
    np.testing.assert_allclose(A, np.eye(order) + 0.5 * np.diag(np.ones(order - 1), 1))
    G = np.asarray(ssm.transition.noise.matrix)
    H = np.eye(order)[:1]
    effects = [float(np.abs(H @ np.linalg.matrix_power(A, k) @ G).max()) for k in range(order)]
    assert all(e == 0.0 for e in effects[:-1]) and effects[-1] > 0  # relative degree order - 1
    np.testing.assert_allclose(ssm.transition.noise.cov(), 4.0 * 0.5 * np.diag(np.eye(order)[-1]), rtol=1e-12)


@pytest.mark.pde
def test_partial_noise_page_runs_and_ks_forcing_stays_in_low_modes():
    """Executes every code block of docs/partial_noise.md (one namespace), then checks the KS recipe."""
    import re
    from pathlib import Path

    pytest.importorskip("exponax")
    text = (Path(__file__).resolve().parents[1] / "docs" / "partial_noise.md").read_text()
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", text, flags=re.S):
        exec(compile(block, "docs/partial_noise.md", "exec"), namespace)
    draws = namespace["forcing"].sample(jax.random.PRNGKey(4), (20,))
    spectrum = np.abs(np.fft.rfft(np.asarray(draws), axis=-1))
    assert spectrum[:, 4:].max() < 1e-10 * spectrum.max()  # nothing above wavenumber 3
    assert bool(jnp.all(jnp.isfinite(namespace["traj"].states)))
