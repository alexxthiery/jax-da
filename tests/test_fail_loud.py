"""Fail early and loud: every realistic mistake raises ValueError with a message naming the problem.

Each case below returned a plausible-looking result (or a cryptic broadcasting
error) before validation was added. The match string pins the message, so a
check that raises for the wrong reason does not pass.
"""
import jax
import jax.numpy as jnp
import pytest

import jax_da as jd
from jax_da import metrics

KEY = jax.random.PRNGKey(0)
OBS = jd.Additive(jd.Selector.every(4, 2), jd.Gaussian.isotropic(2, 1.0))
TRANSITION = jd.Additive(jd.Linear(0.9 * jnp.eye(4)), jd.Gaussian.isotropic(4, 0.1))
MODEL = jd.StateSpaceModel(jd.Gaussian.isotropic(4, 1.0), TRANSITION, OBS)

CONSTRUCTION = {
    "full cov not positive definite": (lambda: jd.Gaussian.full(jnp.array([[1.0, 2.0], [2.0, 1.0]])), "positive definite"),
    "full cov asymmetric": (lambda: jd.Gaussian.full(jnp.array([[1.0, 0.9], [0.0, 1.0]])), "symmetric"),
    "full cov not square": (lambda: jd.Gaussian.full(jnp.ones((2, 3))), "square"),
    "full cov with NaN": (lambda: jd.Gaussian.full(jnp.array([[jnp.nan, 0.0], [0.0, 1.0]])), "positive definite"),
    "diagonal std not a vector": (lambda: jd.Gaussian.diagonal(jnp.ones((2, 2))), "std"),
    "direct std length != dim": (lambda: jd.Gaussian(dim=3, std=jnp.ones(2)), r"std.*\(3,\)"),
    "loc length != dim": (lambda: jd.Gaussian.isotropic(3, 1.0, loc=jnp.ones(2)), r"loc.*\(3,\)"),
    "loc length 1 for dim 3": (lambda: jd.Gaussian.isotropic(3, 1.0, loc=jnp.ones(1)), r"loc.*\(3,\)"),
    "Gaussian without std or scale_tril": (lambda: jd.Gaussian(dim=3).sample(KEY), "isotropic, diagonal, or full"),
    "StudentT negative int df": (lambda: jd.StudentT(3, df=-1), "df must be positive"),
    "StudentT zero int scale": (lambda: jd.StudentT(3, scale=0), "scale must be positive"),
    "StudentT scale length != dim": (lambda: jd.StudentT(3, scale=jnp.ones(2)), r"scale.*\(3,\)"),
    "Laplace negative int scale": (lambda: jd.Laplace(2, scale=-1), "scale must be positive"),
    "Cauchy NaN loc": (lambda: jd.Cauchy(2, loc=jnp.array([0.0, jnp.nan])), "loc.*finite"),
    "PointMass scalar value": (lambda: jd.PointMass(jnp.array(1.0)), "PointMass value"),
    "Linear offset wrong length": (lambda: jd.Linear(jnp.eye(3), offset=jnp.ones(2)), r"offset.*\(3,\)"),
    "Linear offset length 1 for dim 3": (lambda: jd.Linear(jnp.eye(3), offset=jnp.ones(1)), r"offset.*\(3,\)"),
    "Linear matrix not 2D": (lambda: jd.Linear(jnp.ones(3)), "matrix"),
    "Linear matrix with NaN": (lambda: jd.Linear(jnp.array([[1.0, jnp.nan]])), "matrix.*finite"),
    "Selector duplicate indices": (lambda: jd.Selector(4, (1, 1)), "distinct"),
    "Selector non-integer indices": (lambda: jd.Selector(4, (0.5, 1.0)), "integer"),
    "Selector.every stride 0": (lambda: jd.Selector.every(4, 0), "stride"),
    "TanhSquared non-square B": (lambda: jd.TanhSquared(jnp.ones((3, 4))), "square"),
    "Additive with a plain function": (lambda: jd.Additive(lambda x: x, jd.Gaussian.isotropic(2, 1.0)), "jd.Function"),
    "Poisson with a plain function": (lambda: jd.Poisson(lambda x: x), "jd.Function"),
    "initial law is conditional": (lambda: jd.StateSpaceModel(TRANSITION, TRANSITION, OBS), "initial"),
    "transition is not conditional": (lambda: jd.StateSpaceModel(jd.Gaussian.isotropic(4, 1.0), jd.Gaussian.isotropic(4, 1.0), OBS), "transition"),
    "Lorenz96 zero substeps": (lambda: jd.Lorenz96(dim=8, substeps=0), "substeps"),
    "Lorenz63 zero substeps": (lambda: jd.Lorenz63(substeps=0), "substeps"),
    "Ring with no sites": (lambda: jd.Ring(0), "positive"),
    "Torus with no rows": (lambda: jd.Torus2D(0, 3), "positive"),
}

CALLS = {
    "Poisson negative count": (lambda: jd.Poisson(jd.Linear(jnp.eye(2))).log_prob(jnp.array([-1.0, 2.0]), jnp.zeros(2)), "counts"),
    "Poisson fractional count": (lambda: jd.Poisson(jd.Linear(jnp.eye(2))).log_prob(jnp.array([0.5, 2.0]), jnp.zeros(2)), "counts"),
    "simulate zero steps": (lambda: MODEL.simulate(KEY, 0), "n_steps"),
    "simulate negative steps": (lambda: MODEL.simulate(KEY, -3), "n_steps"),
    "simulate NaN x0": (lambda: MODEL.simulate(KEY, 3, x0=jnp.full(4, jnp.nan)), "x0.*finite"),
    "oracle observations wrong width": (lambda: jd.KalmanOracle.from_ssm(MODEL).filter(jnp.zeros((5, 3))), r"observations.*\(T, 2\)"),
    "oracle observations 1D": (lambda: jd.KalmanOracle.from_ssm(MODEL).filter(jnp.zeros(5)), r"observations.*\(T, 2\)"),
    "oracle observations NaN": (lambda: jd.KalmanOracle.from_ssm(MODEL).filter(jnp.full((5, 2), jnp.nan)), "observations.*finite"),
    "oracle smoother with singular prediction": (
        lambda: jd.KalmanOracle.from_ssm(MODEL.replace(initial=jd.PointMass(jnp.ones(4)),
                                                       transition=jd.Additive(jd.Linear(0.9 * jnp.eye(4))))).smooth(jnp.zeros((5, 2))),
        "singular"),
    "metrics NaN ensemble": (lambda: metrics.rmse(jnp.full((3, 5, 2), jnp.nan), jnp.zeros((3, 2))), "ensemble.*finite"),
    "metrics NaN truth": (lambda: metrics.crps(jnp.zeros((3, 5, 2)), jnp.full((3, 2), jnp.nan)), "truth.*finite"),
    "crps_gaussian zero std": (lambda: metrics.crps_gaussian(jnp.zeros(2), jnp.zeros(2), jnp.ones(2)), "std must be positive"),
    "crps_gaussian shape mismatch": (lambda: metrics.crps_gaussian(jnp.zeros(3), jnp.ones(2), jnp.ones(2)), "same shape"),
}

PRESETS = {
    "linear_gaussian_full obs_dim 0": (lambda: jd.problems.linear_gaussian_full(obs_dim=0), "obs_dim"),
    "stochastic_volatility beta 0": (lambda: jd.problems.stochastic_volatility(beta=0.0), "beta"),
    "stochastic_volatility sigma negative": (lambda: jd.problems.stochastic_volatility(sigma=-0.1), "sigma"),
    "advection_diffusion obs_every 0": (lambda: jd.problems.advection_diffusion((16,), obs_every=0), "obs_every"),
}


@pytest.mark.parametrize("name", CONSTRUCTION)
def test_invalid_construction_raises(name):
    build, message = CONSTRUCTION[name]
    with pytest.raises(ValueError, match=message):
        build()


@pytest.mark.parametrize("name", CALLS)
def test_invalid_call_raises(name):
    call, message = CALLS[name]
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize("name", PRESETS)
def test_invalid_preset_argument_raises(name):
    build, message = PRESETS[name]
    with pytest.raises(ValueError, match=message):
        build()


def test_valid_models_still_work_under_jit_vmap_and_rebuilds():
    """The checks above must not break the transformations the library promises."""
    traced = jax.jit(lambda m, k: m.simulate(k, 5).states)(MODEL, KEY)
    assert bool(jnp.all(jnp.isfinite(traced)))
    batched = jax.vmap(lambda s: jd.Gaussian.isotropic(3, s, loc=1.0))(jnp.array([0.5, 1.0]))
    assert batched.loc.shape == (2, 3)
    shapes = jax.tree.map(jnp.shape, MODEL)  # tuples as leaves: a common debugging rebuild
    assert shapes.geometry == MODEL.geometry
    grad = jax.grad(lambda s: jd.StudentT(3, df=4.0, scale=s).log_prob(jnp.ones(3)))(0.7)
    assert bool(jnp.isfinite(grad))
