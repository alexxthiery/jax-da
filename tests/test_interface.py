"""Shared contracts, parametrized over every public map, law, and conditional law."""
import jax
import jax.numpy as jnp
import pytest

import jax_da as jd
from jax_da.protocols import ConditionalLaw, Geometry, Law, Map

MAPS = {
    "Linear": lambda: jd.Linear.random_orthonormal(jax.random.PRNGKey(0), 6, 3),
    "Selector": lambda: jd.Selector.every(6, 2),
    "Elementwise": lambda: jd.Elementwise(jd.Selector.every(6, 2), "arctan"),
    "Function": lambda: jd.Function(lambda x: jnp.tanh(x[:3] * x[3:]), 6, 3),
    "Lorenz63": lambda: jd.Lorenz63(),
    "Lorenz96": lambda: jd.Lorenz96(dim=10),
    "Lorenz96TwoScale": lambda: jd.Lorenz96TwoScale(n_slow=4, n_fast=3),
    "TanhSquared": lambda: jd.TanhSquared(jnp.eye(3)),
    "KuramotoSivashinsky": lambda: jd.KuramotoSivashinsky(num_points=32),
    "KolmogorovFlow": lambda: jd.KolmogorovFlow(resolution=8),
}
PDE = {"KuramotoSivashinsky", "KolmogorovFlow"}
LAWS = {
    "Gaussian": lambda: jd.Gaussian.isotropic(3, 0.5, loc=1.0),
    "StudentT": lambda: jd.StudentT(3, df=4.0, loc=1.0),
    "Cauchy": lambda: jd.Cauchy(3),
    "Laplace": lambda: jd.Laplace(3),
    "GaussianMixture": lambda: jd.GaussianMixture(3),
    "PointMass": lambda: jd.PointMass(jnp.arange(3.0)),
    "Embedded": lambda: jd.Embedded(jd.Gaussian.isotropic(2, 1.0), jnp.array([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]])),
    "History": lambda: jd.History(jd.Gaussian.isotropic(1, 1.0), jd.Additive(jd.Linear(jnp.eye(1)), jd.Gaussian.isotropic(1, 0.5)), 2),
}
NO_DENSITY = {"PointMass", "Embedded"}
CONDITIONAL = {
    "Additive": lambda: jd.Additive(jd.Selector.every(6, 2), jd.Gaussian.isotropic(3, 0.5)),
    "Multiplicative": lambda: jd.Multiplicative(jd.Linear(0.5 * jnp.eye(6)[:3]), jd.Gaussian.isotropic(3, 1.0)),
    "Poisson": lambda: jd.Poisson(jd.Linear(0.3 * jnp.eye(6)[:3], offset=0.5)),
    "Precomposed": lambda: jd.Precomposed(jd.Additive(jd.Linear(jnp.ones((3, 2))), jd.Gaussian.isotropic(3, 1.0)),
                                          jd.Selector(6, (1, 4))),
    "Lagged": lambda: jd.Lagged(jd.Additive(jd.Linear(0.9 * jnp.eye(2)), jd.Gaussian.isotropic(2, 0.3)), 2),
}


def test_every_public_component_is_covered():
    public = set(jd.dynamics.__all__) | {"Linear", "Selector", "Elementwise", "Function", "Gaussian", "StudentT",
                                          "Cauchy", "Laplace", "GaussianMixture", "PointMass", "Embedded", "History",
                                          "Additive", "Multiplicative", "Poisson", "Precomposed", "Lagged"}
    assert public == set(MAPS) | set(LAWS) | set(CONDITIONAL)
    assert public <= set(jd.__all__)


def test_user_written_components_plug_in_without_subclassing():
    class Drift:  # a user map: in_dim, dim, __call__
        in_dim = dim = 2

        def __call__(self, x):
            return 0.9 * x

    class SignObservation:  # a user conditional law: y = sign(x_0) with probability 0.9
        in_dim, dim = 2, 1

        def mean(self, x):
            return 0.8 * jnp.sign(x[..., :1])

        def sample(self, key, x):
            flip = jax.random.bernoulli(key, 0.1, x.shape[:-1] + (1,))
            return jnp.where(flip, -1.0, 1.0) * jnp.sign(x[..., :1])

        def log_prob(self, y, x):
            agree = (y[..., 0] == jnp.sign(x[..., 0]))
            return jnp.where(agree, jnp.log(0.9), jnp.log(0.1))

    assert isinstance(Drift(), Map) and isinstance(SignObservation(), ConditionalLaw)
    ssm = jd.StateSpaceModel(jd.Gaussian.isotropic(2, 1.0), jd.Additive(Drift(), jd.Gaussian.isotropic(2, 0.1)),
                             SignObservation())
    traj = ssm.simulate(jax.random.PRNGKey(0), 3)
    assert traj.states.shape == (3, 2) and ssm.log_likelihood(traj.observations, traj.states).shape == (3,)


def map_params():
    return [pytest.param(name, marks=pytest.mark.pde) if name in PDE else name for name in MAPS]


@pytest.mark.parametrize("name", map_params())
def test_map_contract(name):
    if name in PDE:
        pytest.importorskip("exponax")
    f = MAPS[name]()
    assert isinstance(f, Map)
    x = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (2, 3, f.in_dim))
    assert f(x).shape == (2, 3, f.dim)
    assert jnp.allclose(jax.jit(lambda m, v: m(v))(f, x), f(x), rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError):
        f(jnp.zeros(f.in_dim + 1))
    if hasattr(f, "geometry"):
        assert isinstance(f.geometry, Geometry) and f.geometry.dim == f.dim
    if hasattr(f, "initial_condition"):
        assert f.initial_condition(jax.random.PRNGKey(1)).shape == (f.dim,)


@pytest.mark.parametrize("name", LAWS)
def test_law_contract(name):
    law = LAWS[name]()
    assert isinstance(law, Law)
    draws = law.sample(jax.random.PRNGKey(0), (4, 2))
    assert draws.shape == (4, 2, 3)
    if name in ("Cauchy", "History"):
        with pytest.raises(NotImplementedError):
            law.cov()
    else:
        assert law.cov().shape == (3, 3)
    if name in NO_DENSITY:
        with pytest.raises(ValueError):
            law.log_prob(draws)
        return
    assert law.log_prob(draws).shape == (4, 2)
    assert jnp.allclose(jax.jit(lambda l, v: l.log_prob(v))(law, draws), law.log_prob(draws))
    with pytest.raises(ValueError):
        law.log_prob(jnp.zeros((4, 2)))


@pytest.mark.parametrize("name", CONDITIONAL)
def test_conditional_law_contract(name):
    law = CONDITIONAL[name]()
    assert isinstance(law, ConditionalLaw)
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 2, law.in_dim))
    out = law.sample(jax.random.PRNGKey(1), x)
    assert out.shape == (4, 2, law.dim) and law.mean(x).shape == (4, 2, law.dim)
    assert law.log_prob(out, x).shape == (4, 2)
    assert jnp.allclose(jax.jit(lambda l, o, v: l.log_prob(o, v))(law, out, x), law.log_prob(out, x))
    with pytest.raises(ValueError):
        law.log_prob(jnp.zeros((4, 2, law.dim + 1)), x)
