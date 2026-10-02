"""Exponax-backed PDE models: conservation laws and batching (marker: pde)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.pde
pytest.importorskip("exponax")

from jax_da.dynamics.kolmogorov import KolmogorovFlow  # noqa: E402
from jax_da.dynamics.ks import KuramotoSivashinsky  # noqa: E402


def test_ks_conserves_the_spatial_mean_and_reaches_a_chaotic_field():
    m = KuramotoSivashinsky()
    x = m.spinup(m.initial_condition(jax.random.PRNGKey(0)) + 0.3, 300)
    np.testing.assert_allclose(x.mean(), 0.3 + float(m.initial_condition(jax.random.PRNGKey(0)).mean()), atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(x))) and 0.5 < float(x.std()) < 3.0


def test_kolmogorov_mean_vorticity_decays_by_the_drag():
    m = KolmogorovFlow(resolution=32)
    x = m.initial_condition(jax.random.PRNGKey(1)) + 0.2
    mean0 = float(x.mean())
    np.testing.assert_allclose(float(m(x).mean()), mean0 * np.exp(m.drag * m.dt), rtol=1e-4)


@pytest.mark.parametrize("model", [KuramotoSivashinsky(num_points=64), KolmogorovFlow(resolution=16)])
def test_batched_flow_matches_member_by_member(model):
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 3, model.dim)) * 0.5
    batched = model(x)
    assert batched.shape == x.shape
    np.testing.assert_allclose(batched[1, 2], model(x[1, 2]), rtol=1e-9, atol=1e-9)


def test_stepper_first_built_inside_jit_does_not_leak_tracers():
    model = KuramotoSivashinsky(num_points=48)  # a configuration no other test builds
    x = model.initial_condition(jax.random.PRNGKey(3))
    traced = jax.jit(lambda v: model(v))(x)
    np.testing.assert_allclose(model(x), traced, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("short, long", [
    (KuramotoSivashinsky(num_points=64, dt=0.5), KuramotoSivashinsky(num_points=64, dt=1.0)),
    (KolmogorovFlow(resolution=16, dt=0.1), KolmogorovFlow(resolution=16, dt=0.2)),
])
def test_flow_over_dt_equals_two_flows_over_half_dt(short, long):
    # Catches a wrong inner step count: the flow must integrate exactly dt.
    x = short.initial_condition(jax.random.PRNGKey(5))
    np.testing.assert_allclose(short(short(x)), long(x), rtol=1e-8, atol=1e-8)


def test_dt_must_be_a_multiple_of_the_inner_step():
    with pytest.raises(ValueError, match="multiple"):
        KuramotoSivashinsky(dt=0.3, dt_inner=0.25)
