"""Shape and parameter validation helpers."""
import jax
import jax.numpy as jnp
import pytest

from jax_da._validation import check_event_shape, is_concrete


@pytest.mark.parametrize("shape", [(3,), (2, 3), (0, 3), (4, 5, 3)])
def test_trailing_shape_accepted_for_any_batch(shape):
    check_event_shape(jnp.zeros(shape), (3,))


@pytest.mark.parametrize("shape", [(2,), (3, 2), ()])
def test_wrong_or_missing_event_axis_rejected(shape):
    with pytest.raises(ValueError, match=r"\(3,\)"):
        check_event_shape(jnp.zeros(shape), (3,))


def test_shape_check_runs_under_jit():
    f = jax.jit(lambda x: (check_event_shape(x, (3,)), x.sum())[1])
    assert f(jnp.ones(3)) == 3.0
    with pytest.raises(ValueError):
        f(jnp.ones(2))


def test_is_concrete_false_for_tracers_and_placeholders():
    assert is_concrete(1.0) and is_concrete(jnp.ones(2))
    assert not is_concrete(object())
    jax.jit(lambda x: (assert_false(is_concrete(x)), x)[1])(1.0)


def assert_false(value):
    assert value is False
