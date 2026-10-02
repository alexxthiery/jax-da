"""Observation operators: values, structure, Jacobians, validation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_da.observations import Elementwise, Linear, Selector


def test_selector_values_matrix_and_batches():
    op = Selector.every(10, 3)
    assert op.indices == (0, 3, 6, 9) and op.dim == 4
    x = jnp.arange(20.0).reshape(2, 10)
    np.testing.assert_array_equal(op.apply(x), x[:, [0, 3, 6, 9]])
    np.testing.assert_array_equal(x @ op.matrix.T, op.apply(x))


def test_random_orthonormal_rows_scaled_as_documented():
    op = Linear.random_orthonormal(jax.random.PRNGKey(0), 12, 5, row_scale_span=0.25)
    gram = op.matrix @ op.matrix.T
    np.testing.assert_allclose(gram, np.diag(np.linspace(0.75, 1.25, 5) ** 2), atol=1e-10)


def test_polynomial_degree_one_is_identity_and_arctan_saturates():
    base = Selector(4, (0, 2))
    x = jnp.array([-3.0, 1.0, 0.5, 2.0])
    np.testing.assert_allclose(Elementwise(base, "polynomial", 1).apply(x), [-3.0, 0.5])
    np.testing.assert_allclose(Elementwise(base, "polynomial", 3).apply(x),
                               [0.5 * -3 * (1 + 1.5 ** 2), 0.5 * 0.5 * (1 + 0.25 ** 2)])
    assert jnp.all(jnp.abs(Elementwise(base, "arctan").apply(1e6 * x)) < jnp.pi / 2 + 1e-9)


def test_jacobian_of_nonlinear_operator_matches_finite_differences():
    op = Elementwise(Linear.random_orthonormal(jax.random.PRNGKey(1), 6, 3), "polynomial", 3)
    x = jax.random.normal(jax.random.PRNGKey(2), (6,))
    jac = jax.jacfwd(op.apply)(x)
    eps = 1e-6
    fd = np.stack([(op.apply(x + eps * e) - op.apply(x - eps * e)) / (2 * eps) for e in np.eye(6)], axis=1)
    np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=1e-8)


def test_invalid_operators_and_shapes_rejected():
    with pytest.raises(ValueError):
        Selector(4, (0, 4))
    with pytest.raises(ValueError):
        Elementwise(Selector(4, (0,)), "polynomial", 2)
    with pytest.raises(ValueError):
        Selector(4, (0,)).apply(jnp.zeros(5))
