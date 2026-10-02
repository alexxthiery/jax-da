"""Geometry: distances against hand values and brute force."""
import numpy as np
import pytest

from jax_da.geometry import Ring, Torus2D, Unstructured


def test_ring_distances_hand_values():
    d = Ring(6).distances()
    assert d.shape == (6, 6)
    np.testing.assert_array_equal(d[0], [0, 1, 2, 3, 2, 1])
    np.testing.assert_array_equal(Ring(6).distances([5])[:, 0], [1, 2, 3, 2, 1, 0])


def test_torus_flattening_is_row_major_and_distances_match_brute_force():
    g = Torus2D(3, 4)
    np.testing.assert_array_equal(g.coordinates([0, 5, 11]), [[0, 0], [1, 1], [2, 3]])
    brute = np.zeros((12, 12))
    for a in range(12):
        for b in range(12):
            (i, j), (k, l) = divmod(a, 4), divmod(b, 4)
            di, dj = min(abs(i - k), 3 - abs(i - k)), min(abs(j - l), 4 - abs(j - l))
            brute[a, b] = np.hypot(di, dj)
    np.testing.assert_allclose(g.distances(), brute)
    np.testing.assert_allclose(g.distances([7]), brute[:, [7]])


def test_bad_indices_and_unstructured_rejected():
    with pytest.raises(ValueError):
        Ring(4).distances([4])
    with pytest.raises(NotImplementedError):
        Unstructured(3).distances()
