import numpy as np

from thermal_solver.vectors import (
    versor,
    rotate_around_axis,
)


def setup_function():
    from thermal_solver.utils import NameGenerator
    NameGenerator._clear()


def test_versor():
    assert all(versor([10, 0, 0]) == [1, 0, 0])
    assert all(np.isclose(versor([1, 1, 1]), [3**-.5, 3**-.5, 3**-.5]))


def test_rotate_around_axis():
    assert all(np.isclose(rotate_around_axis(
        [1, 0, 0], [0, 0, 1], np.pi / 2), [0, 1, 0]))
    assert all(rotate_around_axis(
        [1, 0, 0], [0, 0, 1], np.pi / 2, round_to=8) == [0, 1, 0])
