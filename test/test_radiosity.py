import numpy as np
import pytest

from thermal_solver.radiosity import solve_radiosity


def test_radiosity():

    class Surface:
        def __init__(self, area_m2, emmissivity):
            self.area_m2 = area_m2
            self.emmissivity = emmissivity

    surfaces = [Surface(10, 0.8), Surface(15, 0.9), Surface(1, 0.5)]
    temperatures_K = [40, 60, 80]
    # NOTE: The following view factor values fullfill the requirement that Ai * F_i->j == Aj * F_j->i
    view_factor_matrix = [
        [0.0, 1.0, 0.0],
        [1.0 * 10 / 15, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    J, G, Q = solve_radiosity(
        surfaces=surfaces,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
    )

    assert np.allclose(J, [0.20707594, 0.68210007, 1.16129268])
    assert np.allclose(G, [0.45473338, 0.20707594, 0.])
    assert np.allclose(Q, [-2.47657434, 7.12536184,  1.16129268])

    # Check that the solver catches view factor assymmettry
    # Originally, A0 * F_0->1 = A1 * F_1->0 = 10
    # Modified so F_1->0 = 11 / 15, therefore  (A0 * F_0->1 = 10) != (A1 * F_1->0 = 11)
    view_factor_matrix[1][0] = 1.0 * 11 / 15  # Instead of 1.0 * 10 / 15
    with pytest.raises(ValueError):
        solve_radiosity(
            surfaces=surfaces,
            temperatures_K=temperatures_K,
            view_factor_matrix=view_factor_matrix,
        )
