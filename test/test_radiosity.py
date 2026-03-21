import copy
import numpy as np
import pytest

from thermal_solver.radiosity import solve_radiosity


class Surface:
    def __init__(self, area_m2, emissivity):
        self.area_m2 = area_m2
        self.emissivity = emissivity


def test_radiosity():

    surfaces = [Surface(10, 0.8), Surface(15, 0.9), Surface(1, 0.5)]
    temperatures_K = [40, 60, 80]
    # NOTE: The following view factor values fullfill the requirement that Ai * F_i->j == Aj * F_j->i
    view_factor_matrix = [
        [0.0, 1.0, 0.0],
        [1.0 * 10 / 15, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    # Expected result
    J_expected = [2.51764645e-01, 6.78176882e-01, 1.16129419e+00]
    G_expected = [6.78176882e-01, 1.67844101e-01, 3.01346945e-06]
    Q_expected = [-4.26412238, 7.65499172, 1.16129117]

    J, G, Q = solve_radiosity(
        surfaces=surfaces,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
        infer_losses_to_deep_space=True,
    )

    assert np.allclose(J[:-1], J_expected)
    assert np.allclose(G[:-1], G_expected)
    assert np.allclose(Q[:-1], Q_expected)
    assert np.isclose(np.sum(Q), 0)

    # Check that the solver catches view factor assymmettry
    # Originally, A0 * F_0->1 = A1 * F_1->0 = 10
    # Modified so F_1->0 = 11 / 15, therefore  (A0 * F_0->1 = 10) != (A1 * F_1->0 = 11)
    bad_view_factor_matrix = copy.deepcopy(view_factor_matrix)
    bad_view_factor_matrix[1][0] = 1.0 * 11 / 15  # Instead of 1.0 * 10 / 15
    with pytest.raises(ValueError) as excinfo:
        solve_radiosity(
            surfaces=surfaces,
            temperatures_K=temperatures_K,
            view_factor_matrix=bad_view_factor_matrix,
        )
    assert f'Consistency check failed: The areas and view factors' in str(
        excinfo.value)

    # Check that the solver catches view factors not adding up 1 when
    # deactivating losses inference.
    with pytest.raises(ValueError) as excinfo:
        solve_radiosity(
            surfaces=surfaces,
            temperatures_K=temperatures_K,
            view_factor_matrix=view_factor_matrix,
            infer_losses_to_deep_space=False,
        )
    assert str(excinfo.value) == 'View factor rows must add up to 1'

    # Check that the solver catches view factors adding up more than 1
    bad_view_factor_matrix_2 = copy.deepcopy(view_factor_matrix)
    bad_view_factor_matrix_2[0][0] = 0.1
    with pytest.raises(ValueError) as excinfo:
        solve_radiosity(
            surfaces=surfaces,
            temperatures_K=temperatures_K,
            view_factor_matrix=bad_view_factor_matrix_2,
        )
    assert str(excinfo.value) == 'View factor rows cannot exceed 1'

    # Test no losses
    # Ensure all the rows sum 1
    view_factor_matrix = [
        [0.0, 1.0, 0.0],
        [1.0 * 10 / 15, 1 / 3, 0.0],
        [0.0, 0.0, 1.0],
    ]

    # Expected result
    J_expected = [0.25650712, 0.70188926, 2.32258536]
    G_expected = [0.70188926, 0.40496783, 2.32258536]
    Q_expected = [-4.45382136, 4.45382136, 0.0]

    J, G, Q = solve_radiosity(
        surfaces=surfaces,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
        infer_losses_to_deep_space=False,
    )

    assert np.allclose(J, J_expected)
    assert np.allclose(G, G_expected)
    assert np.allclose(Q, Q_expected)
