import numpy as np
import pytest

from thermal_solver.radiosity import solve_radiosity


def test_radiosity():

    class Surface:
        def __init__(self, area_m2, emissivity):
            self.area_m2 = area_m2
            self.emissivity = emissivity

    surfaces = [Surface(10, 0.8), Surface(15, 0.9), Surface(1, 0.5)]
    temperatures_K = [40, 60, 80]
    # NOTE: The following view factor values fullfill the requirement that Ai * F_i->j == Aj * F_j->i
    view_factor_matrix = [
        [0.0, 1.0, 0.0],
        [1.0 * 10 / 15, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    # Keeping transpose:
    # J_expected = [0.20707594, 0.68210007, 1.16129268]
    # G_expected = [0.45473338, 0.20707594, 0.]
    # Q_expected = [-2.47657434,  7.12536184,  1.16129268]
    # G_expected_2 = G_expected
    # Q_expected_2 = Q_expected

    # Removing transpose:
    # J_expected = [2.51764645e-01, 6.78176882e-01, 1.16129419e+00]
    # G_expected = [0.67817678, 0.16784308, 0.]
    # Q_expected = [-4.26412156, 7.65500547, 1.16129268]
    # G_expected_2 = G_expected
    # Q_expected_2 = Q_expected

    # Removing transpose and fixing env view factor
    # J_expected = [2.51764645e-01, 6.78176882e-01, 1.16129419e+00, 3.01346945e-06][:-1]
    # G_expected = [6.78176882e-01, 1.67844101e-01, 3.01346945e-06, 7.58696433e-01][:-1]
    # Q_expected = [-4.26412238, 7.65499172, 1.16129117, -4.55216052][:-1]
    # G_expected_2 = G_expected
    # Q_expected_2 = Q_expected

    # Removing transpose, fixing env view factor and setting deep space temp to 2.7 K
    # G and Q not the same anymore for deep space as a surface and deep space ignored.
    J_expected = [2.51764645e-01, 6.78176882e-01, 1.16129419e+00]
    G_expected = [6.78176882e-01, 0.16784308, 0]
    Q_expected = [-4.26412156, 7.65500547, 1.16129268]
    G_expected_2 = [6.78176882e-01, 1.67844101e-01, 3.01346945e-06]
    Q_expected_2 = [-4.26412238, 7.65499172, 1.16129117]

    J, G, Q = solve_radiosity(
        surfaces=surfaces,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
    )

    assert np.allclose(J, J_expected)
    assert np.allclose(G, G_expected)
    assert np.allclose(Q, Q_expected)

    # Including the losses to deep space, the surfaces results should not change
    J, G, Q = solve_radiosity(
        surfaces=surfaces,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
        include_losses_to_deep_space=True,
    )

    assert np.allclose(J[:-1], J_expected)
    assert np.allclose(G[:-1], G_expected_2)
    assert np.allclose(Q[:-1], Q_expected_2)
    assert np.isclose(np.sum(Q), 0)

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
