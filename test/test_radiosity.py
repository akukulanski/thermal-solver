import copy
import numpy as np
import pytest

from thermal_solver.radiosity import solve_radiosity, fill_gaps


def test_radiosity():

    areas_m2 = [10, 15, 1]
    emissivities = [0.8, 0.9, 0.5]
    temperatures_K = [40, 60, 80]
    # NOTE: The following view factor values fullfill the requirement that Ai * F_i->j == Aj * F_j->i
    view_factor_matrix = [
        [0.0, 1.0, 0.0],
        [1.0 * 10 / 15, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

    # NOTE: not closed, there are gaps
    J_expected = [0.25176462, 0.67817678, 1.16129268,]
    G_expected = [0.67817678, 0.16784308, 0.,]
    Q_expected = [-4.26412156, 7.65500547, 1.16129268]

    J, G, Q = solve_radiosity(
        areas_m2=areas_m2,
        emissivities=emissivities,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
    )

    assert np.allclose(J, J_expected)
    assert np.allclose(G, G_expected)
    assert np.allclose(Q, Q_expected)
    # assert np.isclose(np.sum(Q), 0) # Not closed, so unbalanced Q is expected.

    # Filling the gaps with deep space T=0 should give the same result
    (
        areas_m2_2,
        emissivities_2,
        temperatures_K_2,
        view_factor_matrix_2,
    ) = fill_gaps(
        areas_m2=areas_m2,
        emissivities=emissivities,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
        external_temp_K=0,
    )
    J, G, Q = solve_radiosity(
        areas_m2=areas_m2_2,
        emissivities=emissivities_2,
        temperatures_K=temperatures_K_2,
        view_factor_matrix=view_factor_matrix_2,
    )

    assert np.allclose(J, [*J_expected, 0.0])
    assert np.allclose(G, [*G_expected, 0.7586961])
    assert np.allclose(Q, [*Q_expected, -4.55217658])
    assert np.isclose(np.sum(Q), 0)

    # Now let's fill the gaps but with deep space temp 2.7K (default)
    # The result will slightly change because of the non-zero temp.
    (
        areas_m2_3,
        emissivities_3,
        temperatures_K_3,
        view_factor_matrix_3,
    ) = fill_gaps(
        areas_m2=areas_m2,
        emissivities=emissivities,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
        # external_temp_K=,
    )

    # Expected result
    J_expected = [2.51764645e-01, 6.78176882e-01,
                  1.16129419e+00, 3.01346945e-06]
    G_expected = [6.78176882e-01, 1.67844101e-01,
                  3.01346945e-06, 7.58696433e-01]
    Q_expected = [-4.26412238, 7.65499172, 1.16129117, -4.55216052]

    J, G, Q = solve_radiosity(
        areas_m2=areas_m2_3,
        emissivities=emissivities_3,
        temperatures_K=temperatures_K_3,
        view_factor_matrix=view_factor_matrix_3,
    )

    assert np.allclose(J, J_expected)
    assert np.allclose(G, G_expected)
    assert np.allclose(Q, Q_expected)
    assert np.isclose(np.sum(Q), 0)

    # Check that the solver catches view factor assymmettry
    # Originally, A0 * F_0->1 = A1 * F_1->0 = 10
    # Modified so F_1->0 = 11 / 15, therefore  (A0 * F_0->1 = 10) != (A1 * F_1->0 = 11)
    bad_view_factor_matrix = copy.deepcopy(view_factor_matrix)
    bad_view_factor_matrix[1][0] = 1.0 * 11 / 15  # Instead of 1.0 * 10 / 15
    with pytest.raises(ValueError) as excinfo:
        solve_radiosity(
            areas_m2=areas_m2,
            emissivities=emissivities,
            temperatures_K=temperatures_K,
            view_factor_matrix=bad_view_factor_matrix,
        )
    assert f'Consistency check failed: The areas and view factors' in str(
        excinfo.value)

    # Check that the solver catches view factors not adding up 1 when
    # enabling no gaps check
    with pytest.raises(ValueError) as excinfo:
        solve_radiosity(
            areas_m2=areas_m2,
            emissivities=emissivities,
            temperatures_K=temperatures_K,
            view_factor_matrix=view_factor_matrix,
            check_no_gaps=True,
        )
    assert str(excinfo.value) == 'View factor rows must add up to 1'

    # Check that the solver catches view factors adding up more than 1
    bad_view_factor_matrix_2 = copy.deepcopy(view_factor_matrix)
    bad_view_factor_matrix_2[0][0] = 0.1
    with pytest.raises(ValueError) as excinfo:
        solve_radiosity(
            areas_m2=areas_m2,
            emissivities=emissivities,
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
        areas_m2=areas_m2,
        emissivities=emissivities,
        temperatures_K=temperatures_K,
        view_factor_matrix=view_factor_matrix,
        check_no_gaps=True,
    )

    assert np.allclose(J, J_expected)
    assert np.allclose(G, G_expected)
    assert np.allclose(Q, Q_expected)
