import numpy as np

from .constants import STEFAN_BOLTZMANN_W_PER_M2_PER_K4, DEEP_SPACE_TEMP_K


def solve_radiosity(
    areas_m2: np.ndarray | list,
    emissivities: np.ndarray | list,
    temperatures_K: np.ndarray | list,
    view_factor_matrix: np.ndarray | list[list],
    check_no_gaps: bool = False,
) -> tuple:
    """

    Imlementation.
    - A: Area
    - emm: Emissivity
    - F[i->j]: View factor from surface i to surface j
    - T: Temperature
    - J: Radiosity

    J[i] * A[i] = emi[i] * stef_bol * A[i] * T[i]**4 + (1 - emm[i]) * (A[0] *  F[0->i] * J[0] + A[1] * F[1->i] * J[1] + ...)
                = emi[i] * stef_bol * A[i] * T[i]**4 + (1 - emm[i]) * (A[i] *  F[i->0] * J[0] + A[i] * F[i->1] * J[1] + ...)
                = emi[i] * stef_bol * A[i] * T[i]**4 + (1 - emm[i]) *  A[i] * (F[i->0] * J[0] +        F[i->1] * J[1] + ...)
    J[i]        = emi[i] * stef_bol        * T[i]**4 + (1 - emm[i])         * (F[i->0] * J[0] +        F[i->1] * J[1] + ...)

    M . J = b

    For 3 surfaces it would expand to:

        [1 0 0]   [(1 - emi[0])*F[0->0]     (1 - emi[0])*F[0->1]    (1 - emi[0])*F[0->2]]
    M = [0 1 0] - [(1 - emi[1])*F[1->0]     (1 - emi[1])*F[1->1]    (1 - emi[1])*F[1->2]]
        [0 0 1]   [(1 - emi[2])*F[2->0]     (1 - emi[2])*F[2->1]    (1 - emi[2])*F[2->2]]

        [J0]
    J = [J1]
        [J2]

        [emm[0] * stef_bol * T[0]**4]
    b = [emm[1] * stef_bol * T[1]**4]
        [emm[2] * stef_bol * T[2]**4]


    G[i] = F[0->i] * J[i] + F[1->i] * J[i] + F[2->i] * J[i] + ...
         = sum[j=0 to j=N](F[j->i] * J[i])

    Q[i] =  J[i] * A[i] - sum[j=0 to j=N](A[j] * F[j->i] * J[j])
         =  J[i] * A[i] - sum[j=0 to j=N](A[i] * F[i->j] * J[j])
         =  J[i] * A[i] - sum[j=0 to j=N](A[i] * G[j])
         = (J[i]        - sum[j=0 to j=N](       G[j]))) * A[i]

    Q = A . (J - G)


    Args:
        surfaces: List of surfaces, each containing area_m2 and emmisivity
        temperatures_K: List of temperatures (consistent with surfaces)
        view_factor_matrix: Matrix with the view factors F_i->j (from i to j), as:
        [
            [F_0->0, F_0->1, F_0->2, ...],
            [F_1->0, F_1->1, F_1->2, ...],
            ...
        ]
        view_factor_matrix[i, j] is the fraction leaving i that arrives at j
        Note that the final view_factor_matrix will be (N+1)x(N+1), as the escaped
        radiation will be added ensuring that sum(F_i->*) = 1.

    """

    # Inputs
    # Areas
    areas_m2 = np.array(areas_m2, dtype=float)
    # Emissivities
    emissivities = np.array(emissivities, dtype=float)
    # Temperatures in K
    temperatures_K = np.array(temperatures_K, dtype=float)
    # NxN view-factor matrix
    view_factor_matrix = np.array(view_factor_matrix, dtype=float)

    # Check that Ai * F_i->j == Aj * F_j->i
    # areas_repeated = [
    #   [A0, A0, A0, ...],
    #   [A1, A1, A1, ...],
    #   ...
    # ]
    areas_repeated = np.array(
        [areas_m2 for _ in range(len(areas_m2))], dtype=float).T
    area_eff = areas_repeated * view_factor_matrix
    # Check for symmetry
    if not np.allclose(area_eff, area_eff.T):
        raise ValueError(
            f'Consistency check failed: The areas and view factors don\'t comply with "Ai * F_i->j == Aj * F_j->i".\n'
            f'Areas: {areas_m2}\n'
            f'View Factors F[i, j]:\n',
            f'{view_factor_matrix}\n'
            f'The following matrix should be symmetric:\n'
            f'{area_eff}'
        )

    # Complete the matrix with radiation to deep space to ensure the sum of view factors is 1.
    row_sums = view_factor_matrix.sum(axis=1)
    if np.any(row_sums > 1 + 1e-10):
        raise ValueError('View factor rows cannot exceed 1')

    if check_no_gaps and not np.allclose(row_sums, 1.0):
        raise ValueError('View factor rows must add up to 1')

    # M . J = b
    M = np.eye(len(areas_m2)) - np.dot(
        np.diag(1 - emissivities), view_factor_matrix
    )
    b = emissivities * STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * temperatures_K**4

    if np.linalg.cond(M) > 1e12:
        raise ValueError('Radiosity matrix is ill-conditioned')

    # Radiosity in [W/m2]
    J = np.linalg.solve(M, b)  # or scipy.linalg.solve(...)
    # Incomming radiation [W/m2]
    G = np.dot(view_factor_matrix, J)
    # Neat heat out
    Q = areas_m2 * (J - G)

    # Sanity check
    if np.allclose(row_sums, 1.0) and np.abs(Q.sum()) > 1e-6:
        # Something went wrong: No gaps but sum of Q non-zero.
        raise ValueError(
            f'Warning: Energy not conserved (Q={Q}; Q.sum() = {Q.sum()})')

    return J, G, Q


def fill_gaps(
    areas_m2: np.ndarray | list,
    emissivities: np.ndarray | list,
    temperatures_K: np.ndarray | list,
    view_factor_matrix: np.ndarray | list[list],
    external_temp_K: float = DEEP_SPACE_TEMP_K,
):
    # Areas
    # [s.area_m2 for s in surfaces], dtype=float)
    areas_m2 = np.array(areas_m2, dtype=float)
    # Emissivities
    # [s.emissivity for s in surfaces], dtype=float)
    emissivities = np.array(emissivities, dtype=float)
    # Temperatures in K
    temperatures_K = np.array(temperatures_K, dtype=float)
    # NxN view-factor matrix
    view_factor_matrix = np.array(view_factor_matrix, dtype=float)

    row_sums = view_factor_matrix.sum(axis=1)
    if np.any(row_sums > 1 + 1e-10):
        raise ValueError('View factor rows cannot exceed 1')

    # Calculate expansion elements
    F_space = 1.0 - row_sums
    F_space = np.clip(F_space, 0.0, 1.0)  # Clip potential rounding errors

    # Create the expanded matrix
    N = len(areas_m2)
    F_ext = np.zeros((N + 1, N + 1), dtype=float)
    F_ext[:N, :N] = view_factor_matrix
    F_ext[:N, N] = F_space
    aux = areas_m2 * F_space
    dummy_area = np.sum(aux)
    F_ext[N, :N] = aux / dummy_area
    assert np.isclose(sum(F_ext[N, :N]), 1), f'{sum(F_ext[N, :N])}'

    # Update the original vectors/matrixes with the expanded versions
    areas_m2 = np.append(areas_m2, dummy_area)  # dummy
    emissivities = np.append(emissivities, 1.0)
    temperatures_K = np.append(temperatures_K, external_temp_K)
    view_factor_matrix = F_ext

    return (
        areas_m2,
        emissivities,
        temperatures_K,
        view_factor_matrix,
    )
