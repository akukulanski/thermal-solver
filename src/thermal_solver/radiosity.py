import numpy as np

from .constants import STEFAN_BOLTZMANN_W_PER_M2_PER_K4, DEEP_SPACE_TEMP_K


def solve_radiosity(
    surfaces: list,
    temperatures_K: np.ndarray | list,
    view_factor_matrix: np.ndarray | list[list],
    include_losses_to_deep_space: bool = False,
) -> tuple:
    """

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
    areas_m2 = np.array([s.area_m2 for s in surfaces], dtype=float)
    # Emissivities
    emissivities = np.array([s.emissivity for s in surfaces], dtype=float)
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
            f"Consistency check failed: The areas and view factors don't comply with 'Ai * F_i->j == Aj * F_j->i'.\n"
            f"Areas: {areas_m2}\n"
            f"View Factors F[i, j]:\n",
            f"{view_factor_matrix}\n"
            f"The following matrix should be symmetric:\n"
            f"{area_eff}"
        )

    # Complete the matrix with radiation to deep space to ensure the sum of view factors is 1.
    row_sums = view_factor_matrix.sum(axis=1)
    if np.any(row_sums > 1 + 1e-10):
        raise ValueError("View factor rows cannot exceed 1")

    if include_losses_to_deep_space:
        # Calculate expansion elements
        F_space = 1.0 - row_sums
        F_space = np.clip(F_space, 0.0, 1.0)  # Clip potential rounding errors

        # dummy_area = 6 # OK

        # Create the expanded matrix
        N = len(areas_m2)
        F_ext = np.zeros((N + 1, N + 1), dtype=float)
        F_ext[:N, :N] = view_factor_matrix
        F_ext[:N, N] = F_space
        # F_ext[N, N] = 1.0  # Space sees nothing back -> NO.
        # F_ext[N, :N] = areas_m2 / dummy_area * F_space # OK!
        aux = areas_m2 * F_space
        dummy_area = np.sum(aux)
        F_ext[N, :N] = aux / dummy_area
        assert np.isclose(sum(F_ext[N, :N]), 1), f"{sum(F_ext[N, :N])}"

        # Update the original vectors/matrixes with the expanded versions
        areas_m2 = np.append(areas_m2, dummy_area)  # dummy
        emissivities = np.append(emissivities, 1.0)
        temperatures_K = np.append(temperatures_K, DEEP_SPACE_TEMP_K)
        view_factor_matrix = F_ext

    print(f"\n=== Inputs ===")
    print(f"areas_m2={areas_m2}\n")
    print(f"emissivities={emissivities}\n")
    print(f"temperatures_K={temperatures_K}\n")
    print(f"view_factor_matrix=\n{view_factor_matrix}\n")

    # M . J = b
    M = np.eye(len(areas_m2)) - np.dot(
        # np.diag(1 - emissivities), view_factor_matrix.T
        np.diag(1 - emissivities), view_factor_matrix
    )
    b = emissivities * STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * temperatures_K**4

    if np.linalg.cond(M) > 1e12:
        raise ValueError("Radiosity matrix is ill-conditioned")

    print(f"\n=== System of Equations ===")
    print(f"M=\n{M}\n")
    print(f"b={b}\n")

    # Radiosity in [W/m2]
    J = np.linalg.solve(M, b)  # or scipy.linalg.solve(...)
    # Incomming radiation [W/m2]
    # G = view_factor_matrix.T @ J
    G = np.dot(view_factor_matrix, J)
    # Neat heat out
    Q = areas_m2 * (J - G)

    print(f"\n=== Solutions ===")
    print(f"J={J}")
    print(f"G={G}")
    print(f"Q={Q}")

    # Sanity check
    if include_losses_to_deep_space and np.abs(Q.sum()) > 1e-6:
        raise ValueError(f"Warning: Energy not conserved (Q={Q}; Q.sum() = {Q.sum()})")

    return J, G, Q
