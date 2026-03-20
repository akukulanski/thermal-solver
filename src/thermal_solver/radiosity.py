import numpy as np

from .constants import STEFAN_BOLTZMANN_W_PER_M2_PER_K4


def solve_radiosity(
    surfaces: list,
    temperatures_K: np.ndarray | list,
    view_factor_matrix: np.ndarray | list[list],
) -> tuple:
    """
    STEFAN_BOLTZMANN_W_PER_M2_PER_K4 = 5.670374419e-8  # [W / (m2 * K4)]

    class Surface:
        def __init__(self, area_m2, emmissivity):
            self.area_m2 = area_m2
            self.emmissivity = emmissivity

    surfaces = [Surface(10, 0.8), Surface(15, 0.9), Surface(1, 0.5)]
    temperatures_K = [40, 60, 80]
    view_factor_matrix = [
        [0.0, 1.0, 0.0],
        [1.0 * 10 / 15, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

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

    """

    # Inputs
    # Areas
    areas_m2 = np.array([s.area_m2 for s in surfaces], dtype=float)
    # Emissivities
    emmissivities = np.array([s.emmissivity for s in surfaces], dtype=float)
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

    print(f"\n=== Inputs ===")
    print(f"areas_m2={areas_m2}\n")
    print(f"emmissivities={emmissivities}\n")
    print(f"temperatures_K={temperatures_K}\n")
    print(f"view_factor_matrix=\n{view_factor_matrix}\n")

    # M . J = b
    M = np.eye(len(areas_m2)) - np.diag(1 -
                                        emmissivities) @ view_factor_matrix.T
    b = emmissivities * STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * temperatures_K**4

    print(f"\n=== System of Equations ===")
    print(f"M=\n{M}\n")
    print(f"b={b}\n")

    # Radiosity in [W/m2]
    J = np.linalg.solve(M, b)  # or scipy.linalg.solve(...)
    # Incomming radiation [W/m2]
    G = view_factor_matrix.T @ J
    # Neat heat out
    Q = areas_m2 * (J - G)

    print(f"\n=== Solutions ===")
    print(f"J={J}\n")
    print(f"G={G}\n")
    print(f"Q={Q}\n")

    return J, G, Q
