import numpy as np

from .vectors import versor


def _get_func_name_():
    import inspect
    return inspect.currentframe().f_back.f_code.co_name


def calculate_effective_area_factor(orientation_a, orientation_b) -> float:
    """Return the dot procut between the two versors of the orientation of the
    surfaces, inverted in sign (opposing for positive factor), and at least 0"""
    v1, v2 = versor(orientation_a), versor(orientation_b)
    return max(0, -np.dot(v1, v2))
