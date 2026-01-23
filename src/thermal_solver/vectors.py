import numpy as np
from scipy.spatial.transform import Rotation as R


def versor(x: np.array) -> np.array:
    return x / np.linalg.norm(x)


def get_rotation_matrix_around_axis(
    axis: np.ndarray | list,
    angle_rad: float
) -> np.ndarray:
    # Normalize the axis
    axis = np.array(axis) / np.linalg.norm(axis)
    # Create a rotation object from axis-angle
    r = R.from_rotvec(axis * angle_rad)
    # Get the rotation matrix
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def rotate_around_axis(
    vector: np.ndarray | list,
    axis: np.ndarray | list,
    angle_rad: float,
    round_to: int | None = None
) -> np.ndarray:
    rotation_matrix = get_rotation_matrix_around_axis(axis, angle_rad)
    rotated_vector = rotation_matrix @ vector
    if round_to is not None:
        rotated_vector = rotated_vector.round(round_to)
    return rotated_vector
