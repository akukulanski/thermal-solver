

def get_sun_position(t: float) -> np.ndarray:
    """Start in [1, 0, 0] and rotate around z [0, 0, 1] at 360 deg / 24 hs"""
    w = 2 * np.pi / 24 / 3600
    return rotate_around_axis(
        [1, 0, 0], axis=[0, 0, 1], angle_rad=w*t, round_to=6
    )
