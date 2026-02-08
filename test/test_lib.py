import numpy as np
import pytest

from thermal_solver.components import (
    RadiationSurface,
)
from thermal_solver.constants import (
    P_SOLAR_W_PER_M2,
)
from thermal_solver.lib import Sun, FixedTemperatureNode
from thermal_solver.properties import (
    RadiationSurfaceProperties,
    RadiationInterfaceProperties,
)
from thermal_solver.vectors import (
    rotate_around_axis,
)


def setup_function():
    from thermal_solver.utils import NameGenerator
    NameGenerator._clear()


def sun_vector_getter(t: float) -> np.ndarray:
    """Start in [1, 0, 0] and rotate around z [0, 0, 1] at 360 deg / 24 hs"""
    w = 2 * np.pi / 24 / 3600
    return rotate_around_axis(
        [1, 0, 0], axis=[0, 0, 1], angle_rad=w*t, round_to=6
    )


def test_sun():

    sun = Sun(sun_vector_getter=sun_vector_getter)
    assert sun.name == 'Sun'
    # NOTE: orientation = -position (to match criteria used for surfaces)
    assert all(np.isclose(sun.get_orientation(t=0), [-1, 0, 0]))
    assert all(np.isclose(sun.get_orientation(
        t=3 * 3600), [-1 / 2**.5, -1 / 2**.5, 0]))
    assert all(np.isclose(sun.get_orientation(t=6 * 3600), [0, -1, 0]))
    assert all(np.isclose(sun.get_orientation(t=12 * 3600), [+1, 0, 0]))
    assert all(np.isclose(sun.get_orientation(t=18 * 3600), [0, +1, 0]))
    assert all(np.isclose(sun.get_orientation(t=24 * 3600), [-1, 0, 0]))

    assert sun.calculate_heat_transfered_W(
        t=0,
        area_exposed_m2=1,
        orientation=[1, 0, 0],
        absorptivity=0.2,
    ) == pytest.approx(P_SOLAR_W_PER_M2 * 0.2)
    assert sun.calculate_heat_transfered_W(
        t=3 * 3600,
        area_exposed_m2=1,
        orientation=[1, 0, 0],
        absorptivity=0.2,
    ) == pytest.approx(P_SOLAR_W_PER_M2 * 0.2 / 2**.5)
    assert sun.calculate_heat_transfered_W(
        t=6 * 3600,
        area_exposed_m2=1,
        orientation=[1, 0, 0],
        absorptivity=0.2,
    ) == 0
    assert sun.calculate_heat_transfered_W(
        t=9 * 3600,
        area_exposed_m2=1,
        orientation=[1, 0, 0],
        absorptivity=0.2,
    ) == 0
    assert sun.calculate_heat_transfered_W(
        t=12 * 3600,
        area_exposed_m2=1,
        orientation=[1, 0, 0],
        absorptivity=0.2,
    ) == 0
    assert sun.calculate_heat_transfered_W(
        t=0,
        area_exposed_m2=1,
        orientation=[-1, 0, 0],
        absorptivity=0.2,
    ) == 0
    assert sun.calculate_heat_transfered_W(
        t=0,
        area_exposed_m2=1,
        orientation=[-1 / 2**.5, -1 / 2**.5, 0],
        absorptivity=0.2,
    ) == 0
    assert sun.calculate_heat_transfered_W(
        t=0,
        area_exposed_m2=1,
        orientation=[+1 / 2**.5, +1 / 2**.5, 0],
        absorptivity=0.2,
    ) == pytest.approx(P_SOLAR_W_PER_M2 * 0.2 / 2**.5)

    surface = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=3, orientation=[1, 0, 0],
            emissivity=0.5, solar_absorptivity=0.2,
            # emission_spectrum=
        ),
        name='a',
    )
    surface.add_input_interface(sun, properties=RadiationInterfaceProperties(
        view_factor=0.9,
    ))
    assert surface.calculate_received_heat_power_W(t=0) == pytest.approx(
        sum(-x.q_out_W for x in surface.get_input_heat_fluxes(t=0))
    )
    assert sum(-x.q_out_W for x in surface.get_input_heat_fluxes(t=0)) == pytest.approx(
        P_SOLAR_W_PER_M2
        * 3  # area
        * 0.9  # view factor
        * 0.2  # solar_absorptivity
        * 1  # vectors aligned
    )
    assert sum(x.q_out_W for x in surface.get_input_heat_fluxes(t=6 * 3600)) == 0


def test_fixed_temperature_node():
    node = FixedTemperatureNode(temperature_K=300)
    assert node.temperature_K == 300
    node.temperature_K = 350
    assert node.temperature_K == 300
