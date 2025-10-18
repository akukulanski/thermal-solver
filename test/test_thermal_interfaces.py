import numpy as np
import pytest
from unittest import mock

from thermal_solver.constants import (
    STEFAN_BOLTZMANN_W_PER_M2_PER_K4,
    P_SOLAR_W_PER_M2,
)
from thermal_solver.vectors import (
    versor,
    rotate_around_axis,
)
from thermal_solver.thermal_interfaces import (
    NodeProperties,
    Node,
    RadiationSurfaceProperties,
    ContactSurfaceProperties,
    Spectrum,
    RadiationInterfaceProperties,
    RadiationSurface,
    Sun,
    ThermalSystem,
    SimpleSystemTwoNodes,
    calculate_effective_area_factor,
)


def get_sun_position(t: float) -> np.ndarray:
    """Start in [1, 0, 0] and rotate around z [0, 0, 1] at 360 deg / 24 hs"""
    w = 2 * np.pi / 24 / 3600
    return rotate_around_axis(
        [1, 0, 0], axis=[0, 0, 1], angle_rad=w*t, round_to=6
    )


def test_versor():
    assert all(versor([10, 0, 0]) == [1, 0, 0])
    assert all(np.isclose(versor([1, 1, 1]), [3**-.5, 3**-.5, 3**-.5]))


def test_calculate_effective_area_factor():
    assert calculate_effective_area_factor(
        orientation_a=[10, 0, 0], orientation_b=[-10, 0, 0]
    ) == 1
    assert calculate_effective_area_factor(
        orientation_a=[0, 10, 0], orientation_b=[10, 0, 0]
    ) == 0
    assert calculate_effective_area_factor(
        orientation_a=[1, 1, 0], orientation_b=[-1, 0, 0]
    ) == pytest.approx(1 / 2**.5)
    # Not opposed, never negative, then zero
    assert calculate_effective_area_factor(
        orientation_a=[10, 0, 0], orientation_b=[10, 0, 0]
    ) == 0


def test_rotate_around_axis():
    assert all(np.isclose(rotate_around_axis([1, 0, 0], [0, 0, 1], np.pi / 2), [0, 1, 0]))
    assert all(rotate_around_axis([1, 0, 0], [0, 0, 1], np.pi / 2, round_to=8) == [0, 1, 0])


def test_node_properties():
    properties = NodeProperties(mass_kg=2, specific_heat_J_per_kg_per_K=300)
    assert properties.thermal_capacity_J_per_K == 600


def test_radiation_surface_properties():
    properties = RadiationSurfaceProperties(
        area_m2=2,
        orientation=[1, 0, 0],
        emissivity=0.7,
        solar_absorptivity=0.2,
        emission_spectrum=Spectrum.VISIBLE,
    )
    assert properties.solar_absorptivity == 0.2
    assert properties.get_absorptivity(Spectrum.IR) == 0.7
    assert properties.get_absorptivity(Spectrum.VISIBLE) == 0.2
    assert properties.emission_spectrum == Spectrum.VISIBLE

    properties = RadiationSurfaceProperties(
        area_m2=2,
        orientation=[1, 0, 0],
        emissivity=0.7,
    )
    assert properties.solar_absorptivity == 0.7  # default is same as IR emissivity
    assert properties.get_absorptivity(Spectrum.IR) == 0.7
    assert properties.get_absorptivity(Spectrum.VISIBLE) == 0.7
    assert properties.emission_spectrum == Spectrum.IR # default is IR


def test_radiation_interface_properties():
    properties = RadiationInterfaceProperties(view_factor=0.7)
    assert properties.view_factor == 0.7

    properties = RadiationInterfaceProperties()
    assert properties.view_factor == 1.0  # default is 1


def test_radiation_surface():

    surface = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=2,
            orientation=[1, 0, 0],
            emissivity=0.7,
            solar_absorptivity=0.2,
        )
    )
    node = mock.MagicMock()
    node.temperature = 300
    node.radiation_surfaces = [surface]
    surface._assign_node(node)

    with pytest.raises(AssertionError):
        # Node already assigned
        surface._assign_node(node)

    expected_heat_power_out_W = (
        STEFAN_BOLTZMANN_W_PER_M2_PER_K4
        * 0.7
        * 2
        * 300**4
    )
    assert surface.calculate_heat_power_out_W(t=0) == pytest.approx(expected_heat_power_out_W)
    assert surface.calculate_heat_power_in_W(t=0) == 0
    assert surface.calculate_neat_heat_power_out_W(t=0) == pytest.approx(expected_heat_power_out_W)

    # Not opposing faces, dot product forced to 0.
    assert surface.calculate_heat_transfered_W(
        t=0, area_exposed_m2=2, orientation=[1, 0, 0], absorptivity=0.5
    ) == 0
    # Opposing faces, dot product > 0.
    assert surface.calculate_heat_transfered_W(
        t=0, area_exposed_m2=2, orientation=[-1, 0, 0], absorptivity=0.5
    ) == pytest.approx(STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * 0.7 * 2 * 1 * 0.5 * 300**4)
    assert surface.calculate_heat_transfered_W(
        t=0, area_exposed_m2=3, orientation=[-1, 0, 0], absorptivity=0.4
    ) == pytest.approx(STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * 0.7 * 3 * 1 * 0.4 * 300**4)
    assert surface.calculate_heat_transfered_W(
        t=0, area_exposed_m2=3, orientation=[-1, -1, 0], absorptivity=0.4
    ) == pytest.approx(STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * 0.7 * 3 * 1 * 0.4 / 2**.5 * 300**4)

    other_surface_1 = RadiationSurface(properties=RadiationSurfaceProperties(
        area_m2=4, orientation=[-1, 0, 0],
        emissivity=0.8, solar_absorptivity=0.1,
        emission_spectrum=Spectrum.IR
    ))
    other_surface_2 = RadiationSurface(properties=RadiationSurfaceProperties(
        area_m2=7, orientation=[-1 / 2**.5, -1 / 2**.5, 0],
        emissivity=0.4, solar_absorptivity=0.2,
        emission_spectrum=Spectrum.VISIBLE
    ))
    other_surface_3 = RadiationSurface(properties=RadiationSurfaceProperties(
        area_m2=9, orientation=[1 / 2**.5, 1 / 2**.5, 0],
        emissivity=0.4, solar_absorptivity=0.2,
        emission_spectrum=Spectrum.VISIBLE
    ))
    nodes = [mock.MagicMock() for _ in range(3)]
    nodes[0].radiation_surfaces = [other_surface_1]
    nodes[1].radiation_surfaces = [other_surface_2]
    nodes[2].radiation_surfaces = [other_surface_3]
    other_surface_1._assign_node(nodes[0])
    other_surface_2._assign_node(nodes[1])
    other_surface_3._assign_node(nodes[2])
    other_surface_1.node.temperature = 400
    other_surface_2.node.temperature = 500
    other_surface_3.node.temperature = 1000
    surface.add_input_interface(
        source=other_surface_1,
        properties=RadiationInterfaceProperties(view_factor=0.9)
    )
    surface.add_input_interface(
        source=other_surface_2,
        properties=RadiationInterfaceProperties(view_factor=0.8)
    )
    surface.add_input_interface(
        source=other_surface_3,
        properties=RadiationInterfaceProperties(view_factor=0.8)
    )

    expected_heat_power_in_W = (
        # from first surface
        (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * 2 # area of surface
            * 0.9  # view factor surface -> other_surface_1
            * 1 # same orientation
            * 0.8 # emissivity of other_surface_1
            * 0.7 # absorptivity of surface in IR
            * 400**4 # temperature of other_surface_1
        )
        # from second surface
        + (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * 2 # area of surface
            * 0.8  # view factor surface -> other_surface_2
            * 2**-.5 # 45 deg
            * 0.4 # emissivity of other_surface_2
            * 0.2 # absorptivity of surface in VISIBLE
            * 500**4 # temperature of other_surface_1
        )
        # from third surface (not opposing faces, 0)
        + (
            0
        )
    )
    assert surface.calculate_heat_power_in_W(t=0) == pytest.approx(expected_heat_power_in_W)
    assert surface.calculate_neat_heat_power_out_W(t=0) == pytest.approx(expected_heat_power_out_W - expected_heat_power_in_W)


def test_sun():

    sun = Sun(sun_vector_getter=get_sun_position)
    # NOTE: orientation = -position (to match criteria used for surfaces)
    assert all(np.isclose(sun.get_orientation(t=0), [-1, 0, 0]))
    assert all(np.isclose(sun.get_orientation(t=3 * 3600), [-1 / 2**.5, -1 / 2**.5, 0]))
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
        )
    )
    surface.add_input_interface(sun, properties=RadiationInterfaceProperties(
        view_factor=0.9,
    ))
    assert surface.calculate_heat_power_in_W(t=0) == pytest.approx(
        P_SOLAR_W_PER_M2
        * 3 # area
        * 0.9 # view factor
        * 0.2 # solar_absorptivity
        * 1 # vectors aligned
    )
    assert surface.calculate_heat_power_in_W(t=6 * 3600) == 0


@pytest.mark.skip(reason='Test not implemented')
def test_contact_surface_properties():
    ContactSurfaceProperties
    raise NotImplementedError()


def test_node():
    properties = NodeProperties(mass_kg=2, specific_heat_J_per_kg_per_K=300)
    node = Node(properties=properties)

    surface_1 = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=3, orientation=[1, 0, 0],
            emissivity=0.5, solar_absorptivity=0.25,
            emission_spectrum=Spectrum.IR,
        )
    )
    surface_2 = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=8, orientation=[-1 / 2**.5, -1 / 2**.5, 0],
            emissivity=0.33, solar_absorptivity=0.3,
            emission_spectrum=Spectrum.IR,
        )
    )
    sun = Sun(sun_vector_getter=get_sun_position)

    surface_1.add_input_interface(sun, properties=RadiationInterfaceProperties())
    surface_2.add_input_interface(sun, properties=RadiationInterfaceProperties())

    surface_1.add_input_interface(surface_2, properties=RadiationInterfaceProperties(
        view_factor=1
    ))
    surface_2.add_input_interface(surface_1, properties=RadiationInterfaceProperties(
        view_factor=3 / 8
    )) # FIXME: create symetric interface by default!

    node.add_radiation_surface(surface_1)
    node.add_radiation_surface(surface_2)

    node.temperature = 350

    # Don't check numbers as they are checked in unit tests of each class, but check consistency
    for t in range(0, 24, 3):
        assert node.calculate_neat_heat_power_out_W(t=t) == pytest.approx(
            surface_1.calculate_neat_heat_power_out_W(t=t)
            + surface_2.calculate_neat_heat_power_out_W(t=t)
        )
        assert surface_1.calculate_neat_heat_power_out_W(t=t) == pytest.approx(
            surface_1.calculate_heat_power_out_W(t=t)
            - surface_1.calculate_heat_power_in_W(t=t)
        )
        assert surface_2.calculate_neat_heat_power_out_W(t=t) == pytest.approx(
            surface_2.calculate_heat_power_out_W(t=t)
            - surface_2.calculate_heat_power_in_W(t=t)
        )
        assert surface_1.calculate_heat_power_in_W(t=t) == pytest.approx(
            sun.calculate_heat_transfered_W(t=t, area_exposed_m2=3, orientation=[1, 0, 0], absorptivity=0.25)
            + surface_2.calculate_heat_transfered_W(t=t, area_exposed_m2=3*1, orientation=[1, 0, 0], absorptivity=0.5)
        )
        assert surface_2.calculate_heat_power_in_W(t=t) == pytest.approx(
            sun.calculate_heat_transfered_W(t=t, area_exposed_m2=8, orientation=[-1 / 2**.5, -1 / 2**.5, 0], absorptivity=0.3)
            + surface_1.calculate_heat_transfered_W(t=t, area_exposed_m2=8*3/8, orientation=[-1 / 2**.5, -1 / 2**.5, 0], absorptivity=0.33)
        )
