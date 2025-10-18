import numpy as np
import pytest
from unittest import mock

from thermal_solver.thermal_interfaces import (
    STEFAN_BOLTZMANN_W_PER_M2_PER_K4,
    P_SOLAR_W_PER_M2,
    versor,
    calculate_effective_area_factor,
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
)

def test_versor():
    assert versor([10, 0, 0]) == [1, 0, 0]
    assert all(np.isclose(versor([1, 1, 1]), [3**-.5, 3**-.5, 3**-.5]))


def test_calculate_effective_area_factor():
    assert calculate_effective_area_factor(
        orientation_a=[10, 0, 0], orientation_b=[10, 0, 0]
    ) == 1
    assert calculate_effective_area_factor(
        orientation_a=[0, 10, 0], orientation_b=[10, 0, 0]
    ) == 0
    assert calculate_effective_area_factor(
        orientation_a=[1, 1, 0], orientation_b=[1, 0, 0]
    ) == pytest.approx(1 / 2**.5)



def test_node_properties():
    properties = NodeProperties(mass_kg=2, specific_heat_J_per_kg_per_K=300)
    assert properties.thermal_capacity_J_per_K == 600


def test_radiation_surface_properties():
    properties = RadiationSurfaceProperties(
        area_m2=2,
        orientation=[1, 0, 0],
        emissivity=0.7,
        solar_absorptivity=0.2,
    )
    assert properties.solar_absorptivity == 0.2
    assert properties.get_absorptivity(Spectrum.IR) == 0.7
    assert properties.get_absorptivity(Spectrum.VISIBLE) == 0.2

    properties = RadiationSurfaceProperties(
        area_m2=2,
        orientation=[1, 0, 0],
        emissivity=0.7,
    )
    assert properties.solar_absorptivity == 0.7  # default is same as IR emissivity
    assert properties.get_absorptivity(Spectrum.IR) == 0.7
    assert properties.get_absorptivity(Spectrum.VISIBLE) == 0.7


@pytest.mark.skip(reason='Test not implemented')
def test_contact_surface_properties():
    ContactSurfaceProperties


def test_radiation_interface_properties():
    properties = RadiationInterfaceProperties(
        view_factor=0.7,
        spectrum=Spectrum.VISIBLE,
    )
    assert properties.spectrum == Spectrum.VISIBLE

    properties = RadiationInterfaceProperties(
        view_factor=0.7,
    )
    assert properties.spectrum == Spectrum.IR  # default is IR


@pytest.mark.skip(reason='Test not implemented')
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
    surface.assign_node(node)

    expected_heat_power_out_W = (
        STEFAN_BOLTZMANN_W_PER_M2_PER_K4
        * 0.7
        * 2
        * 300**4
    )
    assert surface.calculate_heat_power_out_W(t=0) == pytest.approx(expected_heat_power_out_W)
    assert surface.calculate_heat_power_in_W(t=0) == 0
    assert surface.calculate_neat_heat_power_out_W(t=0) == pytest.approx(expected_heat_power_out_W)

    assert surface.calculate_heat_transfered(
        t=0, area_exposed_m2=2, orientation=[1, 0, 0], absorptivity=0.5
    ) == pytest.approx(STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * 0.7 * 2 * 1 * 0.5)

    assert surface.calculate_heat_transfered(
        t=0, area_exposed_m2=3, orientation=[1, 0, 0], absorptivity=0.4
    ) == pytest.approx(STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * 0.7 * 3 * 1 * 0.4)
    assert surface.calculate_heat_transfered(
        t=0, area_exposed_m2=3, orientation=[1, 1, 0], absorptivity=0.4
    ) == pytest.approx(STEFAN_BOLTZMANN_W_PER_M2_PER_K4 * 0.7 * 3 * 1 * 0.4 / 2**.5)

    other_surface_1 = RadiationSurface(properties=RadiationSurfaceProperties(
        area_m2=4, orientation=[1, 0, 0], emissivity=0.8, solar_absorptivity=0.1
    ))
    other_surface_2 = RadiationSurface(properties=RadiationSurfaceProperties(
        area_m2=7, orientation=[1 / 2**.5, 1 / 2**.5, 0], emissivity=0.4, solar_absorptivity=0.2
    ))
    other_surface_1.assign_node(mock.MagicMock())
    other_surface_2.assign_node(mock.MagicMock())
    other_surface_1.node.temperature = 400
    other_surface_2.node.temperature = 500
    surface.add_input_interface(
        source=other_surface_1,
        properties=RadiationInterfaceProperties(view_factor=0.9, spectrum=Spectrum.IR)
    )
    surface.add_input_interface(
        source=other_surface_2,
        properties=RadiationInterfaceProperties(view_factor=0.8, spectrum=Spectrum.VISIBLE)
        # FIXME: the emission spectrum should be part of RadiationSurface and not part of
        # the interface.
    )

    expected_heat_power_in_W = (
        # from first surface
        (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * 2 # area of surface
            * 0.9  # view factor surface -> other_surface_1
            * 1 # same orientation
            * 0.8 # emissivity of other_surface_1
            * 0.7 # emissivity or absorptivity of surface
            * 400**4 # temperature of other_surface_1
        )
        # from second surface
        + (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * 2 # area of surface
            * 0.8  # view factor surface -> other_surface_2
            * 2**-.5 # 45 deg
            * 0.4 # emissivity of other_surface_2
            * 0.7 # emissivity or absorptivity of surface
            * 500**4 # temperature of other_surface_1
        )
    )
    assert surface.calculate_heat_power_in_W(t=0) == pytest.approx(expected_heat_power_in_W)
    assert surface.calculate_neat_heat_power_out_W(t=0) == pytest.approx(expected_heat_power_out_W - expected_heat_power_in_W)


@pytest.mark.skip(reason='Test not implemented')
def test_sun():
    Sun


@pytest.mark.skip(reason='Test not implemented')
def test_node():
    properties = NodeProperties(mass_kg=2, specific_heat_J_per_kg_per_K=300)
    node = Node(properties=properties)

    surface_1 = RadiationSurface(
        properties=RadiationSurfaceProperties(area_m2=1, orientation=[1, 0, 0], emissivity=0.5, solar_absorptivity=0.2)
    )
    surface_2 = RadiationSurface(
        properties=RadiationSurfaceProperties(area_m2=12, orientation=[0, 1, 0], emissivity=0.3, solar_absorptivity=0.3)
    )
    node.add_radiation_surface(surface_1)
    node.add_radiation_surface(surface_2)

    raise NotImplementedError()
    assert node.calculate_neat_heat_power_out_W(t=0) == (

    )