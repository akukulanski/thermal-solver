import pytest
from unittest import mock

from thermal_solver.components import (
    HeatSource,
    RadiationSurface,
    ConductionComponent,
)
from thermal_solver.constants import (
    STEFAN_BOLTZMANN_W_PER_M2_PER_K4,
)
from thermal_solver.properties import (
    Spectrum,
    RadiationSurfaceProperties,
    HeatSourceProperties,
    RadiationInterfaceProperties,
    ConductionProperties,
    ConductionInterfaceProperties,
)


def setup_function():
    from thermal_solver.utils import NameGenerator
    NameGenerator._clear()


def test_radiation_surface():

    surface = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=2,
            orientation=[1, 0, 0],
            emissivity=0.7,
            solar_absorptivity=0.2,
        ),
        name='abc',
    )
    node = mock.MagicMock()
    node.temperature = 300
    node.components = [surface]
    surface._assign_node(node)

    expected_heat_power_out_W = (
        STEFAN_BOLTZMANN_W_PER_M2_PER_K4
        * 0.7
        * 2
        * 300**4
    )

    with pytest.raises(AssertionError):
        # Node already assigned
        surface._assign_node(node)

    assert surface.name == 'abc'
    assert surface.calculate_emmited_heat_power_W(
        t=0) == pytest.approx(expected_heat_power_out_W)
    assert surface.calculate_received_heat_power_W(t=0) == 0
    assert surface.calculate_neat_heat_power_out_W(
        t=0) == pytest.approx(expected_heat_power_out_W)

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

    other_surface_1 = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=4, orientation=[-1, 0, 0],
            emissivity=0.8, solar_absorptivity=0.1,
            emission_spectrum=Spectrum.IR
        ),
        name='a',
    )
    other_surface_2 = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=7, orientation=[-1 / 2**.5, -1 / 2**.5, 0],
            emissivity=0.4, solar_absorptivity=0.2,
            emission_spectrum=Spectrum.VISIBLE
        ),
        name='b',
    )
    other_surface_3 = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=9, orientation=[1 / 2**.5, 1 / 2**.5, 0],
            emissivity=0.4, solar_absorptivity=0.2,
            emission_spectrum=Spectrum.VISIBLE
        ),
        name='c',
    )
    nodes = [mock.MagicMock() for _ in range(3)]
    nodes[0].components = [other_surface_1]
    nodes[1].components = [other_surface_2]
    nodes[2].components = [other_surface_3]
    other_surface_1._assign_node(nodes[0])
    other_surface_2._assign_node(nodes[1])
    other_surface_3._assign_node(nodes[2])
    other_surface_1.node.temperature = 400
    other_surface_2.node.temperature = 500
    other_surface_3.node.temperature = 1000
    surface.add_input_interface(
        source=other_surface_1,
        properties=RadiationInterfaceProperties(view_factor=0.9),
        add_symmetric_interface=False,
    )
    with pytest.raises(ValueError):
        surface.add_input_interface(
            source=other_surface_1,
            properties=RadiationInterfaceProperties(view_factor=0.9),
            add_symmetric_interface=False,
        )
    surface.add_input_interface(
        source=other_surface_2,
        properties=RadiationInterfaceProperties(view_factor=0.8),
        add_symmetric_interface=False,
    )
    surface.add_input_interface(
        source=other_surface_3,
        properties=RadiationInterfaceProperties(view_factor=0.8),
        add_symmetric_interface=True,
    )

    iface = other_surface_3.get_source_interface(surface)
    assert not other_surface_1._source_in_interfaces(surface)
    assert not other_surface_2._source_in_interfaces(surface)
    assert other_surface_3._source_in_interfaces(surface)
    assert other_surface_3.input_interfaces[0][0] is surface
    assert other_surface_3.input_interfaces[0][1] is iface
    assert other_surface_3.properties.area_m2 * iface.view_factor == pytest.approx(
        surface.properties.area_m2 * 0.8
    )

    expected_heat_power_in_W = (
        # from first surface
        (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * 2  # area of surface
            * 0.9  # view factor surface -> other_surface_1
            * 1  # same orientation
            * 0.8  # emissivity of other_surface_1
            * 0.7  # absorptivity of surface in IR
            * 400**4  # temperature of other_surface_1
        )
        # from second surface
        + (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * 2  # area of surface
            * 0.8  # view factor surface -> other_surface_2
            * 2**-.5  # 45 deg
            * 0.4  # emissivity of other_surface_2
            * 0.2  # absorptivity of surface in VISIBLE
            * 500**4  # temperature of other_surface_1
        )
        # from third surface (not opposing faces, 0)
        + (
            0
        )
    )
    assert surface.calculate_received_heat_power_W(
        t=0) == pytest.approx(expected_heat_power_in_W)
    assert surface.calculate_neat_heat_power_out_W(t=0) == pytest.approx(
        expected_heat_power_out_W - expected_heat_power_in_W)


def test_heat_source():
    source = HeatSource(
        properties=HeatSourceProperties(constant_power_W=0),
        name='src_a'
    )
    assert source.name == 'src_a'
    assert source.calculate_neat_heat_power_out_W(t=0) == 0
    source = HeatSource(
        properties=HeatSourceProperties(constant_power_W=3),
        name='src_b'
    )
    assert source.name == 'src_b'
    assert source.calculate_neat_heat_power_out_W(
        t=0) == -3  # heat source, power out is negative
    source = HeatSource(
        properties=HeatSourceProperties(power_getter=lambda t: 2 * t),
        name='src_c'
    )
    assert source.name == 'src_c'
    assert source.calculate_neat_heat_power_out_W(t=0) == 0
    assert source.calculate_neat_heat_power_out_W(
        t=3) == -6  # heat source, power out is negative


def test_conduction_component():
    component_1 = ConductionComponent(
        properties=ConductionProperties(), name='a')
    component_2 = ConductionComponent(
        properties=ConductionProperties(), name='b')
    component_3 = ConductionComponent(
        properties=ConductionProperties(), name='c')
    component_1.add_input_interface(
        source=component_2,
        properties=ConductionInterfaceProperties(conductance_W_per_K=10),
    )
    component_1.add_input_interface(
        source=component_3,
        properties=ConductionInterfaceProperties(conductance_W_per_K=20),
        add_symmetric_interface=False,
    )
    assert component_1.name == 'a'
    assert component_2.name == 'b'
    assert component_3.name == 'c'
    assert component_1.get_source_interface(
        component_2).conductance_W_per_K == 10
    assert component_2.get_source_interface(
        component_1).conductance_W_per_K == 10
    assert component_1.get_source_interface(
        component_3).conductance_W_per_K == 20
    assert component_3.get_source_interface(component_1) is None  # not added

    # Test node assignment and power calculation
    node_1 = mock.MagicMock()
    node_1.temperature = 300
    node_1.components = [component_1]
    component_1._assign_node(node_1)
    node_2 = mock.MagicMock()
    node_2.temperature = 400
    node_2.components = [component_2, component_3]
    component_2._assign_node(node_2)
    component_3._assign_node(node_2)

    assert component_1.calculate_neat_heat_power_out_W(t=0) == pytest.approx(
        (300 - 400) * 10  # Out to component_2
        + (300 - 400) * 20  # Out to component_3
    )
    assert component_2.calculate_neat_heat_power_out_W(t=0) == pytest.approx(
        (400 - 300) * 10  # Out to component_1
    )
    assert component_3.calculate_neat_heat_power_out_W(
        t=0) == 0  # No input interfaces


def test_calculate_effective_area_factor():
    assert RadiationSurface.calculate_effective_area_factor(
        orientation_a=[10, 0, 0], orientation_b=[-10, 0, 0]
    ) == 1
    assert RadiationSurface.calculate_effective_area_factor(
        orientation_a=[0, 10, 0], orientation_b=[10, 0, 0]
    ) == 0
    assert RadiationSurface.calculate_effective_area_factor(
        orientation_a=[1, 1, 0], orientation_b=[-1, 0, 0]
    ) == pytest.approx(1 / 2**.5)
    # Not opposed, never negative, then zero
    assert RadiationSurface.calculate_effective_area_factor(
        orientation_a=[10, 0, 0], orientation_b=[10, 0, 0]
    ) == 0
