import numpy as np
import pandas as pd
import pytest

from thermal_solver.components import (
    HeatSource,
    RadiationSurface,
    ConductionComponent,
    HeatFluxElement,
)
from thermal_solver.lib import Sun, FixedTemperatureNode
from thermal_solver.node import Node
from thermal_solver.properties import (
    Spectrum,
    NodeProperties,
    RadiationSurfaceProperties,
    HeatSourceProperties,
    RadiationInterfaceProperties,
    ConductionProperties,
    ConductionInterfaceProperties,
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


def heat_flux_element_to_flat_dict(element: HeatFluxElement) -> dict:
    return {
        # 'dest_node': element.dest.node,  # Node id and Node name?
        'dest_component': element.dest.name,  # Component id and Component name?
        # 'source_node': element.source.node, # Node id and Node name?
        'source_component': element.source.name,  # Component id and Component name?
        'iface_properties': element.iface_properties,
        'q_out_W': element.q_out_W,
    }


def test_node():
    properties = NodeProperties(mass_kg=2, specific_heat_J_per_kg_per_K=300)
    node = Node(properties=properties)
    other_node = FixedTemperatureNode(temperature_K=300)

    # Create node surfaces
    surface_1 = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=3, orientation=[1, 0, 0],
            emissivity=0.5, solar_absorptivity=0.25,
            emission_spectrum=Spectrum.IR,
        ),
        name='surface_1'
    )
    surface_2 = RadiationSurface(
        properties=RadiationSurfaceProperties(
            area_m2=8, orientation=[-1 / 2**.5, -1 / 2**.5, 0],
            emissivity=0.33, solar_absorptivity=0.3,
            emission_spectrum=Spectrum.IR,
        ),
        name='surface_2'
    )

    # Create sun
    sun = Sun(sun_vector_getter=sun_vector_getter)

    # Create interfaces between surfaces and sun
    surface_1.add_input_interface(
        sun, properties=RadiationInterfaceProperties())
    surface_2.add_input_interface(
        sun, properties=RadiationInterfaceProperties())

    # Create interfaces between surface_1 and surface_2
    surface_1.add_input_interface(surface_2, properties=RadiationInterfaceProperties(
        view_factor=1
    ))

    # Create heat source
    heat_source_1 = HeatSource(
        properties=HeatSourceProperties(constant_power_W=100),
        name='heat_source_1'
    )

    # Create conduction components
    component_1 = ConductionComponent(
        properties=ConductionProperties(), name='component_1'
    )
    component_2 = ConductionComponent(
        properties=ConductionProperties(), name='component_2'
    )
    component_1.add_input_interface(
        source=component_2,
        properties=ConductionInterfaceProperties(conductance_W_per_K=10),
    )

    # Add surfaces and heat sources to node
    node.add_component(surface_1)
    node.add_component(surface_2)
    node.add_component(heat_source_1)
    node.add_component(component_1)

    other_node.add_component(component_2)

    # Set the node temperature
    node.temperature = 350

    # Don't check numbers as they are checked in unit tests of each class, but check consistency
    for t in range(0, 24, 3):
        assert node.calculate_neat_heat_power_out_W(t=t) == pytest.approx(
            surface_1.calculate_neat_heat_power_out_W(t=t)
            + surface_2.calculate_neat_heat_power_out_W(t=t)
            + heat_source_1.calculate_neat_heat_power_out_W(t=t)
            + component_1.calculate_neat_heat_power_out_W(t=t)
        )
        assert surface_1.calculate_neat_heat_power_out_W(t=t) == pytest.approx(
            surface_1.calculate_emmited_heat_power_W(t=t)
            - surface_1.calculate_received_heat_power_W(t=t)
        )
        assert surface_2.calculate_neat_heat_power_out_W(t=t) == pytest.approx(
            surface_2.calculate_emmited_heat_power_W(t=t)
            - surface_2.calculate_received_heat_power_W(t=t)
        )
        assert surface_1.calculate_received_heat_power_W(t=t) == pytest.approx(
            sun.calculate_heat_transfered_W(
                t=t,
                area_exposed_m2=3,
                orientation=[1, 0, 0],
                absorptivity=0.25
            ) + surface_2.calculate_heat_transfered_W(
                t=t, area_exposed_m2=3*1, orientation=[1, 0, 0], absorptivity=0.5
            )
        )
        assert surface_2.calculate_received_heat_power_W(t=t) == pytest.approx(
            sun.calculate_heat_transfered_W(
                t=t,
                area_exposed_m2=8,
                orientation=[-1 / 2**.5, -1 / 2**.5, 0],
                absorptivity=0.3
            ) + surface_1.calculate_heat_transfered_W(
                t=t,
                area_exposed_m2=8*3/8,
                orientation=[-1 / 2**.5, -1 / 2**.5, 0],
                absorptivity=0.33
            )
        )

    # Check Heat Flux elements
    heat_flux_elements = node.get_heat_fluxes_W(t=0)

    assert len(heat_flux_elements) == (
        + 3  # surface_1 (sun, surface_2, emmited radiation)
        + 3  # surface_2 (sun, surface_1, emmited radiation)
        + 1  # heat_source_1
        + 1  # component_1 (component_2)
    )
