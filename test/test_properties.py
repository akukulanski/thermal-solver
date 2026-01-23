import pytest

from thermal_solver.properties import (
    Spectrum,
    NodeProperties,
    RadiationSurfaceProperties,
    HeatSourceProperties,
    RadiationInterfaceProperties,
    ConductionProperties,
    ConductionInterfaceProperties,
)


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
    assert properties.emission_spectrum == Spectrum.IR  # default is IR


def test_radiation_interface_properties():
    properties = RadiationInterfaceProperties(view_factor=0.7)
    assert properties.view_factor == 0.7

    properties = RadiationInterfaceProperties()
    assert properties.view_factor == 1.0  # default is 1


def test_heat_source_properties():
    properties = HeatSourceProperties(constant_power_W=0)
    assert properties.get_power_W(t=0) == 0
    properties = HeatSourceProperties(constant_power_W=3)
    assert properties.get_power_W(t=0) == 3
    properties = HeatSourceProperties(power_getter=lambda t: 2 * t)
    assert properties.get_power_W(t=0) == 0
    assert properties.get_power_W(t=3) == 6
    with pytest.raises(TypeError):
        HeatSourceProperties(constant_power_W=0, power_getter=lambda t: 2 * t)
    with pytest.raises(TypeError):
        HeatSourceProperties(constant_power_W=1, power_getter=lambda t: 2 * t)
    with pytest.raises(TypeError):
        HeatSourceProperties()


def test_conduction_properties():
    properties = ConductionProperties()
    # Nothing else your honor


def test_conduction_interface_properties():
    properties = ConductionInterfaceProperties(conductance_W_per_K=1)
    assert properties.conductance_W_per_K == 1

    properties = ConductionInterfaceProperties.from_area_and_conductivity(
        area_m2=2, conductivity_W_per_m2_per_K=3
    )
    assert properties.conductance_W_per_K == 6
