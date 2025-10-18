from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

from .constants import (
    STEFAN_BOLTZMANN_W_PER_M2_PER_K4,
    P_SOLAR_W_PER_M2,
)
from .vectors import (
    versor,
)


def _get_func_name_():
    import inspect
    return inspect.currentframe().f_back.f_code.co_name


class Spectrum(Enum):
    IR = auto()
    VISIBLE = auto()


def calculate_effective_area_factor(orientation_a, orientation_b) -> float:
    """Return the dot procut between the two versors of the orientation of the
    surfaces, inverted in sign (opposing for positive factor), and at least 0"""
    v1, v2 = versor(orientation_a), versor(orientation_b)
    return max(0, -np.dot(v1, v2))


@dataclass(kw_only=True)
class NodeProperties:
    mass_kg: float
    specific_heat_J_per_kg_per_K: float  # Fe: 444 J / (kg * K)

    @property
    def thermal_capacity_J_per_K(self) -> float:
        return self.mass_kg * self.specific_heat_J_per_kg_per_K


class Node:

    def __init__(self, properties: NodeProperties):
        self.properties = properties
        self.radiation_surfaces: list[RadiationSurface] = []
        self.temperature: float = None

    def add_radiation_surface(self, surface: RadiationSurface):
        assert surface not in self.radiation_surfaces
        self.radiation_surfaces.append(surface)
        surface._assign_node(self)

    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        return sum([
            component.calculate_neat_heat_power_out_W(t=t)
            for component in self.radiation_surfaces
        ])


@dataclass(kw_only=True)
class RadiationSurfaceProperties:
    area_m2: float
    orientation: list[float, float, float]
    # NOTE: the total emissivity is considered, but with a
    # different parameter for solar absorptivity to be able
    # to model emissivity at two different wavelengths:
    # - Sun radiation at T~5800 K (visible light)
    # - Nodes radiation in the model at T<400 K (IR)
    emissivity: float
    solar_absorptivity: float = field(default=None)
    emission_spectrum: Spectrum = field(default=Spectrum.IR)

    def __post_init__(self):
        if self.solar_absorptivity is None:
            self.solar_absorptivity = self.emissivity

    def get_absorptivity(self, spectrum: Spectrum) -> float:
        if spectrum == Spectrum.IR:
            return self.emissivity
        elif spectrum == Spectrum.VISIBLE:
            return self.solar_absorptivity
        else:
            raise ValueError(f'Unknown absorptivity for spectrum: {spectrum}')


@dataclass(kw_only=True)
class ContactSurfaceProperties:
    area_m2: float  # [m2]
    conductivity_W_per_m2_per_K: float  # [W / (m2 * K)]
    node_a: object
    node_b: object


@dataclass(kw_only=True)
class RadiationInterfaceProperties:
    view_factor: float = field(default=1.0)  # from the perspective

    # def get_symmetric(self, area, target_area_m2) -> RadiationInterfaceProperties:
    #     # view_factor * area = new_view_factor * target_area_m2
    #     return RadiationInterfaceProperties(
    #         view_factor=self.view_factor * area / target_area_m2
    #     )


class RadiationSurface:

    def __init__(self, properties: RadiationSurfaceProperties):
        self.properties = properties
        self.input_interfaces: list[tuple[RadiationSurface,
                                          RadiationInterfaceProperties]] = []
        self.node: Node = None

    def _assign_node(self, node: Node):
        assert self.node is None, f'Surface already assigned to node ({self.node})'
        assert self in node.radiation_surfaces, (
            f'Incorrect use of internal method {self.__class__.__name__}.{_get_func_name_()}(): '
            f'Trying to assign node to surface, but surface not in Node.\n'
            f'Use Node.add_radiation_surface() instead.'
        )
        self.node: Node = node

    def add_input_interface(self, source: RadiationSurface, properties: RadiationInterfaceProperties):
        """Add input interface"""
        # NOTE: RadiationInterfaceProperties.view_factor is the view factor from the perspective
        # from THIS radiation surface (self.properties.area_m2), and NOT source.properties.area_m2.
        # The exposed area is then self.properties.area_m2 * properties.view_factor
        self.input_interfaces.append((source, properties))
        # FIXME: add symetric interface by default!

    def calculate_heat_transfered_W(
        self,
        t: float,
        area_exposed_m2: float,
        orientation: np.ndarray,
        absorptivity: float,
    ) -> float:
        effective_area_factor = calculate_effective_area_factor(
            self.properties.orientation, orientation
        )
        return (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * area_exposed_m2 * effective_area_factor
            * self.properties.emissivity * absorptivity
            * self.node.temperature ** 4
        )

    def calculate_heat_power_in_W(self, t: float):
        q_in_W: list = [
            source.calculate_heat_transfered_W(
                t=t,
                area_exposed_m2=self.properties.area_m2 * iface_properties.view_factor,
                orientation=self.properties.orientation,
                absorptivity=self.properties.get_absorptivity(
                    spectrum=source.properties.emission_spectrum),
            )
            for source, iface_properties in self.input_interfaces
        ]
        return sum(q_in_W)

    def calculate_heat_power_out_W(self, t: float) -> float:
        # TODO: add consistency check. Ensure that all the associated interfaces
        # are not adding up more power than the total output power.
        return (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * self.properties.emissivity
            * self.properties.area_m2
            * self.node.temperature ** 4
        )

    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        return self.calculate_heat_power_out_W(t=t) - self.calculate_heat_power_in_W(t=t)


class Sun(RadiationSurface):

    def __init__(self, sun_vector_getter: callable):
        self.sun_vector_getter = sun_vector_getter
        self.properties = RadiationSurfaceProperties(
            area_m2=None,
            orientation=None,
            emissivity=None,
            solar_absorptivity=None,
            emission_spectrum=Spectrum.VISIBLE,
        )

    def get_orientation(self, t: float = 0) -> np.ndarray | list:
        """Sun vector opposed in sign"""
        return -np.array(self.sun_vector_getter(t))

    def calculate_heat_transfered_W(
        self,
        t: float,
        area_exposed_m2: float,
        orientation: np.ndarray,
        absorptivity: float,
    ) -> float:
        sun_orientation = self.get_orientation(t)
        effective_area_factor = calculate_effective_area_factor(
            sun_orientation, orientation
        )
        return (
            P_SOLAR_W_PER_M2
            * area_exposed_m2 * effective_area_factor
            * absorptivity
        )

    def _assign_node(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')

    def add_input_interface(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')

    def calculate_heat_power_in_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')

    def calculate_heat_power_out_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')

    def calculate_neat_heat_power_out_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')
