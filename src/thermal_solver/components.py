from __future__ import annotations

import abc
from dataclasses import dataclass, asdict
import numpy as np
from typing import TYPE_CHECKING, Any

from .constants import (
    STEFAN_BOLTZMANN_W_PER_M2_PER_K4,
)
from .properties import (
    RadiationSurfaceProperties,
    HeatSourceProperties,
    RadiationInterfaceProperties,
    ConductionProperties,
    ConductionInterfaceProperties,
)
from .utils import (
    get_func_name,
    NameGenerator,
)
from .vectors import versor

if TYPE_CHECKING:
    from .node import Node


__all__ = [
    'Component',
    'RadiationSurface',
    'HeatSource',
    'ConductionComponent',
]


@dataclass(kw_only=True)
class HeatFluxElement:
    dest: str
    source: str
    iface_properties: Any
    q_out_W: float

    def to_dict(self) -> dict:
        return asdict(self)


class Component(abc.ABC):

    _name_prefix: str = 'component_'

    def __init__(self, *, name: str = ''):
        self.node: Node | None = None
        self.name = NameGenerator.register_or_create(
            name, prefix=self._name_prefix)

    def _assign_node(self, node: Node):
        assert self.node is None, f'Component already assigned to node ({self.node})'
        assert self in node.components, (
            f'Incorrect use of internal method {self.__class__.__name__}.{get_func_name()}(): '
            f'Trying to assign node to component, but component not in Node.\n'
            f'Use Node.add_component() instead.'
        )
        self.node = node

    @abc.abstractmethod
    def get_neat_q_out_W(self, t: float) -> float:
        # return sum(x.q_out_W for x in self.get_heat_fluxes_W(t=t))
        raise NotImplementedError()

    @abc.abstractmethod
    def get_heat_fluxes_W(self, t: float) -> list[HeatFluxElement]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f'{self.name} ({self.__class__.__name__} - {hex(id(self))})'


class RadiationSurface(Component):

    _name_prefix: str = 'radiation_surface_'

    def __init__(self, properties: RadiationSurfaceProperties, *, name: str = ''):
        super().__init__(name=name)
        self.properties = properties
        self.input_interfaces: list[tuple[RadiationSurface,
                                          RadiationInterfaceProperties]] = []

    def _source_in_interfaces(self, source: RadiationSurface) -> bool:
        return self.get_source_interface(source) is not None

    def get_source_interface(self, source: RadiationSurface) -> RadiationInterfaceProperties | None:
        for src, props in self.input_interfaces:
            if src is source:
                return props
        return None

    def add_input_interface(self, source: RadiationSurface, properties: RadiationInterfaceProperties,
                            add_symmetric_interface: bool = True):
        """Add input interface"""
        # NOTE: RadiationInterfaceProperties.view_factor is the view factor from the perspective
        # from THIS radiation surface (self.properties.area_m2), and NOT source.properties.area_m2.
        # The exposed area is then self.properties.area_m2 * properties.view_factor
        if self._source_in_interfaces(source):
            raise ValueError('Source already added')
        self.input_interfaces.append((source, properties))
        if add_symmetric_interface:
            symmetric_interface_properties = properties.get_symmetric_properties(
                area_m2=self.properties.area_m2,
                target_area_m2=source.properties.area_m2,
            )
            source.add_input_interface(
                self, symmetric_interface_properties, add_symmetric_interface=False)

    def get_iface_q_out(self, t: float, source: RadiationSurface, properties: RadiationInterfaceProperties) -> float:
        return (-1) * source.calculate_heat_transfered_W(
            t=t,
            area_exposed_m2=self.properties.area_m2 * properties.view_factor,
            orientation=self.properties.orientation,
            absorptivity=self.properties.get_absorptivity(
                spectrum=source.properties.emission_spectrum
            )
        )

    def get_input_heat_fluxes(self, t) -> list[HeatFluxElement]:
        return [
            HeatFluxElement(
                dest=self.name,
                source=source.name,
                iface_properties=iface_properties,
                q_out_W=self.get_iface_q_out(t, source, iface_properties),
            )
            for source, iface_properties in self.input_interfaces
        ]

    def get_heat_fluxes_W(self, t: float) -> list[HeatFluxElement]:
        q_in_W = self.get_input_heat_fluxes(t=t)
        q_out_W = [
            HeatFluxElement(
                dest=self.name,
                source=NullComponent().name,
                iface_properties=None,
                q_out_W=self.calculate_emmited_heat_power_W(t=t),
            )
        ]
        return q_in_W + q_out_W

    def get_neat_q_out_W(self, t: float) -> float:
        return self.calculate_emmited_heat_power_W(t=t) + sum(
            self.get_iface_q_out(t, source, iface_properties)
            for source, iface_properties in self.input_interfaces
        )

    def calculate_heat_transfered_W(
        self,
        t: float,
        area_exposed_m2: float,
        orientation: np.ndarray | list,
        absorptivity: float,
    ) -> float:
        effective_area_factor = RadiationSurface.calculate_effective_area_factor(
            self.properties.orientation, orientation
        )
        return (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * area_exposed_m2 * effective_area_factor
            * self.properties.emissivity * absorptivity
            * self.node.temperature_K ** 4
        )

    def calculate_emmited_heat_power_W(self, t: float) -> float:
        # TODO: add consistency check. Ensure that all the associated interfaces
        # are not adding up more power than the total output power.
        return (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * self.properties.emissivity
            * self.properties.area_m2
            * self.node.temperature_K ** 4
        )

    def calculate_received_heat_power_W(self, t: float) -> float:
        return sum(-x.q_out_W for x in self.get_input_heat_fluxes(t=t))

    @classmethod
    def calculate_effective_area_factor(
        cls, orientation_a: np.ndarray | list, orientation_b: np.ndarray | list
    ) -> float:
        """Return the dot procut between the two versors of the orientation of the
        surfaces, inverted in sign (opposing for positive factor), and at least 0"""
        v1, v2 = versor(np.array(orientation_a)), versor(
            np.array(orientation_b))
        return max(0, -np.dot(v1, v2))


class HeatSource(Component):

    _name_prefix: str = 'heat_source_'

    def __init__(self, properties: HeatSourceProperties, *, name: str = ''):
        super().__init__(name=name)
        self.properties = properties

    def get_heat_fluxes_W(self, t: float) -> list[HeatFluxElement]:
        return [
            HeatFluxElement(
                dest=self.name,
                source=NullComponent().name,  # self.name or NullComponent().name?
                iface_properties=None,
                q_out_W=-self.properties.power_getter(t=t),
            )
        ]

    def get_neat_q_out_W(self, t: float) -> float:
        return -self.properties.power_getter(t=t)


class ConductionComponent(Component):

    _name_prefix: str = 'conduction_component_'

    def __init__(self, properties: ConductionProperties, *, name: str = ''):
        super().__init__(name=name)
        self.properties = properties
        self.input_interfaces: list[tuple[ConductionComponent,
                                          ConductionInterfaceProperties]] = []

    def _source_in_interfaces(self, source: ConductionComponent) -> bool:
        return self.get_source_interface(source) is not None

    def get_source_interface(self, source: ConductionComponent) -> ConductionInterfaceProperties | None:
        for src, props in self.input_interfaces:
            if src is source:
                return props
        return None

    def add_input_interface(self, source: ConductionComponent, properties: ConductionInterfaceProperties,
                            add_symmetric_interface: bool = True):
        """Add input interface"""
        if self._source_in_interfaces(source):
            raise ValueError('Source already added')
        self.input_interfaces.append((source, properties))
        if add_symmetric_interface:
            symmetric_interface_properties = properties.get_symmetric_properties()
            source.add_input_interface(
                self, symmetric_interface_properties, add_symmetric_interface=False
            )

    def get_iface_q_out_W(self, source: ConductionComponent, properties: ConductionInterfaceProperties) -> float:
        return (self.node.temperature_K - source.node.temperature_K) * properties.conductance_W_per_K

    def get_heat_fluxes_W(self, t: float) -> list[HeatFluxElement]:
        return [
            HeatFluxElement(
                dest=self.name,
                source=source.name,
                iface_properties=properties,
                q_out_W=self.get_iface_q_out_W(source, properties),
            )
            for source, properties in self.input_interfaces
        ]

    def get_neat_q_out_W(self, t: float) -> float:
        return sum(
            self.get_iface_q_out_W(source, properties)
            for source, properties in self.input_interfaces
        )


class _NullComponent(Component):

    def __init__(self):
        super().__init__(name='NULL_COMP')

    def get_heat_fluxes_W(self, t: float) -> list[HeatFluxElement]:
        return []

    def get_neat_q_out_W(self, t: float) -> float:
        return 0.0


_null_component = _NullComponent()


def NullComponent():
    return _null_component
