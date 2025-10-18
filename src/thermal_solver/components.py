from __future__ import annotations

import abc
import numpy as np
from typing import TYPE_CHECKING

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
    _get_func_name_,
    calculate_effective_area_factor,
)

if TYPE_CHECKING:
    from .node import Node


__all__ = [
    'Component',
    'RadiationSurface',
    'HeatSource',
    'ConductionComponent',
]


class Component(abc.ABC):

    def __init__(self):
        self.node: Node = None

    def _assign_node(self, node: Node):
        assert self.node is None, f'Component already assigned to node ({self.node})'
        assert self in node.components, (
            f'Incorrect use of internal method {self.__class__.__name__}.{_get_func_name_()}(): '
            f'Trying to assign node to component, but component not in Node.\n'
            f'Use Node.add_component() instead.'
        )
        self.node: Node = node

    @abc.abstractmethod
    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        raise NotImplementedError()


class RadiationSurface(Component):

    def __init__(self, properties: RadiationSurfaceProperties):
        super().__init__()
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
            raise ValueError(f'Source already added')
        self.input_interfaces.append((source, properties))
        if add_symmetric_interface:
            symmetric_interface_properties = properties.get_symmetric_properties(
                area_m2=self.properties.area_m2,
                target_area_m2=source.properties.area_m2,
            )
            source.add_input_interface(
                self, symmetric_interface_properties, add_symmetric_interface=False)

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


class HeatSource(Component):

    def __init__(self, properties: HeatSourceProperties):
        super().__init__()
        self.properties = properties

    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        return -self.properties.power_getter(t=t)


class ConductionComponent(Component):

    def __init__(self, properties: ConductionProperties):
        super().__init__()
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
            raise ValueError(f'Source already added')
        self.input_interfaces.append((source, properties))
        if add_symmetric_interface:
            symmetric_interface_properties = properties.get_symmetric_properties()
            source.add_input_interface(
                self, symmetric_interface_properties, add_symmetric_interface=False
            )

    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        this_node_temp_K = self.node.temperature
        return sum([
            (this_node_temp_K - source.node.temperature) *
            iface.conductance_W_per_K
            for source, iface in self.input_interfaces
        ])
