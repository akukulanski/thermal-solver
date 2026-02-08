from __future__ import annotations

from .components import (
    Component,
    HeatFluxElement,
)
from .properties import (
    NodeProperties,
)
from .utils import NameGenerator


class Node:

    _name_prefix: str = 'node_'

    def __init__(self, properties: NodeProperties, name: str = ''):
        self.properties = properties
        self.components: list[Component] = []
        # FIXME: temeprature should be a function to allow pass the time
        # as argument. Find a way to allow temperature set for nodes that
        # change according to the ODE, and for infinite-mass nodes that
        # remain independent and where temperature depends strictly on time.
        self.temperature_K: float | None = None
        self.name = NameGenerator.register_or_create(
            name, prefix=self._name_prefix)

    def add_component(self, component: Component):
        assert component not in self.components
        self.components.append(component)
        component._assign_node(self)

    def get_heat_fluxes_W(self, t: float) -> list[HeatFluxElement]:
        return [
            heat_flux_element
            for component in self.components
            for heat_flux_element in component.get_heat_fluxes_W(t=t)
        ]

    def get_neat_q_out_W(self, t: float) -> float:
        return sum(
            component.get_neat_q_out_W(t=t)
            for component in self.components
        )

    def set_temperature_K(self, temperature_K: float):
        self.temperature_K = temperature_K

    def equation_dT_dt(self, t: float) -> float:
        """Thermal equation:

            dT/dt [K / s] = - 1 / C [W * s / K] * Q_out_neat [W]

        where
            * T: Temperature in [K]
            * t: Time in [s]
            * C: Thermal capacity in [W * s / K]
            * Q_out_neat: Neat power out in [W]
        """
        return (
            - (1 / self.properties.thermal_capacity_J_per_K)
            * self.get_neat_q_out_W(t=t)
        )
