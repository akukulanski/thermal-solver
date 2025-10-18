from __future__ import annotations

from .components import (
    Component,
)
from .properties import (
    NodeProperties,
)


class Node:

    def __init__(self, properties: NodeProperties):
        self.properties = properties
        self.components: list[Component] = []
        # FIXME: temeprature should be a function to allow pass the time
        # as argument. Find a way to allow temperature set for nodes that
        # change according to the ODE, and for infinite-mass nodes that
        # remain independent and where temperature depends strictly on time.
        self.temperature: float = None

    def add_component(self, component: Component):
        assert component not in self.components
        self.components.append(component)
        component._assign_node(self)

    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        return sum([
            component.calculate_neat_heat_power_out_W(t=t)
            for component in self.components
        ])
