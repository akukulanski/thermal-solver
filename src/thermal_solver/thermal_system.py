import abc

from .node import Node


class ThermalSystem(abc.ABC):

    def __init__(self):
        self.nodes: list[Node] = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def __call__(self, t, y, *args) -> list[float]:
        """Return the list of dT/dt of the nodes, to be used in
        functions like scipy.integrate.solve_ivp.
        """
        assert len(y) == len(self.nodes)
        # First assign the temperatures to nodes, so calculations of heat power out are correct.
        for node, temperature_K in zip(self.nodes, y):
            node.set_temperature_K(temperature_K)
        # Equations (dT/dt)
        return [node.equation_dT_dt(t=t) for node in self.nodes]
