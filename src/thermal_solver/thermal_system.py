import abc

from .node import Node


class ThermalSystem(abc.ABC):

    def __init__(self, log_heat_fluxes: bool = False):
        self.nodes: list[Node] = []
        self.log_heat_fluxes = log_heat_fluxes
        self._log = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def log(self, t: float, nodes: list[Node]):
        self._log.append((
            t, *[node.xxx() for node in nodes]
        ))

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
