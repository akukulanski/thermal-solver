import abc

from .node import Node


class ThermalSystem(abc.ABC):

    def __init__(self):
        self.nodes = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    @abc.abstractmethod
    def __call__(self, t, y, *args) -> list[float]:
        raise NotImplementedError()
