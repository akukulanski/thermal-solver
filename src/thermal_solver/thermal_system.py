import abc


class ThermalSystem(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, t, y, *args) -> list[float]:
        raise NotImplementedError()
