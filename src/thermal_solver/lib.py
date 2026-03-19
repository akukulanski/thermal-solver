from __future__ import annotations

import numpy as np

from .constants import (
    P_SOLAR_W_PER_M2,
)
from .components import (
    RadiationSurface,
)
from .node import Node
from .utils import (
    get_func_name,
)
from .properties import (
    Spectrum,
    RadiationSurfaceProperties,
    NodeProperties,
)


__all__ = [
    'Sun'
]


class Sun(RadiationSurface):

    _name: str = 'Sun'

    def __init__(self, sun_vector_getter: callable):
        super().__init__(
            name=self._name,
            properties=RadiationSurfaceProperties(
                area_m2=np.inf,
                orientation=None,
                emissivity=None,
                solar_absorptivity=None,
                emission_spectrum=Spectrum.VISIBLE,
            )
        )
        self.sun_vector_getter = sun_vector_getter

    def get_orientation(self, t: float = 0) -> np.ndarray | list:
        """Sun vector opposed in sign"""
        return -np.array(self.sun_vector_getter(t=t))

    def calculate_heat_transfered_W(
        self,
        t: float,
        area_exposed_m2: float,
        orientation: np.ndarray | list,
        absorptivity: float,
    ) -> float:
        sun_orientation = self.get_orientation(t=t)
        effective_area_factor = RadiationSurface.calculate_effective_area_factor(
            sun_orientation, orientation
        )
        return (
            P_SOLAR_W_PER_M2
            * area_exposed_m2 * effective_area_factor
            * absorptivity
        )

    def _assign_node(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {get_func_name()} not implemented for Sun!')

    def add_input_interface(self, *args, **kwargs):
        """Input interfaces for the Sun are ignored"""
        pass

    def calculate_heat_power_out_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {get_func_name()} not implemented for Sun!')

    def get_neat_q_out_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {get_func_name()} not implemented for Sun!')

    def get_heat_fluxes_W(self, t) -> list[dict]:
        raise NotImplementedError(
            f'Method {get_func_name()} not implemented for Sun!')


# class InfinteMassNode(Node):

#     def __init__(self, temperature_getter: callable):
#         super().__init__(properties=NodeProperties(
#             mass_kg=np.inf,
#             specific_heat_J_per_kg_per_K=np.inf,
#         ))
#         self.temperature_getter = temperature_getter

#     def get_temperature_K(self, t: float = 0) -> float:
#         """Get temperature"""
#         return self.temperature_getter(t=t)

#     def temperature(self, t: float = 0):
#         raise NotImplementedError()


class FixedTemperatureNode(Node):

    def __init__(self, temperature_K: float):
        super().__init__(properties=NodeProperties(
            mass_kg=np.inf,
            specific_heat_J_per_kg_per_K=np.inf,
        ))
        self._temperature_K = temperature_K

    @property
    def temperature_K(self):
        """Replace the attribute with a property that returns the fixed value
        regardless of any use of obj.temperature_K = ..."""
        return self._temperature_K

    @temperature_K.setter
    def temperature_K(self, value: float):
        """The setter exists to avoid errors in Node.__init__(), but it's ignored"""
        pass
