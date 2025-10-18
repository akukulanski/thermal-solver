from __future__ import annotations

import numpy as np

from .constants import (
    P_SOLAR_W_PER_M2,
)
from .components import (
    RadiationSurface,
)
from .utils import (
    _get_func_name_,
    calculate_effective_area_factor,
)
from .properties import (
    Spectrum,
    RadiationSurfaceProperties,
)


__all__ = [
    'Sun'
]


class Sun(RadiationSurface):

    def __init__(self, sun_vector_getter: callable):
        super().__init__(properties=RadiationSurfaceProperties(
            area_m2=np.inf,
            orientation=None,
            emissivity=None,
            solar_absorptivity=None,
            emission_spectrum=Spectrum.VISIBLE,
        ))
        self.sun_vector_getter = sun_vector_getter

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
        """Input interfaces for the Sun are ignored"""
        pass

    def calculate_heat_power_in_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')

    def calculate_heat_power_out_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')

    def calculate_neat_heat_power_out_W(self, *args, **kwargs):
        raise NotImplementedError(
            f'Method {_get_func_name_()} not implemented for Sun!')
