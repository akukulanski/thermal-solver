from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


__all__ = [
    'NodeProperties',
    'RadiationSurfaceProperties',
    'HeatSourceProperties',
    'RadiationInterfaceProperties',
    'ContactSurfaceProperties',
]


class Spectrum(Enum):
    IR = auto()
    VISIBLE = auto()


@dataclass(kw_only=True)
class NodeProperties:
    mass_kg: float
    specific_heat_J_per_kg_per_K: float  # Fe: 444 J / (kg * K)

    @property
    def thermal_capacity_J_per_K(self) -> float:
        return self.mass_kg * self.specific_heat_J_per_kg_per_K


@dataclass(kw_only=True)
class RadiationSurfaceProperties:
    area_m2: float
    orientation: list[float, float, float]
    # NOTE: the total emissivity is considered, but with a
    # different parameter for solar absorptivity to be able
    # to model emissivity at two different wavelengths:
    # - Sun radiation at T~5800 K (visible light)
    # - Nodes radiation in the model at T<400 K (IR)
    emissivity: float
    solar_absorptivity: float = field(default=None)
    emission_spectrum: Spectrum = field(default=Spectrum.IR)

    def __post_init__(self):
        if self.solar_absorptivity is None:
            self.solar_absorptivity = self.emissivity

    def get_absorptivity(self, spectrum: Spectrum) -> float:
        if spectrum == Spectrum.IR:
            return self.emissivity
        elif spectrum == Spectrum.VISIBLE:
            return self.solar_absorptivity
        else:
            raise ValueError(f'Unknown absorptivity for spectrum: {spectrum}')


@dataclass(kw_only=True)
class HeatSourceProperties:
    constant_power_W: float = field(default=None)
    power_getter: callable = field(default=None)

    def __post_init__(self):
        if self.constant_power_W is None and self.power_getter is None:
            raise TypeError(
                f'Either constant_power_W or power_getter should be defined')
        elif self.constant_power_W is not None and self.power_getter is not None:
            raise TypeError(
                f'Only one of constant_power_W or power_getter should be defined')
        elif self.constant_power_W is not None:
            self.power_getter = lambda t: self.constant_power_W

    def get_power_W(self, t: float) -> float:
        return self.power_getter(t=t)


@dataclass(kw_only=True)
class RadiationInterfaceProperties:
    view_factor: float = field(default=1.0)  # from the perspective

    def get_symmetric_properties(self, area_m2, target_area_m2) -> RadiationInterfaceProperties:
        # view_factor * area_m2 = new_view_factor * target_area_m2
        return RadiationInterfaceProperties(
            view_factor=self.view_factor * area_m2 / target_area_m2
        )


@dataclass(kw_only=True)
class ContactSurfaceProperties:
    area_m2: float  # [m2]
    conductivity_W_per_m2_per_K: float  # [W / (m2 * K)]
