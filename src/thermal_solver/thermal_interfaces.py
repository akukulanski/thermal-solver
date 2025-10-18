from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


STEFAN_BOLTZMANN_W_PER_M2_PER_K4 = 5.670374419e-8 # [W / (m2 * K4)]
P_SOLAR_W_PER_M2 = 1361 # [W / m2]


def versor(x: np.array) -> np.array:
    return x / np.linalg.norm(x)


class ThermalSystem:

    def __init__(self):
        pass

    def __call__(self, t, y, *args) -> list[float]:
        raise NotImplementedError()


@dataclass(kw_only=True)
class NodeProperties:
    mass_kg: float
    specific_heat_J_per_kg_per_K: float  # Fe: 444 J / (kg * K)

    @property
    def thermal_capacity_J_per_K(self) -> float:
        return self.mass_kg * self.specific_heat_J_per_kg_per_K


class Node:

    def __init__(self, properties: NodeProperties):
        self.properties = properties
        self.radiation_surfaces: list[RadiationSurface] = []
        self.temperature: float = None

    def add_radiation_surface(self, surface: RadiationSurface):
        assert surface not in self.radiation_surfaces
        self.radiation_surfaces.append(surface)
        if surface.node is None:
            surface.assign_node(self)

    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        return sum([
            component.calculate_neat_heat_power_out_W(t=t)
            for component in self.radiation_surfaces
        ])



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

    def __post__init__(self):
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
class ContactSurfaceProperties:
    area_m2: float # [m2]
    conductivity_W_per_m2_per_K: float # [W / (m2 * K)]
    node_a: object
    node_b: object


class Spectrum(Enum):
    IR = auto()
    VISIBLE = auto()


@dataclass(kw_only=True)
class RadiationInterfaceProperties:
    view_factor: float  # from the perspective
    spectrum: Spectrum = field(default=Spectrum.IR)

    # def get_symmetric(self, area, target_area_m2) -> RadiationInterfaceProperties:
    #     # view_factor * area = new_view_factor * target_area_m2
    #     return RadiationInterfaceProperties(
    #         view_factor=self.view_factor * area / target_area_m2
    #     )

def calculate_effective_area_factor(orientation_a, orientation_b) -> float:
    """Return the dot procut between the two versors of the orientation of the
    surfaces, inverted in sign (opposing for positive factor), and at least 0"""
    v1, v2 = versor(orientation_a), versor(orientation_b)
    return max(0, -np.dot(v1, v2))


class RadiationSurface:

    def __init__(self, properties: RadiationSurfaceProperties):
        self.properties = properties
        self.input_interfaces: list[tuple[RadiationSurface, RadiationInterfaceProperties]] = []
        self.node: Node = None

    def assign_node(self, node: Node):
        assert self.node is None, f'Surface already assigned to node ({self.node})'
        self.node: Node = node
        node.add_radiation_surface(self)

    def add_input_interface(self, source: RadiationSurface, properties: RadiationInterfaceProperties):
        """Add input interface"""
        # NOTE: RadiationInterfaceProperties.view_factor is the view factor from the perspective
        # from THIS radiation surface (self.properties.area_m2), and NOT source.properties.area_m2.
        # The exposed area is then self.properties.area_m2 * properties.view_factor
        self.input_interfaces.append((source, properties))

    def calculate_heat_transfered(
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
            source.calculate_heat_transfered(
                t=t,
                area_exposed_m2=self.properties.area_m2 * iface_properties.view_factor,
                orientation=self.properties.orientation,
                absorptivity=self.properties.get_absorptivity(spectrum=iface_properties.spectrum),
            )
            for source, iface_properties in self.input_interfaces
        ]
        return sum(q_in_W)

    def calculate_heat_power_out_W(self, t: float) -> float:
        return (
            STEFAN_BOLTZMANN_W_PER_M2_PER_K4
            * self.properties.emissivity
            * self.properties.area_m2
            * self.node.temperature ** 4
        )

    def calculate_neat_heat_power_out_W(self, t: float) -> float:
        return self.calculate_heat_power_out_W(t=t) - self.calculate_heat_power_in_W(t=t)



class Sun(RadiationSurface):

    def __init__(self, sun_vector_getter: callable):
        self.sun_vector_getter = sun_vector_getter
        # self.properties = RadiationSurfaceProperties(
        #     area_m2=None, orientation=None,
        #     emissivity=None, solar_absorptivity=None,
        # )

    def get_orientation(self, t: float = 0) -> np.ndarray | list:
        """Sun vector opposed in sign"""
        return -np.array(self.sun_vector_getter(t))

    # def calculate_heat_power_out(self) -> float:
    #     return P_SOLAR_W_PER_M2 # * self.properties.area_m2

    def calculate_heat_transfered(
        self,
        t: float,
        area_exposed_m2: float,
        orientation: np.ndarray,
        absorptivity: float,
        # t,  # FIXME: add t everywhere! Or make global?
    ) -> float:
        t = 0
        sun_orientation = self.get_orientation(t)
        effective_area_factor = calculate_effective_area_factor(
            sun_orientation, orientation
        )
        return (
            P_SOLAR_W_PER_M2
            * area_exposed_m2 * effective_area_factor
            * absorptivity
        )


class SimpleSystemTwoNodes(ThermalSystem):

    def __init__(
        self,
    ):
        self.node_radiator = Node(
            properties=NodeProperties(
                mass_kg=10,
                specific_heat_J_per_kg_per_K=444,
            )
        )
        self.node_solar_panels = Node(
            properties=NodeProperties(
                mass_kg=20,
                specific_heat_J_per_kg_per_K=444,
            )
        )

        # Create Sun
        sun = Sun(sun_vector_getter=lambda t: versor([-1, -1, 0]))

        radiator = RadiationSurface(
            properties=RadiationSurfaceProperties(
                area_m2=1,
                orientation=[1, 0, 0], # +x
                emissivity=0.8,
                solar_absorptivity=0.2,  # white paint
            )
        )
        solar_panel_xm = RadiationSurface(
            properties=RadiationSurfaceProperties(
                area_m2=1,
                orientation=[-1, 0, 0], # -x
                emissivity=0.9,
                solar_absorptivity=0.9,
            )
        )
        solar_panel_yp = RadiationSurface(
            properties=RadiationSurfaceProperties(
                area_m2=1,
                orientation=[0, 1, 0], # +y
                emissivity=0.9,
                solar_absorptivity=0.9,
            )
        )

        radiator.add_input_interface(
            source=solar_panel_xm,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )
        radiator.add_input_interface(
            source=solar_panel_yp,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )
        radiator.add_input_interface(
            source=sun,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
                spectrum=Spectrum.VISIBLE,
            ),
        )

        solar_panel_xm.add_input_interface(
            source=radiator,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )
        solar_panel_xm.add_input_interface(
            source=solar_panel_yp,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )
        solar_panel_xm.add_input_interface(
            source=sun,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
                spectrum=Spectrum.VISIBLE,
            ),
        )

        solar_panel_yp.add_input_interface(
            source=radiator,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )
        solar_panel_yp.add_input_interface(
            source=solar_panel_xm,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )
        solar_panel_yp.add_input_interface(
            source=sun,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
                spectrum=Spectrum.VISIBLE,
            ),
        )

        radiator.assign_node(self.node_radiator)
        solar_panel_xm.assign_node(self.node_solar_panels)
        solar_panel_yp.assign_node(self.node_solar_panels)


    def __call__(self, t, y, *args) -> list[float]:
        # def eq_system(t, y):
        #     T1, T2 = y
        #     area_in_1_eff = get_area_in_1_eff(t)
        #     area_in_2_eff = get_area_in_2_eff(t)
        #     return [
        #         # - m1 * c1 * dT1/dt = SF * emm_1 * area_rad_1 * T1**4 - abs_1 * area_in_1_eff * I_SUN - cond_12 * area_cond_12 * (T1 - T2)
        #         - (1 / m1 / c1) * (SF * emm_1 * area_rad_1 * T1**4 - abs_1 * area_in_1_eff * I_SUN - cond_12 * area_cond_12 * (T1 - T2)),
        #         # - m2 * c2 * dT2/dt = SF * emm_2 * area_rad_2 * T2**4 - abs_2 * area_in_2_eff * I_SUN - cond_21 * area_cond_21 * (T2 - T1)
        #         - (1 / m2 / c2) * (SF * emm_2 * area_rad_2 * T2**4 - abs_2 * area_in_2_eff * I_SUN - cond_21 * area_cond_21 * (T2 - T1)),
        #     ]
        T1, T2 = y
        # First assign temperatures to nodes, so calculations of heat power out are correct.
        self.node_radiator.temperature = T1
        self.node_solar_panels.temperature = T2

        # eq_dT1_dt = (
        #     - (1 / self.node_radiator.properties.specific_heat_J_per_kg_per_K)
        #     * self.node_radiator.calculate_neat_heat_power_out_W(t=t)
        # )
        # eq_dT2_dt = (
        #     - (1 / self.node_solar_panels.properties.specific_heat_J_per_kg_per_K)
        #     * self.node_solar_panels.calculate_neat_heat_power_out_W(t=t)
        # )
        # return [eq_dT1_dt, eq_dT2_dt]

        equations = [
            - (1 / node.properties.specific_heat_J_per_kg_per_K) * node.calculate_neat_heat_power_out_W(t=t)
            for node in (self.node_radiator, self.node_solar_panels)
        ]
        return equations


def run():
    import math
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp
    matplotlib.use('qtagg')

    system = SimpleSystemTwoNodes()

    # Initial conditions and time span
    y0 = [300, 300]
    t_span = (0, 3600 * 4)

    # Parameters for the Lotka-Volterra system
    args = ()

    # Solve the IVP
    sol = solve_ivp(fun=system, t_span=t_span, y0=y0, args=args, dense_output=True)

    # Access the solution
    print("Times:", sol.t)
    print("Solution at computed times:\n", sol.y)

    # Evaluate the solution at specific times (using dense_output)
    t_eval = np.linspace(*t_span, 100)
    y_interp = sol.sol(t_eval)

    fig = plt.figure()
    plt.plot(t_eval, y_interp[0], label='T1')
    plt.plot(t_eval, y_interp[1], label='T2')
    plt.xlabel('t')
    # plt.legend(['T1', 'T2'], shadow=True)
    plt.legend(shadow=True)
    plt.title('Thermal System')
    plt.show()


def main():
    run()


if __name__ == '__main__':
    main()
