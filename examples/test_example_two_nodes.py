import argparse
import os

from thermal_solver.node import Node
from thermal_solver.vectors import versor
from thermal_solver.properties import (
    NodeProperties,
    RadiationSurfaceProperties,
    RadiationInterfaceProperties,
)
from thermal_solver.components import (
    RadiationSurface,
)
from thermal_solver.lib import (
    Sun,
)
from thermal_solver.thermal_system import ThermalSystem


class SimpleSystemTwoNodes(ThermalSystem):

    def __init__(
        self,
    ):
        super().__init__()

        node_radiator = Node(
            properties=NodeProperties(
                mass_kg=10,
                specific_heat_J_per_kg_per_K=444,
            )
        )
        node_solar_panels = Node(
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
                orientation=[1, 0, 0],  # +x
                emissivity=0.8,
                solar_absorptivity=0.2,  # white paint
            ),
            name='radiator'
        )
        solar_panel_xm = RadiationSurface(
            properties=RadiationSurfaceProperties(
                area_m2=1,
                orientation=[-1, 0, 0],  # -x
                emissivity=0.9,
                solar_absorptivity=0.9,
            ),
            name='panel_xm'
        )
        solar_panel_yp = RadiationSurface(
            properties=RadiationSurfaceProperties(
                area_m2=1,
                orientation=[0, 1, 0],  # +y
                emissivity=0.9,
                solar_absorptivity=0.9,
            ),
            name='panel_yp'
        )

        radiator.add_input_interface(
            source=solar_panel_xm,
            properties=RadiationInterfaceProperties(
                view_factor=0.9,
            ),
        )
        radiator.add_input_interface(
            source=solar_panel_yp,
            properties=RadiationInterfaceProperties(
                view_factor=0.9,
            ),
        )
        radiator.add_input_interface(
            source=sun,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )

        # solar_panel_xm.add_input_interface(
        #     source=radiator,
        #     properties=RadiationInterfaceProperties(
        #         view_factor=1.0,
        #     ),
        # )  # symmetic interface added automatically
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
            ),
        )

        # solar_panel_yp.add_input_interface(
        #     source=radiator,
        #     properties=RadiationInterfaceProperties(
        #         view_factor=1.0,
        #     ),
        # )  # symmetic interface added automatically
        # solar_panel_yp.add_input_interface(
        #     source=solar_panel_xm,
        #     properties=RadiationInterfaceProperties(
        #         view_factor=1.0,
        #     ),
        # )  # symmetic interface added automatically
        solar_panel_yp.add_input_interface(
            source=sun,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )

        node_radiator.add_component(radiator)
        node_solar_panels.add_component(solar_panel_xm)
        node_solar_panels.add_component(solar_panel_yp)

        self.add_node(node=node_radiator)
        self.add_node(node=node_solar_panels)


def run(
    show_fig: bool = False,
    fig_filename: str = None,
):
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
    sol = solve_ivp(fun=system, t_span=t_span, y0=y0,
                    args=args, dense_output=True)

    # Access the solution
    print("Times:", sol.t)
    print("Solution at computed times:\n", sol.y)

    # Evaluate the solution at specific times (using dense_output)
    t_eval = np.linspace(*t_span, 100)
    y_interp = sol.sol(t_eval)

    from thermal_solver.plot import generate_plots
    generate_plots(
        system=system,
        time_vector=t_eval,
        y_vectors=y_interp,
        y_names=['T1', 'T2'],
        show=show_fig,
        base_filename=fig_filename,
    )


def main(sys_args=None):
    default_filename = os.path.join(os.path.dirname(__file__), 'output/test_example_two_nodes.png')
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default=default_filename,
                        help='Output filename')
    parser.add_argument('--show', action='store_true', help='Show image')
    args = parser.parse_args(sys_args)
    args.output = os.path.abspath(args.output)
    run(
        show_fig=args.show,
        fig_filename=args.output,
    )


def test_example():
    main([])


if __name__ == '__main__':
    main()
