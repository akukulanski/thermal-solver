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

        # === Nodes properties ===
        node_radiator_properties = NodeProperties(
            mass_kg=10,
            specific_heat_J_per_kg_per_K=444,
        )

        node_solar_panels_properties = NodeProperties(
            mass_kg=20,
            specific_heat_J_per_kg_per_K=444,
        )

        # === Components properties ===
        radiator_prop = RadiationSurfaceProperties(
            area_m2=1,
            orientation=[1, 0, 0],  # +x
            emissivity=0.8,
            solar_absorptivity=0.2,  # white paint
        )
        solar_panel_xm_prop = RadiationSurfaceProperties(
            area_m2=1,
            orientation=[-1, 0, 0],  # -x
            emissivity=0.9,
            solar_absorptivity=0.9,
        )
        solar_panel_yp_prop = RadiationSurfaceProperties(
            area_m2=1,
            orientation=[0, 1, 0],  # +y
            emissivity=0.9,
            solar_absorptivity=0.9,
        )

        # === Create Nodes ===
        node_radiator = Node(properties=node_radiator_properties)
        node_solar_panels = Node(properties=node_solar_panels_properties)

        # === Create Components ===
        radiator = RadiationSurface(properties=radiator_prop, name='radiator')
        solar_panel_xm = RadiationSurface(properties=solar_panel_xm_prop, name='panel_xm')
        solar_panel_yp = RadiationSurface(properties=solar_panel_yp_prop, name='panel_yp')

        # === Create Sun ===
        sun = Sun(sun_vector_getter=lambda t: versor([-1, -1, 0]))

        # === Assign Components to Nodes ===
        node_radiator.add_component(radiator)
        node_solar_panels.add_component(solar_panel_xm)
        node_solar_panels.add_component(solar_panel_yp)

        # === Assign Nodes to Thermal System ===
        self.add_node(node=node_radiator)
        self.add_node(node=node_solar_panels)

        # === Connect Interfaces ===

        # Connect radiator to {solar_panel_xm, solar_panel_yp, sun}
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

        # Connect solar_panel_xm to {solar_panel_yp, sun} (radiator connected above, symmetry is infered)
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

        # Connect solar_panel_yp to {sun} (radiator and solar_panel_xm connected above, symmetry is infered)
        solar_panel_yp.add_input_interface(
            source=sun,
            properties=RadiationInterfaceProperties(
                view_factor=1.0,
            ),
        )


def run(
    show_fig: bool = False,
    output_dir: str = 'output',
):
    import matplotlib
    import numpy as np
    from scipy.integrate import solve_ivp
    matplotlib.use('qtagg')

    from thermal_solver.results import SimResults
    from thermal_solver.export import generate_plots, export_data

    system = SimpleSystemTwoNodes()

    # Initial conditions and time span
    y0 = [300, 300]
    t_span = (0, 3600 * 4)

    # Parameters for the Lotka-Volterra system
    args = ()

    # Solve the IVP
    sol = solve_ivp(
        fun=system,
        t_span=t_span,
        y0=y0,
        method='RK45',
        args=args,
        dense_output=True
    )

    # Access the solution
    print("Times:", sol.t)
    print("Solution at computed times:\n", sol.y)

    # Evaluate the solution at specific times (using dense_output)
    t_eval = np.linspace(*t_span, 100)
    y_interp = sol.sol(t_eval)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sim_results = SimResults(
        system=system,
        time_vector=t_eval,
        y_vectors=y_interp,
    )
    print('--- Temperature DF ---')
    print(sim_results.get_temperature_df())
    print('')
    print(f'--- Nodes DFs ---')
    print(sim_results.get_nodes_dfs())
    print('')
    for node in system.nodes:
        for component in node.components:
            print(f'--- Node {node.name} - Compoennt {component.name} DF ---')
            print(sim_results.get_component_df(component))

    sim_results.df_consistency_check()

    generate_plots(
        sim_results=sim_results,
        time_vector=t_eval,
        y_vectors=y_interp,
        y_names=[f'T_{n.name}' for n in system.nodes],
        show=show_fig,
        output_dir=output_dir,
    )
    print(f'output_dir={output_dir}')
    if output_dir:
        export_data(
            sim_results=sim_results,
            output_dir=output_dir,
        )


def main(sys_args=None):
    default_output_dir = os.path.join(os.path.dirname(__file__), 'output/test_example_two_nodes')
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', type=str, default=default_output_dir,
                        help='Output directory')
    parser.add_argument('--show', action='store_true', help='Show image')
    args = parser.parse_args(sys_args)
    args.output_dir = os.path.abspath(args.output_dir)
    run(
        show_fig=args.show,
        output_dir=args.output_dir,
    )


def test_example():
    main([])


if __name__ == '__main__':
    main()
