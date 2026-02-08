from thermal_solver.components import Component
from thermal_solver.node import Node
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
matplotlib.use('qtagg')


def match_label_color_(axes):

    # After plotting, iterate through lines to match colors
    lines = [line for ax in axes.flatten() for line in ax.get_lines()]
    # Keep track of labels that already have a defined color
    defined_labels = {}

    for line in lines:
        label = line.get_label()
        if label in defined_labels:
            # If the label exists, set the current line's color to the existing color
            line.set_color(defined_labels[label])
            # Hide the duplicate label in the legend by prepending an underscore
            # line.set_label(f"_{label}")
        else:
            # If new label, store its color
            defined_labels[label] = line.get_color()


def match_label_color(axes, cmap=None):

    # After plotting, iterate through lines to match colors
    lines = [line for ax in axes.flatten() for line in ax.get_lines()]

    n_labels = len(set([line.get_label() for line in lines]))
    color_iter = iter(cmap(np.linspace(0, 1, n_labels))) if cmap else None

    # Keep track of labels that already have a defined color
    defined_labels = {}

    for line in lines:
        label = line.get_label()
        # If new label, store its color
        if label not in defined_labels:
            defined_labels[label] = next(
                color_iter) if color_iter else line.get_color()

        line.set_color(defined_labels[label])

        # Hide the duplicate label in the legend by prepending an underscore
        # line.set_label(f"_{label}")


def generate_plot_temperatures(
    time_vector: np.ndarray,
    y_vectors: list[np.ndarray],
    y_names: list[str],
    show: bool,
    filename: str | None,
):
    assert len(y_vectors) == len(y_names)
    fig = plt.figure()
    plt.plot(time_vector, y_vectors[0], label='T1')
    plt.plot(time_vector, y_vectors[1], label='T2')
    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [K]')
    plt.grid(True)
    # plt.legend(['T1', 'T2'], shadow=True)
    plt.legend(shadow=True)
    plt.title('Thermal System')
    if show:
        plt.show()
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
    return fig


def get_node_q_out_df(node: Node, time_vector) -> pd.DataFrame:
    raise NotImplementedError(f"WARNING: wrong implementation. Temperature not updated!")
    return pd.DataFrame([
        pd.DataFrame(node.get_heat_fluxes_W(t)).groupby(
            'dest')['q_out_W'].sum().to_dict()
        for t in time_vector
    ], index=time_vector)


def get_node_components_q_out_df(node: Node, time_vector) -> dict:
    raise NotImplementedError(f"WARNING: wrong implementation. Temperature not updated!")
    dfs = {}
    for j, comp_name in enumerate(pd.DataFrame(node.get_heat_fluxes_W(t=0))['dest'].unique()):
        df_vs_t = [pd.DataFrame(node.get_heat_fluxes_W(t))
                    for t in time_vector]
        comp_df = pd.DataFrame([
            df_vs_t[k][df_vs_t[i]['dest'] == comp_name].groupby(
                'source')['q_out_W'].sum().to_dict()
            for k in range(len(df_vs_t))
        ], index=time_vector)
        dfs[comp_name] = comp_df
    return dfs



def get_component_q_out_df(component: Component, time_vector) -> pd.DataFrame:
    raise NotImplementedError(f"WARNING: wrong implementation. Temperature not updated!")
    df_vs_t = [pd.DataFrame(component.get_heat_fluxes_W(t)) for t in time_vector]
    comp_df = pd.DataFrame([
        df_vs_t[k].groupby('source')['q_out_W'].sum().to_dict()
        for k in range(len(df_vs_t))
    ], index=time_vector)
    return comp_df


def plot_nodes_and_components(
    nodes: list[Node],
    time_vector: np.ndarray,
    show: bool,
    base_filename: str | None,
):
    print(f"WARNING: wrong implementation. Temperature not updated!")
    return
    n_nodes = len(nodes)
    n_cols = 2
    n_rows = int(math.ceil(n_nodes / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12, n_rows * 3), sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle('Heat Flux Out by Node')

    n_components = len([c for n in nodes for c in n.components])
    n_rows_comp = int(math.ceil(n_components / n_cols))
    fig_comp, axes_comp = plt.subplots(
        n_rows_comp, n_cols, figsize=(12, n_rows_comp * 3), sharey=True)
    axes_comp = np.atleast_1d(axes_comp)
    fig_comp.suptitle('Heat Flux Out by Component')
    comp_id = 0

    for i, node in enumerate(nodes):
        node_q_out_df = get_node_q_out_df(node, time_vector)

        ax = axes.flatten()[i]
        node_q_out_df.plot(ax=ax)
        node_q_out_df.sum(axis=1).plot(
            ax=ax, style='--', label='Total', dashes=[3, 3], lw=3, alpha=0.5
        )
        ax.grid(True)
        # ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Q_out [W]')
        ax.set_title(f'Node: {node.name}')

        for j, comp_name in enumerate(pd.DataFrame(node.get_heat_fluxes_W(t=0))['dest'].unique()):
            df_vs_t = [pd.DataFrame(node.get_heat_fluxes_W(t))
                       for t in time_vector]
            comp_df = pd.DataFrame([
                df_vs_t[k][df_vs_t[i]['dest'] == comp_name].groupby(
                    'source')['q_out_W'].sum().to_dict()
                for k in range(len(df_vs_t))
            ], index=time_vector)

            ax = axes_comp.flatten()[comp_id]
            comp_id += 1
            comp_df.plot(ax=ax)
            comp_df.sum(axis=1).plot(
                ax=ax, style='--', label='Total', dashes=[3, 3], lw=3, alpha=0.5
            )
            ax.grid(True)
            # ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Q_out [W]')
            ax.set_title(f'Node: {node.name} - Component: {comp_name}')

        match_label_color(axes_comp, cmap=plt.cm.rainbow)

    match_label_color(axes, cmap=plt.cm.rainbow)

    for ax in axes.flatten():
        ax.legend()

    for ax in axes_comp.flatten():
        ax.legend()

    if show:
        plt.show()

    if base_filename:
        filename_nodes = f'{base_filename}_nodes.png'
        filename_components = f'{base_filename}_components.png'
        fig.savefig(filename_nodes)
        fig_comp.savefig(filename_components)

    return ((fig, axes), (fig_comp, axes_comp))


def generate_plots(
    system,
    time_vector: np.ndarray,
    y_vectors: list[np.ndarray],
    y_names: list[str],
    show: bool,
    output_dir: str,
):

    filename = os.path.join(output_dir, f'fig_temperatures.png') if output_dir else None
    generate_plot_temperatures(
        time_vector=time_vector,
        y_vectors=y_vectors,
        y_names=y_names,
        show=False,
        filename=filename,
    )

    base_filename = os.path.join(output_dir, 'fig_')
    plot_nodes_and_components(
        nodes=system.nodes,
        time_vector=time_vector,
        show=False,
        base_filename=base_filename,
    )

    if show:
        plt.show()



def export_data(
    system,
    time_vector: np.ndarray,
    y_vectors: list[np.ndarray],
    output_dir: str,
):
    import pandas as pd
    filename_temp_csv = os.path.join(output_dir, 'temperatures.csv')
    pd.DataFrame({system.nodes[i].name: y_vectors[i] for i in range(len(system.nodes))}, index=time_vector).to_csv(filename_temp_csv)

    for node in system.nodes:
        filename = os.path.join(output_dir, f'node_{node.name}_q_out.csv')
        df = get_node_q_out_df(node, time_vector)
        print(f'df {node.name}=\n-----\n{df}\n-----\n')
        df.to_csv(filename, header=True, index=True)
        print(f'Exported file: {filename}')

        for component in node.components:
            filename = os.path.join(output_dir, f'comp_{component.name}_q_out.csv')
            df = get_component_q_out_df(component, time_vector)
            df.to_csv(filename, header=True, index=True)
            print(f'Exported file: {filename}')
