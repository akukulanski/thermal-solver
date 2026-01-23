import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
matplotlib.use('qtagg')

from thermal_solver.node import Node


def match_label_color(axes):

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


def plot_nodes_and_components(
    nodes: list[Node],
    time_vector: np.ndarray,
    show: bool,
    base_filename: str | None,
):
    n_nodes = len(nodes)
    n_cols = 2
    n_rows = int(math.ceil(n_nodes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3), sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle('Heat Flux Out by Node')

    n_components = len([c for n in nodes for c in n.components])
    n_rows_comp = int(math.ceil(n_components / n_cols))
    fig_comp, axes_comp = plt.subplots(n_rows_comp, n_cols, figsize=(12, n_rows_comp * 3), sharey=True)
    axes_comp = np.atleast_1d(axes_comp)
    fig_comp.suptitle('Heat Flux Out by Component')
    comp_id = 0

    for i, node in enumerate(nodes):
        node_q_out = pd.DataFrame([
            pd.DataFrame(node.get_heat_fluxes_W(t)).groupby('dest')['q_out_W'].sum().to_dict()
            for t in time_vector
        ], index=time_vector)

        ax = axes.flatten()[i]
        node_q_out.plot(ax=ax)
        node_q_out.sum(axis=1).plot(ax=ax, style='--', label='Total')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Q_out [W]')
        ax.set_title(f'Node: {node.name}')

        for j, comp_name in enumerate(pd.DataFrame(node.get_heat_fluxes_W(t=0))['dest'].unique()):
            df_vs_t = [pd.DataFrame(node.get_heat_fluxes_W(t)) for t in time_vector]
            comp_df = pd.DataFrame([
                df_vs_t[k][df_vs_t[i]['dest'] == comp_name].groupby('source')['q_out_W'].sum().to_dict()
                for k in range(len(df_vs_t))
            ], index=time_vector)

            ax = axes_comp.flatten()[comp_id]
            comp_id += 1
            comp_df.plot(ax=ax)
            comp_df.sum(axis=1).plot(ax=ax, style='--', label='Total')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Q_out [W]')
            ax.set_title(f'Node: {node.name} - Component: {comp_name}')

        match_label_color(axes_comp)

    match_label_color(axes)

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
    base_filename: str,
):

    generate_plot_temperatures(
        time_vector=time_vector,
        y_vectors=y_vectors,
        y_names=y_names,
        show=False,
        filename=f'{base_filename}.png' if base_filename else None,
    )

    plot_nodes_and_components(
        nodes=system.nodes,
        time_vector=time_vector,
        show=False,
        base_filename=base_filename,
    )

    if show:
        plt.show()

