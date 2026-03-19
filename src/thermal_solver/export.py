import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from .results import SimResults

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


def plot_nodes_and_components(
    sim_results: SimResults,
    show: bool,
    base_filename: str | None,
):
    nodes = sim_results.system.nodes
    n_nodes = len(nodes)
    n_cols = 1
    n_rows = int(math.ceil(n_nodes / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8, n_rows * 4), sharey=True, sharex=True)
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3
    )
    axes = np.atleast_1d(axes)
    fig.suptitle('Heat Flux Out by Node')

    n_components = len([c for n in nodes for c in n.components])
    n_rows_comp = int(math.ceil(n_components / n_cols))
    fig_comp, axes_comp = plt.subplots(
        n_rows_comp, n_cols, figsize=(8, n_rows_comp * 4), sharey=True, sharex=True)
    fig_comp.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.3
    )
    axes_comp = np.atleast_1d(axes_comp)
    fig_comp.suptitle('Heat Flux Out by Component')
    comp_id = 0

    nodes_df = sim_results.get_nodes_dfs()

    for i, node in enumerate(nodes):
        node_q_out_df = nodes_df[i]

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

        for j, component in enumerate(node.components):
            comp_df = sim_results.get_component_df(component)

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
            ax.set_title(f'Node: {node.name} - Component: {component.name}')

        match_label_color(axes_comp, cmap=plt.cm.rainbow)

    match_label_color(axes, cmap=plt.cm.rainbow)

    ax = axes.flatten()[0]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0], xlim[1] + 0.20 * (xlim[1] - xlim[0]))
    ax.set_ylim(ylim[0], ylim[1] + 0.15 * (ylim[1] - ylim[0]))
    ax.legend(loc='upper right', fontsize=8)

    ax = axes_comp.flatten()[0]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim[0], xlim[1] + 0.20 * (xlim[1] - xlim[0]))
    ax.set_ylim(ylim[0], ylim[1] + 0.15 * (ylim[1] - ylim[0]))
    ax.legend(loc='upper right', fontsize=8)

    if show:
        plt.show()

    if base_filename:
        filename_nodes = f'{base_filename}_nodes.png'
        filename_components = f'{base_filename}_components.png'
        fig.savefig(filename_nodes)
        fig_comp.savefig(filename_components)

    return ((fig, axes), (fig_comp, axes_comp))


def generate_plots(
    sim_results,
    time_vector: np.ndarray,
    y_vectors: list[np.ndarray],
    y_names: list[str],
    show: bool,
    output_dir: str,
):

    filename = os.path.join(
        output_dir, 'fig_temperatures.png') if output_dir else None
    generate_plot_temperatures(
        time_vector=time_vector,
        y_vectors=y_vectors,
        y_names=y_names,
        show=False,
        filename=filename,
    )

    base_filename = os.path.join(output_dir, 'fig_')
    plot_nodes_and_components(
        sim_results=sim_results,
        show=False,
        base_filename=base_filename,
    )

    if show:
        plt.show()


def export_data(
    sim_results: SimResults,
    output_dir: str,
):

    filename_temp_csv = os.path.join(output_dir, 'temperatures.csv')
    sim_results.get_temperature_df().to_csv(filename_temp_csv)
    print(f'Exported file: {filename_temp_csv}')

    nodes_df = sim_results.get_nodes_dfs()
    for i, node in enumerate(sim_results.system.nodes):
        filename_node = os.path.join(output_dir, f'node_{node.name}_q_out.csv')
        nodes_df[i].to_csv(filename_node, header=True, index=True)
        print(f'Exported file: {filename_node}')

        for component in node.components:
            filename_component = os.path.join(
                output_dir, f'comp_{component.name}_q_out.csv')
            sim_results.get_component_df(component).to_csv(
                filename_component, header=True, index=True)
            print(f'Exported file: {filename_component}')
