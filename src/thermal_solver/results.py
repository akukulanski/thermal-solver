import functools
import numpy as np
import pandas as pd

from .components import Component
from .node import Node


def build_temperatures_df(
    system,
    time_vector: np.ndarray,
    y_vectors: list[np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(
        {node.name: y_vectors[i] for i, node in enumerate(system.nodes)},
        index=time_vector
    )


def setup_nodes_temperature(nodes, row):
    for n in nodes:
        n.set_temperature_K(getattr(row, n.name))


def extract_node_heat_fluxes(nodes: list[None], node: Node, row) -> pd.Series:
    # Set temperatures before calculating heat flux
    setup_nodes_temperature(nodes, row)
    # Get the heat fluxes
    return pd.Series({
        component.name + '_' + heat_flux_element.source: heat_flux_element.q_out_W
        for component in node.components
        for heat_flux_element in component.get_heat_fluxes_W(t=row.index)
    })


def extract_comp_heat_fluxes(nodes: list[Node], component: Component, row) -> pd.Series:
    # Set temperatures before calculating heat flux
    setup_nodes_temperature(nodes, row)
    # Get the heat fluxes
    return pd.Series({
        component.name + '_' + heat_flux_element.source: heat_flux_element.q_out_W
        for heat_flux_element in component.get_heat_fluxes_W(t=row.index)
    })


def build_nodes_dfs(
    system,
    temperatures_df: pd.DataFrame,
) -> tuple[pd.DataFrame]:

    dfs = []
    for i, node in enumerate(system.nodes):
        df = temperatures_df.apply(functools.partial(
            extract_node_heat_fluxes, system.nodes, node), axis=1)
        dfs.append(df)

    return tuple(dfs)


class SimResults:

    def __init__(self, system, time_vector, y_vectors):
        self.system = system
        self.time_vector = time_vector
        self.y_vectors = y_vectors

    @functools.cache
    def get_temperature_df(self) -> pd.DataFrame:
        return build_temperatures_df(
            system=self.system,
            time_vector=self.time_vector,
            y_vectors=self.y_vectors,
        )

    @functools.cache
    def get_nodes_dfs(self) -> tuple[pd.DataFrame, ...]:
        return tuple(
            self.get_temperature_df().apply(
                functools.partial(extract_node_heat_fluxes,
                                  self.system.nodes, node),
                axis=1
            )
            for node in self.system.nodes
        )

    @functools.cache
    def get_component_df(self, component: Component) -> pd.DataFrame:
        return self.get_temperature_df().apply(
            functools.partial(extract_comp_heat_fluxes,
                              self.system.nodes, component),
            axis=1
        )

    def df_consistency_check(self):
        # Consistency check
        for i, node in enumerate(self.system.nodes):
            node_q_out_W = self.get_nodes_dfs()[i].sum(axis=1)
            components_q_out_W = sum(
                self.get_component_df(component).sum(axis=1)
                for component in node.components
            )
            assert np.allclose(node_q_out_W.to_numpy(), components_q_out_W.to_numpy()), (
                f'{node_q_out_W} != {components_q_out_W}'
            )
