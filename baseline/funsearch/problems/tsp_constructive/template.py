import numpy as np


def select_next_node(
    current_node: int,
    destination_node: int,
    unvisited_nodes: set,
    distance_matrix: np.ndarray,
) -> int:
    """
    Select the next node to visit in a constructive TSP heuristic.

    Inputs:
    - current_node: index of the current node
    - destination_node: index of the final destination node (usually the start node)
    - unvisited_nodes: set of unvisited node indices
    - distance_matrix: pairwise distance matrix of shape [n, n]

    Output:
    - index of the next node to visit

    Notes:
    - The returned node must belong to unvisited_nodes.
    - The function should be deterministic for the same inputs.
    - The objective is to help construct a short Hamiltonian tour.
    """
    raise NotImplementedError("This function body should be evolved by FunSearch.")