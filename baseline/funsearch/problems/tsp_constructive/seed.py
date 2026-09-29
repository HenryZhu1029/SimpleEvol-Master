import numpy as np


def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set,distance_matrix: np.ndarray) -> int:
    """Select the next node to visit from the unvisited nodes."""
    if not unvisited_nodes:
        return destination_node

    scores = {}
    for node in unvisited_nodes:
        scores[node] = 1

    next_node = min(scores, key=scores.get)
    return int(next_node)