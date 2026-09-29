
## problem-specific prompts for tsp constructive heuristic

TASK_DESC = """
Solving Traveling Salesman Problem (TSP) with constructive heuristics. 
TSP requires finding the shortest path that visits all given nodes and returns to the starting node.
The code must define a function named `select_next_node` with the following function signature:

```python
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int

```

Inputs:
- current_node: index of the current node
- destination_node: index of the destination node (usually the starting node)
- unvisited_nodes: set of unvisited nodes
- distance_matrix: distance matrix between nodes (numpy array with shape [n, n])

The function MUST return the index (int) of the next node to visit.

Avoid reproducing the standard greedy nearest-neighbour heuristic or trivial variants of it.
Prefer exploring structurally different constructive strategies.
"""