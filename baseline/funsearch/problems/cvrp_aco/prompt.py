
TASK_DESC = """
Solving Capacitated Vehicle Routing Problem (CVRP) via stochastic solution sampling. 
CVRP requires finding the shortest path that visits all given nodes and returns to the starting node.
Each node has a demand and each vehicle has a capacity. The total demand of the nodes visited by a vehicle cannot exceed the vehicle capacity. 
When the total demand exceeds the vehicle capacity, the vehicle must return to the starting node.

The code must define a function named `heuristics` with the following function signature:
```python
def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:

    return heuristic_matrix
```



where
Inputs:
- distance_matrix: a NumPy 2D array of shape (n, n). Symmetric, with zeros on the diagonal.
- coordinates: a NumPy 2D array of shape (n, 2). Euclidean coordinates of nodes.
- demands: a NumPy 1D array of shape (n). demands[0] = 0 for the depot.
- capacity: an integer vehicle capacity.

Outputs:
- heuristic_matrix: a NumPy 2D array of shape (n, n).  It is a prior indicator of how promising it is to include each edge in a solution
and a larger matrix element (i, j) means is more promising to include the edge (i, j).

The return is of the same shape as the distance_matrix. The depot node is indexed by 0.
"""


