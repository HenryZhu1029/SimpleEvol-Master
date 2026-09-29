TASK_DESC = """\
You are solving the Flow Shop Scheduling Problem (FSSP) using Guided Local Search (GLS).
There are n jobs and m machines.
Each job must be processed on all machines in the same order.
The goal is to find a job sequence that minimizes the makespan.

Heuristic description:
The heuristic is used inside a GLS framework with local search operators such as Swap and Relocate.
At each GLS step, the heuristic updates the execution time matrix
and selects a subset of jobs to perturb in order to escape local optima.
The goal is to design an effective heuristic strategy that modifies the execution time matrix
and identifies important jobs to perturb, so as to guide GLS toward schedules with a minimized makespan.

The code must define a function named `get_matrix_and_jobs` with the following function signature:
```python
def get_matrix_and_jobs(current_sequence: np.ndarray, time_matrix: np.ndarray, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:

    return new_matrix, perturb_jobs
```


Inputs:
- current_sequence: a NumPy array representing the current job sequence.
- time_matrix: a NumPy array of shape (n, m), representing execution times of jobs on machines.
- m: number of machines.
- n: number of jobs.

Outputs:
- new_matrix: a NumPy array of shape (n, m), representing the updated execution time matrix.
- perturb_jobs: a NumPy array containing indices of jobs selected for perturbation.

Constraints:
- Do NOT modify the input time_matrix in-place. Use time_matrix.copy() before modifying it.
- The output new_matrix must have the same shape as time_matrix.
- perturb_jobs must contain valid job indices in [0, n).
"""