import numpy as np
def get_matrix_and_jobs(current_sequence: np.ndarray, time_matrix: np.ndarray, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:

    # keep matrix unchanged
    new_matrix = time_matrix.copy()

    # compute job total processing time
    job_sum = np.sum(time_matrix, axis=1)

    # select top-k jobs (EOH commonly uses k <= 5)
    k = min(3, n)
    perturb_jobs = np.argsort(-job_sum)[:k]

    return new_matrix, perturb_jobs