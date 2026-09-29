import time
import traceback
from dataclasses import dataclass, asdict
from typing import Optional
import signal
import numpy as np
from scipy.spatial import distance_matrix as calc_dist_matrix
from copy import copy
from pathlib import Path
import importlib.util


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data"


@dataclass
class TSPEvalResult:
    obj: Optional[float] = None
    time: Optional[float] = None
    error: Optional[str] = None
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


class HeuristicTimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise HeuristicTimeoutError("Code execution timeout")


class TSPEvalTool:
    def __init__(
        self,
        problem_size: int = 50,
        n_instances: int = 64,
        seed: int = 1234,
        data_path: Optional[str] = None,
        mode: str = "train",
        timeout: float = 60.0,
    ):
        self.problem_size = problem_size
        self.n_instances = n_instances
        self.seed = seed
        self.mode = mode
        self.timeout = timeout

        if data_path is None:
            data_path = DEFAULT_DATA_PATH
        else:
            data_path = Path(data_path)

        self.data_path = Path(data_path)

    def _load_instances(self, problem_size: int, mode: str, n_instances: int) -> np.ndarray:
        data_file = self.data_path / f"{mode}{problem_size}_dataset.npy"

        if data_file.exists():
            instances = np.load(data_file)
            n_instances = min(n_instances, instances.shape[0])
            return instances[:n_instances]

        np.random.seed(self.seed)
        return np.random.rand(n_instances, problem_size, 2)

    def _run_heuristic(self, select_next_node_func, node_positions: np.ndarray) -> float:
        problem_size = node_positions.shape[0]
        dist_mat = calc_dist_matrix(node_positions, node_positions)

        start_node = 0
        solution = [start_node]
        unvisited = set(range(problem_size))
        unvisited.remove(start_node)

        for _ in range(problem_size - 1):
            next_node = select_next_node_func(
                current_node=solution[-1],
                destination_node=start_node,
                unvisited_nodes=copy(unvisited),
                distance_matrix=dist_mat.copy(),
                # node_positions=node_positions.copy(),
            )
            solution.append(next_node)

            if next_node in unvisited:
                unvisited.remove(next_node)
            else:
                raise ValueError(f"Node {next_node} has been visited or does not exist")

        obj = 0.0
        for i in range(problem_size):
            obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]

        return float(obj)

    def _load_module_from_code_path(self, code_path: str):
        code_path = Path(code_path)
        module_name = f"{code_path.stem}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, str(code_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _evaluate_single_size(self, select_next_node_func, problem_size: int, mode: str, n_instances: int) -> float:
        instances = self._load_instances(problem_size, mode, n_instances)

        total_obj = 0.0
        for i in range(len(instances)):
            obj = self._run_heuristic(select_next_node_func, instances[i])
            total_obj += obj

        return float(total_obj / len(instances))

    def evaluate(self, code_path: str) -> TSPEvalResult:
        start_time = time.time()

        try:
            old_handler = None
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))

            try:
                module = self._load_module_from_code_path(code_path)

                if not hasattr(module, "select_next_node"):
                    return TSPEvalResult(
                        error="The function select_next_node is not defined in the code"
                    )

                select_next_node_func = module.select_next_node

                # -------------------------
                # train mode: single size
                # -------------------------
                if self.mode == "train":
                    avg_obj = self._evaluate_single_size(
                        select_next_node_func=select_next_node_func,
                        problem_size=self.problem_size,
                        mode="train",
                        n_instances=self.n_instances,
                    )
                    elapsed_time = time.time() - start_time
                    return TSPEvalResult(
                        obj=float(avg_obj),
                        time=float(elapsed_time),
                        details={
                            "mode": "train",
                            "size": self.problem_size,
                            "n_instances": self.n_instances,
                        },
                    )

                # -------------------------
                # test mode: fixed multi-size
                # -------------------------
                elif self.mode == "test":
                    test_sizes = [50, 100, 200]
                    per_size_obj = {}
                    total_obj = 0.0

                    for size in test_sizes:
                        avg_obj = self._evaluate_single_size(
                            select_next_node_func=select_next_node_func,
                            problem_size=size,
                            mode="test",
                            n_instances=self.n_instances,
                        )
                        per_size_obj[str(size)] = avg_obj
                        total_obj += avg_obj

                    overall_obj = total_obj / len(test_sizes)
                    elapsed_time = time.time() - start_time

                    return TSPEvalResult(
                        obj=float(overall_obj),
                        time=float(elapsed_time),
                        details={
                            "mode": "test",
                            "sizes": test_sizes,
                            "per_size_obj": per_size_obj,
                            "n_instances_each": self.n_instances,
                        },
                    )

                # -------------------------
                # eval mode: fixed multi-size
                # -------------------------
                elif self.mode == "val":
                    eval_sizes = [20, 50, 100, 200]
                    per_size_obj = {}
                    total_obj = 0.0

                    for size in eval_sizes:
                        avg_obj = self._evaluate_single_size(
                            select_next_node_func=select_next_node_func,
                            problem_size=size,
                            mode="val",
                            n_instances=self.n_instances,
                        )
                        per_size_obj[str(size)] = avg_obj
                        total_obj += avg_obj

                    overall_obj = total_obj / len(eval_sizes)
                    elapsed_time = time.time() - start_time

                    return TSPEvalResult(
                        obj=float(overall_obj),
                        time=float(elapsed_time),
                        details={
                            "mode": "val",
                            "sizes": eval_sizes,
                            "per_size_obj": per_size_obj,
                            "n_instances_each": self.n_instances,
                        },
                    )

                else:
                    return TSPEvalResult(error=f"Unsupported mode: {self.mode}")

            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)

        except HeuristicTimeoutError:
            return TSPEvalResult(error=f"Code execution timeout (limit {self.timeout} s)")

        except Exception as e:
            return TSPEvalResult(
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )


def evaluate_tsp_heuristic(code_path: str, **kwargs) -> TSPEvalResult:
    tool = TSPEvalTool(**kwargs)
    return tool.evaluate(code_path)