import os
import time
import traceback
import inspect
import importlib.util
import signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import distance_matrix

# ---- thread control: evaluator-level, process-wide ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

from .aco import ACO
from . import gen_inst


N_ITER = 100
N_ANTS = 30
CAPACITY = 50
EPS = 1e-9

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data"


@dataclass
class CVRPEvalResult:
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


class CVRPEvalTool:
    def __init__(
        self,
        problem_size: int = 50,
        n_instances: int = 64,
        seed: int = 1234,
        data_path: Optional[str] = None,
        mode: str = "train",
        timeout: float = 60.0,
    ):
        self.problem_size = int(problem_size)
        self.n_instances = int(n_instances)
        self.seed = seed
        self.mode = mode
        self.timeout = timeout

        if data_path is None:
            data_path = DEFAULT_DATA_PATH
        else:
            data_path = Path(data_path)

        self.data_path = Path(data_path)

    # ----------------- dataset helpers -----------------
    def _dataset_path(self, mode: str, size: int) -> Path:
        return self.data_path / f"{mode}{size}_dataset.npy"

    def _ensure_builtin_datasets(self) -> None:
        self.data_path.mkdir(parents=True, exist_ok=True)
        if any(self.data_path.glob("*.npy")):
            return

        # 兼容 generate_datasets(basepath=...) 和 generate_datasets()
        try:
            sig = inspect.signature(gen_inst.generate_datasets)
            if "basepath" in sig.parameters:
                gen_inst.generate_datasets(basepath=str(self.data_path))
            else:
                gen_inst.generate_datasets()
        except Exception:
            gen_inst.generate_datasets()

    def _load_instances(self, problem_size: int, mode: str, n_instances: int):
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._ensure_builtin_datasets()

        fp = self._dataset_path(mode, problem_size)
        if not fp.exists():
            try:
                sig = inspect.signature(gen_inst.generate_datasets)
                if "basepath" in sig.parameters:
                    gen_inst.generate_datasets(basepath=str(self.data_path))
                else:
                    gen_inst.generate_datasets()
            except Exception:
                gen_inst.generate_datasets()

        if not fp.exists():
            raise RuntimeError(f"Dataset file missing: {fp}")

        data = np.load(fp)
        data = data[: min(n_instances, data.shape[0])]
        demands, node_positions = data[:, :, 0], data[:, :, 1:]
        return demands, node_positions

    # ----------------- code loading -----------------
    def _load_module_from_code_path(self, code_path: str):
        code_path = Path(code_path)
        module_name = f"{code_path.stem}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, str(code_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # ----------------- evaluation core -----------------
    def _run_heuristic(self, node_pos, demand, heuristics_func) -> float:
        dist = distance_matrix(node_pos, node_pos)
        dist[np.diag_indices_from(dist)] = 1.0

        args = inspect.getfullargspec(heuristics_func).args
        if len(args) != 4:
            raise RuntimeError("heuristics must take 4 arguments: (distance_matrix, node_positions, demand, capacity)")

        heu = heuristics_func(
            dist.copy(),
            node_pos.copy(),
            demand.copy(),
            CAPACITY,
        )
        heu = np.asarray(heu, dtype=np.float64)

        if heu.shape != dist.shape:
            raise RuntimeError(f"wrong heuristic shape: got {heu.shape}, expected {dist.shape}")
        if not np.isfinite(heu).all():
            raise RuntimeError("heuristic contains NaN/Inf")

        heu = heu + EPS
        heu[heu < EPS] = EPS

        aco = ACO(dist, demand, heu, CAPACITY, n_ants=N_ANTS)
        obj = aco.run(N_ITER)
        return float(obj)

    def _evaluate_single_size(self, heuristics_func, problem_size: int, mode: str, n_instances: int) -> float:
        demands, node_positions = self._load_instances(problem_size, mode, n_instances)

        objs = []
        for node_pos, demand in zip(node_positions, demands):
            obj = self._run_heuristic(node_pos, demand, heuristics_func)
            objs.append(obj)

        return float(np.mean(objs))

    def evaluate(self, code_path: str) -> CVRPEvalResult:
        start_time = time.time()

        try:
            old_handler = None
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))

            try:
                module = self._load_module_from_code_path(code_path)

                if not hasattr(module, "heuristics"):
                    return CVRPEvalResult(
                        error="The function heuristics is not defined in the code"
                    )

                heuristics_func = module.heuristics
                if not callable(heuristics_func):
                    return CVRPEvalResult(
                        error="The attribute heuristics exists but is not callable"
                    )

                # -------------------------
                # train mode: single size
                # -------------------------
                if self.mode == "train":
                    avg_obj = self._evaluate_single_size(
                        heuristics_func=heuristics_func,
                        problem_size=self.problem_size,
                        mode="train",
                        n_instances=self.n_instances,
                    )
                    elapsed_time = time.time() - start_time
                    return CVRPEvalResult(
                        obj=float(avg_obj),
                        time=float(elapsed_time),
                        details={
                            "mode": "train",
                            "size": self.problem_size,
                            "n_instances": self.n_instances,
                            "objective_type": "min",
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
                            heuristics_func=heuristics_func,
                            problem_size=size,
                            mode="test",
                            n_instances=self.n_instances,
                        )
                        per_size_obj[str(size)] = avg_obj
                        total_obj += avg_obj

                    overall_obj = total_obj / len(test_sizes)
                    elapsed_time = time.time() - start_time

                    return CVRPEvalResult(
                        obj=float(overall_obj),
                        time=float(elapsed_time),
                        details={
                            "mode": "test",
                            "sizes": test_sizes,
                            "per_size_obj": per_size_obj,
                            "n_instances_each": self.n_instances,
                            "objective_type": "min",
                        },
                    )

                # -------------------------
                # val mode: fixed multi-size
                # -------------------------
                elif self.mode == "val":
                    eval_sizes = [50, 100, 200]
                    per_size_obj = {}
                    total_obj = 0.0

                    for size in eval_sizes:
                        avg_obj = self._evaluate_single_size(
                            heuristics_func=heuristics_func,
                            problem_size=size,
                            mode="val",
                            n_instances=self.n_instances,
                        )
                        per_size_obj[str(size)] = avg_obj
                        total_obj += avg_obj

                    overall_obj = total_obj / len(eval_sizes)
                    elapsed_time = time.time() - start_time

                    return CVRPEvalResult(
                        obj=float(overall_obj),
                        time=float(elapsed_time),
                        details={
                            "mode": "val",
                            "sizes": eval_sizes,
                            "per_size_obj": per_size_obj,
                            "n_instances_each": self.n_instances,
                            "objective_type": "min",
                        },
                    )

                else:
                    return CVRPEvalResult(error=f"Unsupported mode: {self.mode}")

            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)

        except HeuristicTimeoutError:
            return CVRPEvalResult(error=f"Code execution timeout (limit {self.timeout} s)")

        except Exception as e:
            return CVRPEvalResult(
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )