import os
import time
import types
import warnings
import traceback
import importlib.util
import signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass
import numpy as np
from numba import jit

from .gen_inst import generate_datasets


DEFAULT_TRAIN_DATA_PATH = Path(__file__).resolve().parent / "data"
DEFAULT_TESTING_DATA_PATH = Path(__file__).resolve().parent / "TestingData"


# ======================
# Core Scheduling Utils
# ======================

@jit(nopython=True)
def makespan(order, tasks, m):
    times = np.zeros(m)
    for j in order:
        times[0] += tasks[j, 0]
        for k in range(1, m):
            if times[k] < times[k - 1]:
                times[k] = times[k - 1]
            times[k] += tasks[j, k]
    return np.max(times)


@jit(nopython=True)
def local_search(seq, cmax, tasks, m):
    best = seq[:]
    best_c = cmax

    n = len(seq)

    # Swap
    for i in range(n):
        for j in range(i + 1, n):
            tmp = best[:]
            tmp[i], tmp[j] = tmp[j], tmp[i]
            c = makespan(tmp, tasks, m)
            if c < best_c:
                best, best_c = tmp, c

    # Relocate
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tmp = best[:]
            v = tmp.pop(i)
            tmp.insert(j, v)
            c = makespan(tmp, tasks, m)
            if c < best_c:
                best, best_c = tmp, c

    return best, best_c


@jit(nopython=True)
def local_search_perturb(seq, cmax, tasks, m, jobs):
    best = seq[:]
    best_c = cmax

    for i in jobs:
        for j in range(i + 1, len(seq)):
            tmp = best[:]
            tmp[i], tmp[j] = tmp[j], tmp[i]
            c = makespan(tmp, tasks, m)
            if c < best_c:
                best, best_c = tmp, c

    for i in jobs:
        for j in range(len(seq)):
            tmp = best[:]
            v = tmp.pop(i)
            tmp.insert(j, v)
            c = makespan(tmp, tasks, m)
            if c < best_c:
                best, best_c = tmp, c

    return best, best_c


def sum_and_order(tasks_val, machines_val, tasks):
    tab = []
    tab1 = []
    for i in range(0, tasks_val):
        tab.append(0)
        tab1.append(0)
    for j in range(0, tasks_val):
        for k in range(0, machines_val):
            tab[j] += tasks[j][k]
    place = 0
    iter_ = 0
    while iter_ != tasks_val:
        max_time = 1
        for i in range(0, tasks_val):
            if max_time < tab[i]:
                max_time = tab[i]
                place = i
        tab[place] = 1
        tab1[iter_] = place
        iter_ = iter_ + 1
    return tab1


def insertNEH(sequence, position, value):
    new_seq = sequence[:]
    new_seq.insert(position, value)
    return new_seq


def neh(tasks):
    tasks_val, machines_val = tasks.shape

    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]

    for i in range(1, tasks_val):
        min_cmax = float("inf")
        best_seq = None
        for j in range(0, i + 1):
            tmp = insertNEH(current_seq, j, order[i])
            cmax_tmp = makespan(tmp, tasks, machines_val)
            if min_cmax > cmax_tmp:
                best_seq = tmp
                min_cmax = cmax_tmp
        current_seq = best_seq

    return current_seq, makespan(current_seq, tasks, machines_val)


# ======================
# Taillard Parsing
# ======================

def read_taillard_file(path: Path):
    text = path.read_text().splitlines()
    i = 0
    insts = []

    def _next_nonempty(idx):
        while idx < len(text) and text[idx].strip() == "":
            idx += 1
        return idx

    while i < len(text):
        i = _next_nonempty(i)
        if i >= len(text):
            break

        line = text[i].strip().lower()
        if not line.startswith("number of jobs"):
            i += 1
            continue

        i += 1
        i = _next_nonempty(i)
        if i >= len(text):
            break

        header_nums = text[i].split()
        if len(header_nums) < 5:
            raise ValueError(f"Bad header numeric line in {path} at line {i+1}: {text[i]}")
        n = int(header_nums[0])
        m = int(header_nums[1])
        seed = int(header_nums[2])
        ub = int(header_nums[3])
        lb = int(header_nums[4])

        i += 1
        i = _next_nonempty(i)
        if i >= len(text):
            break

        if "processing times" not in text[i].lower():
            raise ValueError(f"Expected 'processing times' in {path} near line {i+1}, got: {text[i]}")
        i += 1

        mat_m_by_n = []
        for _ in range(m):
            i = _next_nonempty(i)
            if i >= len(text):
                raise ValueError(f"Unexpected EOF while reading processing times in {path}")
            row = text[i].split()
            if len(row) < n:
                raise ValueError(
                    f"Processing time row too short in {path} at line {i+1}: need {n}, got {len(row)}"
                )
            mat_m_by_n.append([int(x) for x in row[:n]])
            i += 1

        mat_m_by_n = np.asarray(mat_m_by_n, dtype=np.float64)  # (m, n)
        tasks = mat_m_by_n.T.copy()  # -> (n, m)

        insts.append(
            {"tasks": tasks, "n": n, "m": m, "seed": seed, "ub": ub, "lb": lb}
        )

    if not insts:
        raise ValueError(f"No Taillard instances parsed from file: {path}")
    return insts


def gap_percent(cmax: float, ub: float):
    if not np.isfinite(cmax) or ub <= 0:
        return float("inf")
    return (cmax - ub) / ub * 100.0


# ======================
# Eval Result
# ======================

@dataclass
class FSSPEvalResult:
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


# ======================
# Eval Tool
# ======================

class FSSPEvalTool:
    def __init__(
        self,
        problem_size: int = 50,
        n_instances: int = 16,
        seed: int = 2024,
        data_path: Optional[str] = None,
        mode: str = "train",
        timeout: float = 60.0,
    ):
        self.problem_size = int(problem_size)
        self.n_instances = int(n_instances)
        self.seed = seed
        self.mode = mode
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive (seconds per heuristic evaluation)")

        self.train_data_path = Path(data_path) if data_path is not None else DEFAULT_TRAIN_DATA_PATH
        self.testing_data_path = DEFAULT_TESTING_DATA_PATH

        # Paper: at most 1000 GLS iterations and 60 seconds per instance.
        self.iter_max = 1000
        self.time_limit = 60.0

    # -------------------------
    # Loading candidate code
    # -------------------------
    def _load_module_from_code_path(self, code_path: str):
        code_path = Path(code_path)
        module_name = f"{code_path.stem}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(module_name, str(code_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    # -------------------------
    # Train data loading
    # -------------------------
    def _ensure_train_instances(self):
        self.train_data_path.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(self.train_data_path.glob("*.txt"))
        if txt_files:
            return txt_files
        generate_datasets(n_jobs=self.problem_size, num_instances=max(self.n_instances, 3), seed=self.seed)
        return sorted(self.train_data_path.glob("*.txt"))

    def _load_train_instances(self):
        txt_files = self._ensure_train_instances()
        txt_files = txt_files[: min(self.n_instances, len(txt_files))]

        insts = []
        for file in txt_files:
            with open(file, "r") as f:
                n, m = map(int, f.readline().split())
                tasks = np.zeros((n, m), dtype=np.float64)
                for i in range(n):
                    row = f.readline().split()
                    for j in range(m):
                        tasks[i, j] = float(row[2 * j + 1])
            insts.append({"tasks": tasks, "name": file.name})
        return insts

    # -------------------------
    # GLS core
    # -------------------------
    def _gls_single(self, tasks, heuristic_module):
        start = time.monotonic()
        cmax_best = 1e10

        seq, cmax = neh(tasks)
        n = len(seq)

        best_seq = seq
        cmax_best = cmax

        it = 0

        while time.monotonic() - start < self.time_limit and it < self.iter_max:
            seq, _ = local_search(seq, cmax, tasks, tasks.shape[1])
            cmax = makespan(seq, tasks, tasks.shape[1])

            if cmax < cmax_best:
                best_seq = seq
                cmax_best = cmax

            new_matrix, jobs = heuristic_module.get_matrix_and_jobs(
                np.array(seq), tasks.copy(), tasks.shape[1], n
            )

            if len(jobs) <= 1:
                return float("inf")

            if len(jobs) > 5:
                jobs = jobs[:5]

            new_matrix = np.asarray(new_matrix, dtype=np.float64)
            if new_matrix.shape != tasks.shape:
                return float("inf")

            cmax = makespan(seq, new_matrix, new_matrix.shape[1])
            seq, _ = local_search_perturb(
                seq, cmax, new_matrix, new_matrix.shape[1], np.array(jobs)
            )

            it += 1

            if it % 50 == 0:
                seq = best_seq
                cmax = cmax_best

        return float(cmax_best)

    # -------------------------
    # Evaluation by mode
    # -------------------------
    def _evaluate_train(self, heuristic_module):
        insts = self._load_train_instances()

        objs = []
        per_case_obj = {}
        for inst in insts:
            obj = self._gls_single(inst["tasks"], heuristic_module)
            objs.append(obj)
            per_case_obj[inst["name"]] = obj

        return float(np.mean(objs)), per_case_obj

    def _evaluate_taillard(self, heuristic_module):
        if not self.testing_data_path.exists():
            raise RuntimeError(f"TestingData folder not found: {self.testing_data_path}")

        # 直接使用所有 Taillard 文件（11 个）
        files = sorted([p for p in self.testing_data_path.glob("t_j*_m*")])

        if not files:
            raise RuntimeError(f"No Taillard files found in {self.testing_data_path}")

        all_gaps = []
        per_file_gap = {}

        for fp in files:
            insts = read_taillard_file(fp)
            file_gaps = []

            for block_id, inst in enumerate(insts):
                tasks = inst["tasks"]
                ub = inst["ub"]

                cmax = self._gls_single(tasks, heuristic_module)
                gp = gap_percent(cmax, ub)

                file_gaps.append(gp)
                all_gaps.append(gp)

            per_file_gap[fp.name] = float(np.mean(file_gaps)) if file_gaps else float("inf")

        return float(np.mean(all_gaps)), per_file_gap

    # -------------------------
    # Public evaluate
    # -------------------------
    def evaluate(self, code_path: str) -> FSSPEvalResult:
        start_time = time.time()

        try:
            old_handler = None
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    module = self._load_module_from_code_path(code_path)

                if not hasattr(module, "get_matrix_and_jobs"):
                    return FSSPEvalResult(
                        error="The function get_matrix_and_jobs is not defined in the code"
                    )

                func = module.get_matrix_and_jobs
                if not callable(func):
                    return FSSPEvalResult(
                        error="The attribute get_matrix_and_jobs exists but is not callable"
                    )

                if self.mode == "train":
                    avg_obj, per_case_obj = self._evaluate_train(module)
                    elapsed_time = time.time() - start_time
                    return FSSPEvalResult(
                        obj=float(avg_obj),
                        time=float(elapsed_time),
                        details={
                            "mode": "train",
                            "size": self.problem_size,
                            "n_instances": len(per_case_obj),
                            "per_case_obj": per_case_obj,
                            "objective_type": "min",
                            "metric": "makespan",
                        },
                    )

                elif self.mode in ("val", "test"):
                    avg_gap, per_file_gap = self._evaluate_taillard(module)
                    elapsed_time = time.time() - start_time
                    return FSSPEvalResult(
                        obj=float(avg_gap),
                        time=float(elapsed_time),
                        details={
                            "mode": self.mode,
                            "benchmark": "taillard_full",
                            "n_files": len(per_file_gap),
                            "files": list(per_file_gap.keys()),
                            "per_case_obj": per_file_gap,
                            "objective_type": "min",
                            "metric": "avg_gap_percent_vs_ub",
                        },
                    )

                else:
                    return FSSPEvalResult(error=f"Unsupported mode: {self.mode}")

            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)

        except HeuristicTimeoutError:
            return FSSPEvalResult(error=f"Code execution timeout (limit {self.timeout} s per heuristic evaluation)")

        except Exception as e:
            return FSSPEvalResult(
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )
