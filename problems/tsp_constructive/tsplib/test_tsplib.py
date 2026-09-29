import os
import sys
import time
import math
import random
import inspect
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


# ============================================================
# Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
TSPLIB_DIR = BASE_DIR / "tsplib_inst"
MODEL_NAMES = ["simple_evol", "funsearch", "eoh", "reevo","simple_new"]
LABELS_XLSX = BASE_DIR / "TSPlib_Problem_Labels.xlsx"
SELECTED_INSTANCES = {
    "ts225",
    "rat99",
    "bier127",
    "lin318",
    "eil51",
    "d493",
    "kroB100",
    "kroC100",
    "ch130",
    "pr299",
    "fl417",
    "kroA150",
    "pr264",
    "pr226",
    "pr439",
}

MAX_NODES = 500                  # 只测 500 以下
N_STARTS = 3                     # 每个 instance 随机 3 次起点
PER_HEURISTIC_WORKERS = 2        # 每个启发式最多 2 个 CPU core / worker
TASK_TIMEOUT = 1800              # 单次构建 tour 超时（秒）
GLOBAL_RANDOM_SEED = 1234

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ============================================================
# Logging
# ============================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        for f in self.files:
            if f not in (sys.__stdout__, sys.__stderr__):
                try:
                    f.close()
                except Exception:
                    pass


timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
log_path = LOG_DIR / f"eval_tsplib_parallel_{timestamp}.log"
log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

print(f"[LOG] TSPLib evaluation started at {timestamp}")
print(f"[LOG] TSPLIB_DIR = {TSPLIB_DIR}")
print(f"[LOG] MODEL_NAMES = {MODEL_NAMES}")
print(f"[LOG] MAX_NODES < {MAX_NODES}")
print(f"[LOG] N_STARTS = {N_STARTS}")
print(f"[LOG] PER_HEURISTIC_WORKERS = {PER_HEURISTIC_WORKERS}")
print()


# ============================================================
# TSPLib parsing
# ============================================================
def parse_tsplib_file(tsp_path: Path) -> np.ndarray:
    """
    读取 .tsp 文件中的 NODE_COORD_SECTION
    返回 shape = (n, 2) 的坐标数组
    """
    coords = []
    in_section = False

    with open(tsp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            upper_line = line.upper()
            if upper_line == "NODE_COORD_SECTION":
                in_section = True
                continue
            if upper_line == "EOF":
                break

            if in_section:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append([x, y])
                    except ValueError:
                        continue

    if not coords:
        raise ValueError(f"No coordinates parsed from {tsp_path}")

    return np.asarray(coords, dtype=np.float64)


def load_all_tsplib_instances(tsplib_dir: Path, max_nodes: int):

    instances = []

    for tsp_path in sorted(tsplib_dir.glob("*.tsp")):
        name = tsp_path.stem

        # 只保留指定的 15 个 instance
        if name not in SELECTED_INSTANCES:
            continue

        try:
            coords = parse_tsplib_file(tsp_path)
        except Exception as e:
            print(f"[WARN] Failed to parse {tsp_path.name}: {e}")
            continue

        n = coords.shape[0]

        # 只测小规模
        if n >= max_nodes:
            continue

        instances.append({
            "name": name,
            "path": str(tsp_path),
            "coords": coords,
            "n": n,
        })

    # 可选：检查有没有漏掉的 instance
    loaded_names = set([inst["name"] for inst in instances])
    missing = SELECTED_INSTANCES - loaded_names
    if missing:
        print(f"[WARN] Missing TSPLib instances: {sorted(missing)}")

    return instances


def load_tsplib_labels(xlsx_path: Path):
    if not xlsx_path.exists():
        raise FileNotFoundError(f"TSPLib label file not found: {xlsx_path}")

    df = pd.read_excel(xlsx_path)

    # 自动识别列名
    name_col = None
    size_col = None
    opt_col = None

    for c in df.columns:
        cl = str(c).strip().lower()
        if name_col is None and "name" in cl:
            name_col = c
        if size_col is None and ("size" in cl or "problem size" in cl):
            size_col = c
        if opt_col is None and ("label" in cl or "bound" in cl):
            opt_col = c

    if name_col is None or opt_col is None:
        raise ValueError(
            f"Cannot identify required columns in {xlsx_path}. "
            f"Found columns: {list(df.columns)}"
        )

    labels = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        if not name or name.lower() == "nan":
            continue

        try:
            opt_val = float(row[opt_col])
        except Exception:
            continue

        labels[name] = {
            "opt": opt_val,
            "size": None if size_col is None else row[size_col],
        }

    return labels
# ============================================================
# Heuristic loading
# ============================================================
def load_module_from_path(module_path: Path):
    module_name = f"{module_path.stem}_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_select_next_node_function(module):
    """
    自动寻找以 select_next_node 为前缀的 callable
    例如：
      - select_next_node_simpleevol
      - select_next_node_reevo
      - select_next_node_funsearch
      - select_next_node_eoh
      - 甚至直接叫 select_next_node 也行
    """
    candidates = []
    for name in dir(module):
        if name.startswith("select_next_node"):
            obj = getattr(module, name)
            if callable(obj):
                candidates.append((name, obj))

    if not candidates:
        raise ValueError("No callable function found with prefix 'select_next_node'.")

    # 优先 exact match，再退化到第一个 prefix match
    for name, func in candidates:
        if name == "select_next_node":
            return name, func

    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def call_heuristic(func, current_node, destination_node, unvisited_nodes, dist_mat):
    """
    兼容不同函数签名：
      - current_node
      - destination_node / 可选
      - unvisited_nodes 或 unvisited_near_nodes
      - distance_matrix
    """
    sig = inspect.signature(func)
    kwargs = {}

    for pname in sig.parameters:
        if pname == "current_node":
            kwargs[pname] = current_node
        elif pname == "destination_node":
            kwargs[pname] = destination_node
        elif pname in ("unvisited_nodes", "unvisited_near_nodes"):
            kwargs[pname] = set(unvisited_nodes)
        elif pname == "distance_matrix":
            kwargs[pname] = dist_mat
        else:
            # 其他参数不传，交给默认值
            pass

    return func(**kwargs)


# ============================================================
# TSP evaluation core
# ============================================================
def build_tour_with_heuristic(func, node_positions: np.ndarray, start_node: int) -> float:
    n = node_positions.shape[0]
    dist_mat = distance_matrix(node_positions, node_positions)

    solution = [start_node]
    unvisited = set(range(n))
    unvisited.remove(start_node)

    for _ in range(n - 1):
        next_node = call_heuristic(
            func=func,
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=unvisited,
            dist_mat=dist_mat,
        )

        if not isinstance(next_node, (int, np.integer)):
            raise TypeError(f"Heuristic returned non-integer node: {type(next_node)}")

        next_node = int(next_node)

        if next_node not in unvisited:
            raise ValueError(f"Invalid next_node={next_node}, not in unvisited set.")

        solution.append(next_node)
        unvisited.remove(next_node)

    obj = 0.0
    for i in range(n):
        obj += dist_mat[solution[i], solution[(i + 1) % n]]

    return float(obj)


def choose_random_start_nodes(n: int, k: int, seed: int):
    rng = random.Random(seed)
    nodes = list(range(n))
    if n >= k:
        return rng.sample(nodes, k)
    return [rng.choice(nodes) for _ in range(k)]


def eval_one_instance_three_starts(model_file: str, tsp_path: str, seed: int):
    """
    单个实例：随机 3 个起点，构建 tour 3 次，返回平均值
    注意：这个函数会在子进程中运行
    """
    tsp_path = Path(tsp_path)
    model_file = Path(model_file)

    t0 = time.perf_counter()

    module = load_module_from_path(model_file)
    func_name, func = find_select_next_node_function(module)

    coords = parse_tsplib_file(tsp_path)
    n = coords.shape[0]

    start_nodes = choose_random_start_nodes(n=n, k=N_STARTS, seed=seed)

    objs = []
    per_start = []

    for s in start_nodes:
        st = time.perf_counter()
        obj = build_tour_with_heuristic(func, coords, s)
        rt = time.perf_counter() - st
        objs.append(obj)
        per_start.append({
            "start_node": s,
            "objective": float(obj),
            "runtime_sec": float(rt),
        })

    total_time = time.perf_counter() - t0

    return {
        "problem": tsp_path.stem,
        "n": n,
        "model_file": model_file.name,
        "function_name": func_name,
        "mean_obj": float(np.mean(objs)),
        "std_obj": float(np.std(objs)),
        "starts": start_nodes,
        "per_start": per_start,
        "avg_runtime_per_start_sec": float(np.mean([x["runtime_sec"] for x in per_start])),
        "total_runtime_sec": float(total_time),
        "status": "ok",
        "error": None,
    }


def safe_eval_one_instance(model_file: str, tsp_path: str, seed: int):
    try:
        return eval_one_instance_three_starts(model_file, tsp_path, seed)
    except Exception as e:
        return {
            "problem": Path(tsp_path).stem,
            "n": None,
            "model_file": Path(model_file).name,
            "function_name": None,
            "mean_obj": None,
            "std_obj": None,
            "starts": None,
            "per_start": None,
            "avg_runtime_per_start_sec": None,
            "total_runtime_sec": None,
            "status": "fail",
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
        }


# ============================================================
# Per-model parallel runner
# ============================================================
def evaluate_single_model(model_name: str, tsplib_dir: str, max_nodes: int, base_seed: int, labels: dict):
    """
    一个模型内部最多开 2 个 worker process
    """
    tsplib_dir = Path(tsplib_dir)
    model_path = BASE_DIR / f"{model_name}.py"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    instances = load_all_tsplib_instances(tsplib_dir, max_nodes=max_nodes)
    print(f"[{model_name}] Loaded {len(instances)} instances with n < {max_nodes}")

    results = []

    with ProcessPoolExecutor(max_workers=PER_HEURISTIC_WORKERS) as executor:
        future_to_name = {}

        for idx, inst in enumerate(instances):
            # 保证不同 instance 的随机起点稳定可复现
            seed = base_seed + idx
            fut = executor.submit(
                safe_eval_one_instance,
                str(model_path),
                inst["path"],
                seed,
            )
            future_to_name[fut] = inst["name"]

        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                res = fut.result(timeout=TASK_TIMEOUT)
            except Exception as e:
                res = {
                    "problem": name,
                    "n": None,
                    "model_file": model_path.name,
                    "function_name": None,
                    "mean_obj": None,
                    "std_obj": None,
                    "starts": None,
                    "per_start": None,
                    "avg_runtime_per_start_sec": None,
                    "total_runtime_sec": None,
                    "status": "fail",
                    "error": f"{type(e).__name__}: {str(e)}",
                }
                
            opt_val = None
            gap = None
            if res["problem"] in labels and res["mean_obj"] is not None:
                opt_val = float(labels[res["problem"]]["opt"])
                gap = (res["mean_obj"] - opt_val) / opt_val * 100.0
                res["opt"] = opt_val
                res["gap_percent"] = gap
            else:
                res["opt"] = None
                res["gap_percent"] = None
                
            if res["status"] == "ok":
                if res["gap_percent"] is not None:
                    print(
                        f"[{model_name}] {res['problem']:<15s} "
                        f"n={res['n']:<4d} "
                        f"obj={res['mean_obj']:.4f} ± {res['std_obj']:.4f} "
                        f"gap={res['gap_percent']:.2f}% "
                        f"avg_t={res['avg_runtime_per_start_sec']:.3f}s"
                    )
                else:
                    print(
                        f"[{model_name}] {res['problem']:<15s} "
                        f"n={res['n']:<4d} "
                        f"obj={res['mean_obj']:.4f} ± {res['std_obj']:.4f} "
                        f"gap=N/A "
                        f"avg_t={res['avg_runtime_per_start_sec']:.3f}s"
                    )

            results.append(res)

    results.sort(key=lambda x: (x["problem"] is None, x["problem"]))
    return model_name, results


# ============================================================
# Main
# ============================================================
def main():
    all_model_results = {}
    labels = load_tsplib_labels(LABELS_XLSX)
    print(f"[LOG] Loaded {len(labels)} TSPLib labels from {LABELS_XLSX}")
    overall_t0 = time.perf_counter()

    # 四个启发式并行；每个启发式内部自己最多 2 个 worker process
    with ThreadPoolExecutor(max_workers=len(MODEL_NAMES)) as outer_executor:
        futures = {
            outer_executor.submit(
            evaluate_single_model,
            model_name,
            str(TSPLIB_DIR),
            MAX_NODES,
            GLOBAL_RANDOM_SEED,
            labels,
            ): model_name
            for i, model_name in enumerate(MODEL_NAMES)
        }

        for fut in as_completed(futures):
            model_name = futures[fut]
            try:
                mname, res = fut.result()
                all_model_results[mname] = res
                print(f"\n[DONE] {model_name}: {len(res)} instances finished.\n")
            except Exception as e:
                print(f"\n[FAIL] {model_name}: {type(e).__name__}: {e}\n")
                all_model_results[model_name] = []

    overall_elapsed = time.perf_counter() - overall_t0




    print("\n============================================================")
    print("[LOG] Evaluation completed.")
    print(f"[LOG] Total wall-clock time: {overall_elapsed:.2f}s")
    print(f"[LOG] Log file       : {log_path}")

    print("============================================================\n")
        
    print("\n================ FINAL SUMMARY ================\n")

    for model_name, results in all_model_results.items():
        valid = [r for r in results if r["status"] == "ok" and r.get("gap_percent") is not None]
        if not valid:
            print(f"{model_name}: ALL FAILED OR NO LABELS")
            continue

        mean_gap = np.mean([r["gap_percent"] for r in valid])
        std_gap = np.std([r["gap_percent"] for r in valid])
        print(f"{model_name}: avg gap over all labeled instances = {mean_gap:.4f}% ± {std_gap:.4f}%")
        
        
if __name__ == "__main__":
    main()