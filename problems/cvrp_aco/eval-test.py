import os
import re
import sys
import ast
import json
import math
import time
import random
import inspect
import logging
import traceback
from copy import copy
from os import path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import distance_matrix

# =========================================================
# Config: 直接在这里改
# =========================================================
ALGORITHM = "simple_evol"   # 对应 saved_codes/<ALGORITHM>/
MODEL_NAMES = [
    "4.1-mini",

]
PROBLEM_SIZES = [50, 100, 200]
WORKERS = max(1, os.cpu_count() // 3)
N_ITERATIONS = 100
N_ANTS = 30
CAPACITY = 50
CODE_SUBDIR = "saved_codes"   # 启发式代码根目录
LOG_SUBDIR = "complexity_logs"  # 输出日志目录

# ---- thread control (process-wide) ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# ---- local imports (support both direct run and module run) ----
try:
    from .aco import ACO
    from .gen_inst import generate_datasets
except Exception:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.append(CURRENT_DIR)
    from aco import ACO
    from gen_inst import generate_datasets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data")
CODE_DIR = os.path.join(BASE_DIR, CODE_SUBDIR)
LOG_DIR = os.path.join(BASE_DIR, LOG_SUBDIR)
os.makedirs(LOG_DIR, exist_ok=True)


class Tee:
    """同时输出到终端和文件。"""
    def __init__(self, filename: str):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message: str):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def _ensure_dataset_exists(problem_sizes=(50, 100, 200)):
    missing = []
    for s in problem_sizes:
        fp = os.path.join(DATASET_DIR, f"test{s}_dataset.npy")
        if not os.path.isfile(fp):
            missing.append(fp)

    if missing:
        print("[*] Dataset missing. Generating datasets...")
        for x in missing:
            print(f"    - {x}")
        generate_datasets()
        print("[*] Dataset generation done.")


def extract_heuristic_function_names(py_path: str):
    with open(py_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=py_path)
    fn_names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and re.match(r"^heuristics(_v\d+)?$", node.name):
            fn_names.append(node.name)
    return fn_names


def load_callable_from_py(py_path: str, func_name: str):
    with open(py_path, "r", encoding="utf-8") as f:
        code_str = f.read()

    exec_ns = {
        "__builtins__": __builtins__,
        "np": np,
        "math": math,
        "random": random,
        "copy": copy,
    }
    exec(code_str, exec_ns)

    if func_name not in exec_ns or not callable(exec_ns[func_name]):
        raise ValueError(f"Function '{func_name}' not found or not callable in {py_path}")
    return exec_ns[func_name]


def solve_one_instance(node_pos, demand, heuristics_func):
    """
    保持与 CVRP ACO eval-test 一致的求解逻辑：
    - 构造距离矩阵
    - 调用 heuristics(...)
    - 运行固定参数的 ACO
    """
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1

    args = inspect.getfullargspec(heuristics_func).args
    if len(args) == 4:
        heu = heuristics_func(dist_mat.copy(), np.array(node_pos).copy(), np.array(demand).copy(), CAPACITY) + 1e-9
    elif len(args) == 2:
        heu = heuristics_func(dist_mat.copy(), np.array(demand).copy() / CAPACITY) + 1e-9
    else:
        raise ValueError(f"Unexpected heuristics signature for {heuristics_func.__name__}: {args}")

    heu = np.asarray(heu, dtype=float)
    heu[heu < 1e-9] = 1e-9

    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return float(obj.item()) if hasattr(obj, "item") else float(obj)


def evaluate_one_function_one_size(py_path: str, func_name: str, problem_size: int):
    dataset_path = path.join(DATASET_DIR, f"test{problem_size}_dataset.npy")
    if not path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    heuristics_func = load_callable_from_py(py_path, func_name)
    dataset = np.load(dataset_path)

    demands = dataset[:, :, 0]
    node_positions = dataset[:, :, 1:]
    n_instances = node_positions.shape[0]

    objs = []
    for node_pos, demand in zip(node_positions, demands):
        obj = solve_one_instance(node_pos, demand, heuristics_func)
        objs.append(float(obj))

    mean_obj = float(np.mean(objs)) if objs else float("inf")
    std_obj = float(np.std(objs)) if objs else float("nan")

    return {
        "source_py": os.path.basename(py_path),
        "function_name": func_name,
        "problem_size": int(problem_size),
        "n_instances": int(n_instances),
        "mean_score": mean_obj,
        "std_score": std_obj,
        "all_scores": objs,
    }


def _func_sort_key(fn: str):
    if fn == "heuristics":
        return (0, 0)
    m = re.match(r"heuristics_v(\d+)$", fn)
    if m:
        return (1, int(m.group(1)))
    return (2, fn)


def format_result_table(results, problem_sizes):
    by_func = {}
    for r in results:
        by_func.setdefault(r["function_name"], {})
        by_func[r["function_name"]][r["problem_size"]] = r["mean_score"]

    header = ["function_name"] + [f"size{sz}" for sz in problem_sizes]
    lines = ["\t".join(header)]

    for fn in sorted(by_func.keys(), key=_func_sort_key):
        row = [fn]
        for sz in problem_sizes:
            val = by_func[fn].get(sz, None)
            row.append(f"{val:.6f}" if val is not None else "NA")
        lines.append("\t".join(row))

    mean_row = ["mean_over_heuristics"]
    for sz in problem_sizes:
        vals = [by_func[fn][sz] for fn in by_func if sz in by_func[fn]]
        mean_row.append(f"{np.mean(vals):.6f}" if vals else "NA")
    lines.append("\t".join(mean_row))

    return lines


def evaluate_one_model(model_name: str, timestamp: str):
    py_path = os.path.join(CODE_DIR, ALGORITHM, f"{model_name}.py")
    run_dir = os.path.join(LOG_DIR, f"{ALGORITHM}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    save_json = os.path.join(run_dir, f"{ALGORITHM}_{model_name}_func_eval_results.json")
    save_fail_json = os.path.join(run_dir, f"{ALGORITHM}_{model_name}_func_eval_failed.json")

    if not path.isfile(py_path):
        print(f"[SKIP] Python source file not found: {py_path}")
        return None

    func_names = extract_heuristic_function_names(py_path)
    if len(func_names) == 0:
        print(f"[SKIP] No function named heuristics / heuristics_v* found in {py_path}")
        return None

    print("\n" + "=" * 80)
    print(f"[*] Algorithm folder: {ALGORITHM}")
    print(f"[*] Model name: {model_name}")
    print(f"[*] Source file: {py_path}")
    print(f"[*] Found {len(func_names)} functions:")
    for fn in sorted(func_names, key=_func_sort_key):
        print(f"    - {fn}")
    print(f"[*] Problem sizes: {PROBLEM_SIZES}")
    print(f"[*] Workers: {WORKERS}")
    print(f"[*] Dataset dir: {DATASET_DIR}")
    print("=" * 80)

    tasks = [(py_path, fn, size) for fn in func_names for size in PROBLEM_SIZES]

    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_task = {
            executor.submit(evaluate_one_function_one_size, py_path, fn, size): (fn, size)
            for py_path, fn, size in tasks
        }

        for future in as_completed(future_to_task):
            fn, size = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[*] Model: {model_name} | Function: {fn} | Average for {size}: {result['mean_score']}")
            except Exception as e:
                err_msg = traceback.format_exc()
                failed.append({
                    "model_name": model_name,
                    "function_name": fn,
                    "problem_size": size,
                    "error": str(e),
                    "traceback": err_msg,
                })
                print(f"[FAIL] model={model_name} | function={fn} | size={size}\n{e}")

    results.sort(key=lambda r: (_func_sort_key(r["function_name"]), r["problem_size"]))

    print(f"\n================ Final Results for {model_name} ================\n")
    table_lines = format_result_table(results, PROBLEM_SIZES)
    for line in table_lines:
        print(line)

    payload = {
        "algorithm": ALGORITHM,
        "model_name": model_name,
        "source_py": py_path,
        "problem_sizes": PROBLEM_SIZES,
        "n_iterations": N_ITERATIONS,
        "n_ants": N_ANTS,
        "capacity": CAPACITY,
        "results": results,
        "failed": failed,
        "summary_table": table_lines,
    }

    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[*] JSON results saved to: {save_json}")

    if failed:
        with open(save_fail_json, "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2, ensure_ascii=False)
        print(f"[!] {len(failed)} tasks failed.")
        print(f"[*] Failed details saved to: {save_fail_json}")
    else:
        print("\n[*] All tasks for this model finished successfully.")

    return {
        "model_name": model_name,
        "results": results,
        "failed": failed,
        "table_lines": table_lines,
        "json_path": save_json,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not path.isdir(os.path.join(CODE_DIR, ALGORITHM)):
        raise NotADirectoryError(f"Algorithm folder not found: {os.path.join(CODE_DIR, ALGORITHM)}")

    if not path.isdir(DATASET_DIR):
        raise NotADirectoryError(f"Dataset dir not found: {DATASET_DIR}")

    _ensure_dataset_exists(problem_sizes=PROBLEM_SIZES)

    log_file = os.path.join(LOG_DIR, f"{ALGORITHM}_{timestamp}_multi_model_eval_log.txt")
    original_stdout = sys.stdout
    tee = Tee(log_file)
    sys.stdout = tee

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    all_model_summaries = []
    total_start = time.time()

    try:
        print(f"[*] Algorithm: {ALGORITHM}")
        print(f"[*] Models to evaluate: {MODEL_NAMES}")
        print(f"[*] Problem sizes: {PROBLEM_SIZES}")
        print(f"[*] Dataset dir: {DATASET_DIR}")
        print(f"[*] Code dir: {CODE_DIR}")
        print(f"[*] Log dir: {LOG_DIR}")
        print(f"[*] ACO: iters={N_ITERATIONS}, ants={N_ANTS}, cap={CAPACITY}")
        print(f"[*] Workers: {WORKERS}")

        for model_name in MODEL_NAMES:
            summary = evaluate_one_model(model_name, timestamp)
            if summary is not None:
                all_model_summaries.append(summary)

        print("\n" + "#" * 100)
        print("[*] All model evaluations finished.")
        print("#" * 100)

        for summary in all_model_summaries:
            print(f"\n----- {summary['model_name']} -----")
            for line in summary["table_lines"]:
                print(line)

        total_elapsed = time.time() - total_start
        print(f"\n[*] Total eval time: {total_elapsed:.3f}s")
        print(f"[*] Text log saved to: {log_file}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()
