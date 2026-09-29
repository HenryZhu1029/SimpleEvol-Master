import os
import re
import ast
import json
import math
import loggingW
import traceback
import sys
from copy import copy
from os import path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix


# =========================================================
# Config: 直接在这里改，不通过 command line 传参
# =========================================================
ALGORITHM = "funsearch"
MODEL_NAMES = [
    "4.1-mini",

]
PROBLEM_SIZES = [50]
WORKERS = max(1, os.cpu_count() // 3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LOG_DIR = os.path.join(BASE_DIR, "complexity_logs")
os.makedirs(LOG_DIR, exist_ok=True)


class Tee:
    """
    同时输出到终端和文件
    """
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def extract_select_next_node_function_names(py_path: str):
    with open(py_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=py_path)

    fn_names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("select_next_node"):
            fn_names.append(node.name)

    return fn_names


def load_callable_from_py(py_path: str, func_name: str):
    with open(py_path, "r", encoding="utf-8") as f:
        code_str = f.read()

    exec_ns = {
        "__builtins__": __builtins__,
        "np": np,
        "math": math,
        "copy": copy,
    }

    exec(code_str, exec_ns)

    if func_name not in exec_ns or not callable(exec_ns[func_name]):
        raise ValueError(f"Function '{func_name}' not found or not callable in {py_path}")

    return exec_ns[func_name]


def eval_heuristic(node_positions: np.ndarray, select_next_node_func) -> float:
    problem_size = node_positions.shape[0]
    dist_mat = distance_matrix(node_positions, node_positions)

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
        )

        if next_node not in unvisited:
            raise KeyError(f"Node {next_node} is invalid or already visited.")

        solution.append(next_node)
        unvisited.remove(next_node)

    obj = 0.0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]

    return float(obj)


def evaluate_one_function_one_size(
    py_path: str,
    func_name: str,
    problem_size: int,
    dataset_dir: str,
):
    dataset_path = path.join(dataset_dir, f"test{problem_size}_dataset.npy")
    if not path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    select_next_node_func = load_callable_from_py(py_path, func_name)

    node_positions = np.load(dataset_path)
    n_instances = node_positions.shape[0]

    objs = []
    for i in range(n_instances):
        obj = eval_heuristic(node_positions[i], select_next_node_func)
        objs.append(obj)

    mean_obj = float(np.mean(objs))
    std_obj = float(np.std(objs))

    return {
        "source_py": os.path.basename(py_path),
        "function_name": func_name,
        "problem_size": problem_size,
        "n_instances": int(n_instances),
        "mean_score": mean_obj,
        "std_score": std_obj,
        "all_scores": objs,
    }


def format_result_table(results, problem_sizes):
    by_func = {}
    for r in results:
        by_func.setdefault(r["function_name"], {})
        by_func[r["function_name"]][r["problem_size"]] = r["mean_score"]

    header = ["function_name"] + [f"size{sz}" for sz in problem_sizes]
    lines = ["\t".join(header)]

    def sort_key(fn):
        if fn == "select_next_node":
            return (0, 0)
        m = re.match(r"select_next_node_v(\d+)$", fn)
        if m:
            return (1, int(m.group(1)))
        return (2, fn)

    for fn in sorted(by_func.keys(), key=sort_key):
        row = [fn]
        for sz in problem_sizes:
            val = by_func[fn].get(sz, None)
            row.append(f"{val:.6f}" if val is not None else "NA")
        lines.append("\t".join(row))

    mean_row = ["mean"]
    for sz in problem_sizes:
        vals = [by_func[fn][sz] for fn in by_func if sz in by_func[fn]]
        mean_row.append(f"{np.mean(vals):.6f}" if vals else "NA")
    lines.append("\t".join(mean_row))

    return lines


def evaluate_one_model(model_name: str, timestamp: str):
    py_path = os.path.join(BASE_DIR, "saved_codes", ALGORITHM, f"{model_name}.py")
    
    run_dir = os.path.join(LOG_DIR, f"{ALGORITHM}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    save_json = os.path.join(run_dir, f"{ALGORITHM}_{model_name}_func_eval_results.json")

    save_fail_json = os.path.join(run_dir, f"{ALGORITHM}_{model_name}_func_eval_failed.json")

    if not path.isfile(py_path):
        print(f"[SKIP] Python source file not found: {py_path}")
        return None

    func_names = extract_select_next_node_function_names(py_path)
    if len(func_names) == 0:
        print(f"[SKIP] No function starting with 'select_next_node' found in {py_path}")
        return None

    print("\n" + "=" * 80)
    print(f"[*] Algorithm folder: {ALGORITHM}")
    print(f"[*] Model name: {model_name}")
    print(f"[*] Source file: {py_path}")
    print(f"[*] Found {len(func_names)} functions:")
    for fn in func_names:
        print(f"    - {fn}")
    print(f"[*] Problem sizes: {PROBLEM_SIZES}")
    print(f"[*] Workers: {WORKERS}")
    print(f"[*] Dataset dir: {DATASET_DIR}")
    print("=" * 80)

    tasks = []
    for fn in func_names:
        for size in PROBLEM_SIZES:
            tasks.append((py_path, fn, size, DATASET_DIR))

    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_task = {
            executor.submit(
                evaluate_one_function_one_size,
                py_path,
                fn,
                size,
                dataset_dir,
            ): (fn, size)
            for py_path, fn, size, dataset_dir in tasks
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc=f"Evaluating {model_name}"
        ):
            fn, size = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"[*] Running ...")
                print(f"[*] Model: {model_name}")
                print(f"[*] Function: {fn}")
                print(f"[*] Average for {size}: {result['mean_score']}")
            except Exception as e:
                err_msg = traceback.format_exc()
                failed.append({
                    "model_name": model_name,
                    "function_name": fn,
                    "problem_size": size,
                    "error": str(e),
                    "traceback": err_msg,
                })
                print(f"[FAIL] model={model_name} | {fn} | size={size}\n{e}")

    def sort_key_result(r):
        fn = r["function_name"]
        if fn == "select_next_node":
            return (0, 0, r["problem_size"])
        m = re.match(r"select_next_node_v(\d+)$", fn)
        if m:
            return (1, int(m.group(1)), r["problem_size"])
        return (2, fn, r["problem_size"])

    results.sort(key=sort_key_result)

    print(f"\n================ Final Results for {model_name} ================\n")
    table_lines = format_result_table(results, PROBLEM_SIZES)
    for line in table_lines:
        print(line)

    payload = {
        "algorithm": ALGORITHM,
        "model_name": model_name,
        "source_py": py_path,
        "problem_sizes": PROBLEM_SIZES,
        "results": results,
        "failed": failed,
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
    }


def main():
    algo_dir = os.path.join(BASE_DIR, "saved_codes", ALGORITHM)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not path.isdir(algo_dir):
        raise NotADirectoryError(f"Algorithm folder not found: {algo_dir}")

    if not path.isdir(DATASET_DIR):
        raise NotADirectoryError(f"Dataset dir not found: {DATASET_DIR}")

    log_file = os.path.join(LOG_DIR, f"{ALGORITHM}_{timestamp}_multi_model_eval_log.txt")

    original_stdout = sys.stdout
    tee = Tee(log_file)
    sys.stdout = tee

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    all_model_summaries = []

    try:
        print(f"[*] Algorithm: {ALGORITHM}")
        print(f"[*] Models to evaluate: {MODEL_NAMES}")
        print(f"[*] Problem sizes: {PROBLEM_SIZES}")
        print(f"[*] Dataset dir: {DATASET_DIR}")
        print(f"[*] Log dir: {LOG_DIR}")

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

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()