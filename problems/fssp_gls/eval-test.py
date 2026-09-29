import os
import re
import ast
import json
import time
import math
import logging
import traceback
import sys
from copy import copy
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# =========================================================
# Config: 直接在这里改，不通过 command line 传参
# =========================================================
ALGORITHM = "simple_evol"
MODEL_NAMES = [
    "4.1-mini",

]

WORKERS = min(6, max(1, os.cpu_count() // 3))   # 最多 6 个 worker
TIME_LIMIT = 30.0
ITER_MAX = 1000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DIR = os.path.join(BASE_DIR, "TestingData")
SAVED_CODES_DIR = os.path.join(BASE_DIR, "saved_codes")
LOG_DIR = os.path.join(BASE_DIR, "eval_test_logs")
os.makedirs(LOG_DIR, exist_ok=True)


# =========================================================
# Tee
# =========================================================
class Tee:
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


# =========================================================
# Core Scheduling Utils
# 与 evaluator.py 对齐
# =========================================================
def makespan(order, tasks, m):
    times = np.zeros(m)
    for j in order:
        times[0] += tasks[j, 0]
        for k in range(1, m):
            if times[k] < times[k - 1]:
                times[k] = times[k - 1]
            times[k] += tasks[j, k]
    return np.max(times)


def local_search(seq, cmax, tasks, m):
    best = seq[:]
    best_c = cmax
    n = len(seq)

    for i in range(n):
        for j in range(i + 1, n):
            tmp = best[:]
            tmp[i], tmp[j] = tmp[j], tmp[i]
            c = makespan(tmp, tasks, m)
            if c < best_c:
                best, best_c = tmp, c

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
    for i in range(tasks_val):
        tab.append(0)
        tab1.append(0)
    for j in range(tasks_val):
        for k in range(machines_val):
            tab[j] += tasks[j][k]
    place = 0
    iter_ = 0
    while iter_ != tasks_val:
        max_time = 1
        for i in range(tasks_val):
            if max_time < tab[i]:
                max_time = tab[i]
                place = i
        tab[place] = 1
        tab1[iter_] = place
        iter_ += 1
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


# =========================================================
# Taillard Parsing
# =========================================================
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

        mat_m_by_n = np.asarray(mat_m_by_n, dtype=np.float64)
        tasks = mat_m_by_n.T.copy()

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


def nm_from_name(p: Path):
    m = re.search(r"_j(\d+)_m(\d+)", p.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (10**9, 10**9)


# =========================================================
# Heuristic extraction / loading
# =========================================================
def heuristic_sort_key(fn):
    if fn == "get_matrix_and_jobs":
        return (0, 0)
    m = re.match(r"get_matrix_and_jobs_v(\d+)$", fn)
    if m:
        return (1, int(m.group(1)))
    return (2, fn)


def extract_heuristic_function_names(py_path: str):
    with open(py_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=py_path)

    fn_names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("get_matrix_and_jobs"):
            fn_names.append(node.name)

    return sorted(fn_names, key=heuristic_sort_key)


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


# =========================================================
# GLS solve for one instance
# =========================================================
def gls_solve_single(tasks: np.ndarray, heuristic_func, time_limit: float, iter_max: int):
    seq, cmax = neh(tasks)
    n = len(seq)

    best_seq = seq
    cmax_best = cmax

    start = time.time()
    it = 0

    while time.time() - start < time_limit and it < iter_max:
        seq, _ = local_search(seq, cmax, tasks, tasks.shape[1])
        cmax = makespan(seq, tasks, tasks.shape[1])

        if cmax < cmax_best:
            best_seq = seq
            cmax_best = cmax

        new_matrix, jobs = heuristic_func(
            np.array(seq), tasks.copy(), tasks.shape[1], n
        )

        if len(jobs) <= 1:
            return float("inf"), best_seq

        if len(jobs) > 5:
            jobs = jobs[:5]

        new_matrix = np.asarray(new_matrix, dtype=np.float64)
        if new_matrix.shape != tasks.shape:
            return float("inf"), best_seq

        cmax = makespan(seq, new_matrix, new_matrix.shape[1])
        seq, _ = local_search_perturb(
            seq, cmax, new_matrix, new_matrix.shape[1], np.array(jobs)
        )

        it += 1

        if it % 50 == 0:
            seq = best_seq
            cmax = cmax_best

    return float(cmax_best), best_seq


# =========================================================
# Worker: one function on one file
# =========================================================
def evaluate_one_function_one_file(
    py_path: str,
    func_name: str,
    taillard_file: str,
    time_limit: float,
    iter_max: int,
):
    heuristic_func = load_callable_from_py(py_path, func_name)
    fp = Path(taillard_file)

    insts = read_taillard_file(fp)
    file_gaps = []
    block_results = []

    for block_id, inst in enumerate(insts):
        tasks = inst["tasks"]
        ub = inst["ub"]
        lb = inst["lb"]
        n = inst["n"]
        m = inst["m"]

        cmax, _ = gls_solve_single(tasks, heuristic_func, time_limit=time_limit, iter_max=iter_max)
        gp = gap_percent(cmax, ub)

        file_gaps.append(gp)
        block_results.append({
            "block_id": int(block_id),
            "n": int(n),
            "m": int(m),
            "lb": float(lb),
            "ub": float(ub),
            "cmax": float(cmax),
            "gap_percent": float(gp),
        })

    return {
        "source_py": os.path.basename(py_path),
        "function_name": func_name,
        "taillard_file": fp.name,
        "n_blocks": len(insts),
        "mean_gap": float(np.mean(file_gaps)) if file_gaps else float("inf"),
        "median_gap": float(np.median(file_gaps)) if file_gaps else float("inf"),
        "all_gaps": [float(x) for x in file_gaps],
        "block_results": block_results,
    }


# =========================================================
# Aggregate
# =========================================================
def aggregate_function_results(results):
    by_func = {}
    for r in results:
        by_func.setdefault(r["function_name"], [])
        by_func[r["function_name"]].append(r)

    function_summaries = []
    for func_name, items in by_func.items():
        items = sorted(items, key=lambda x: nm_from_name(Path(x["taillard_file"])))

        all_gaps = []
        per_file_mean_map = {}
        for item in items:
            all_gaps.extend(item["all_gaps"])
            per_file_mean_map[item["taillard_file"]] = float(item["mean_gap"])

        function_summaries.append({
            "function_name": func_name,
            "n_files": len(items),
            "files": [x["taillard_file"] for x in items],
            "mean_gap_over_all_blocks": float(np.mean(all_gaps)) if all_gaps else float("inf"),
            "median_gap_over_all_blocks": float(np.median(all_gaps)) if all_gaps else float("inf"),
            "mean_gap_over_files": float(np.mean(list(per_file_mean_map.values()))) if per_file_mean_map else float("inf"),
            "per_file_mean_map": per_file_mean_map,
            "per_file_results": items,
        })

    function_summaries.sort(key=lambda x: heuristic_sort_key(x["function_name"]))
    return function_summaries


def aggregate_across_heuristics_per_file(results):
    by_file = {}
    for r in results:
        by_file.setdefault(r["taillard_file"], [])
        by_file[r["taillard_file"]].append(float(r["mean_gap"]))

    summary = []
    for tf, vals in by_file.items():
        summary.append({
            "taillard_file": tf,
            "n_heuristics": len(vals),
            "mean_gap_across_heuristics": float(np.mean(vals)) if vals else float("inf"),
            "median_gap_across_heuristics": float(np.median(vals)) if vals else float("inf"),
        })

    summary.sort(key=lambda x: nm_from_name(Path(x["taillard_file"])))
    return summary

def aggregate_global_overall(results):
    all_file_means = [float(r["mean_gap"]) for r in results]

    all_block_gaps = []
    for r in results:
        all_block_gaps.extend(r["all_gaps"])

    return {
        "overall_mean_across_all_heuristics_and_files": float(np.mean(all_file_means)) if all_file_means else float("inf"),
        "overall_median_across_all_heuristics_and_files": float(np.median(all_file_means)) if all_file_means else float("inf"),
        "overall_mean_across_all_heuristics_and_all_blocks": float(np.mean(all_block_gaps)) if all_block_gaps else float("inf"),
        "overall_median_across_all_heuristics_and_all_blocks": float(np.median(all_block_gaps)) if all_block_gaps else float("inf"),
    }
# =========================================================
# Print helpers
# =========================================================
def print_per_function_report(function_summaries):
    print("\n" + "=" * 100)
    print("[1] Per-heuristic per-file mean gap")
    print("=" * 100)

    for item in function_summaries:
        print(f"\n[{item['function_name']}]")
        for tf in item["files"]:
            print(f"{tf}\t{item['per_file_mean_map'][tf]:.6f}%")
        print(f"overall_mean_gap_over_all_blocks\t{item['mean_gap_over_all_blocks']:.6f}%")
        print(f"overall_mean_gap_over_files\t{item['mean_gap_over_files']:.6f}%")


def print_across_heuristics_report(across_file_summary):
    print("\n" + "=" * 100)
    print("[2] Mean gap across all heuristics for each Taillard file")
    print("=" * 100)
    print("[Across all heuristics]")
    for item in across_file_summary:
        print(f"{item['taillard_file']}\t{item['mean_gap_across_heuristics']:.6f}%")


def print_compact_table(function_summaries, taillard_file_names):
    print("\n" + "=" * 100)
    print("[3] Compact summary table")
    print("=" * 100)
    header = ["function_name"] + taillard_file_names + ["overall_blocks_mean", "overall_files_mean"]
    print("\t".join(header))

    for item in function_summaries:
        row = [item["function_name"]]
        for tf in taillard_file_names:
            val = item["per_file_mean_map"].get(tf, None)
            row.append(f"{val:.6f}" if val is not None else "NA")
        row.append(f"{item['mean_gap_over_all_blocks']:.6f}")
        row.append(f"{item['mean_gap_over_files']:.6f}")
        print("\t".join(row))

def print_global_overall_report(global_overall):
    print("\n" + "=" * 100)
    print("[4] Global overall summary")
    print("=" * 100)
    print(f"overall_mean_across_all_heuristics_and_files\t{global_overall['overall_mean_across_all_heuristics_and_files']:.6f}%")
    print(f"overall_median_across_all_heuristics_and_files\t{global_overall['overall_median_across_all_heuristics_and_files']:.6f}%")
    print(f"overall_mean_across_all_heuristics_and_all_blocks\t{global_overall['overall_mean_across_all_heuristics_and_all_blocks']:.6f}%")
    print(f"overall_median_across_all_heuristics_and_all_blocks\t{global_overall['overall_median_across_all_heuristics_and_all_blocks']:.6f}%")
# =========================================================
# Evaluate one model
# =========================================================
def evaluate_one_model(model_name: str, timestamp: str, taillard_files: list[str]):
    py_path = os.path.join(SAVED_CODES_DIR, ALGORITHM, f"{model_name}.py")

    if not os.path.isfile(py_path):
        print(f"[SKIP] Python source file not found: {py_path}")
        return None

    func_names = extract_heuristic_function_names(py_path)
    if len(func_names) == 0:
        print(f"[SKIP] No function starting with 'get_matrix_and_jobs' found in {py_path}")
        return None

    taillard_file_names = [os.path.basename(x) for x in taillard_files]

    save_txt = os.path.join(LOG_DIR, f"{ALGORITHM}_{model_name}_{timestamp}_eval_log.txt")
    save_json = os.path.join(LOG_DIR, f"{ALGORITHM}_{model_name}_{timestamp}_func_eval_results.json")
    save_fail_json = os.path.join(LOG_DIR, f"{ALGORITHM}_{model_name}_{timestamp}_func_eval_failed.json")

    model_tee = Tee(save_txt)
    old_stdout = sys.stdout
    sys.stdout = model_tee

    try:
        print("\n" + "=" * 100)
        print(f"[*] Algorithm folder: {ALGORITHM}")
        print(f"[*] Model name: {model_name}")
        print(f"[*] Source file: {py_path}")
        print(f"[*] Found {len(func_names)} heuristic functions:")
        for fn in func_names:
            print(f"    - {fn}")

        print(f"[*] Found {len(taillard_files)} Taillard files:")
        for fp in taillard_file_names:
            print(f"    - {fp}")

        print(f"[*] Workers: {WORKERS}")
        print(f"[*] Time limit per block: {TIME_LIMIT}")
        print(f"[*] Iter max per block: {ITER_MAX}")
        print("=" * 100)

        tasks = []
        for fn in func_names:
            for tf in taillard_files:
                tasks.append((py_path, fn, tf, TIME_LIMIT, ITER_MAX))

        results = []
        failed = []

        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            future_to_task = {
                executor.submit(
                    evaluate_one_function_one_file,
                    py_path,
                    fn,
                    tf,
                    time_limit,
                    iter_max,
                ): (fn, tf)
                for py_path, fn, tf, time_limit, iter_max in tasks
            }

            total = len(future_to_task)
            done_count = 0
            print("[*] Parallel evaluation started...")

            for future in as_completed(future_to_task):
                fn, tf = future_to_task[future]
                done_count += 1
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[*] Progress {done_count}/{total} done: {fn} | {os.path.basename(tf)}")
                except Exception as e:
                    err_msg = traceback.format_exc()
                    failed.append({
                        "model_name": model_name,
                        "function_name": fn,
                        "taillard_file": os.path.basename(tf),
                        "error": str(e),
                        "traceback": err_msg,
                    })
                    print(f"[FAIL] Progress {done_count}/{total}: {fn} | {os.path.basename(tf)}\n{e}")

        results.sort(
            key=lambda r: (
                heuristic_sort_key(r["function_name"]),
                nm_from_name(Path(r["taillard_file"]))
            )
        )

        function_summaries = aggregate_function_results(results)
        across_file_summary = aggregate_across_heuristics_per_file(results)
        
        
        global_overall = aggregate_global_overall(results)
        print_per_function_report(function_summaries)
        print_across_heuristics_report(across_file_summary)
        print_compact_table(function_summaries, taillard_file_names)
        print_global_overall_report(global_overall)
        
        
        payload = {
            "algorithm": ALGORITHM,
            "model_name": model_name,
            "source_py": py_path,
            "time_limit": TIME_LIMIT,
            "iter_max": ITER_MAX,
            "workers": WORKERS,
            "taillard_files": taillard_file_names,
            "heuristic_functions": func_names,
            "raw_results": results,
            "function_summaries": function_summaries,
            "across_heuristics_per_file": across_file_summary,
            "failed": failed,
            "global_overall": global_overall,
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

        print(f"[*] Text log saved to: {save_txt}")

        return {
            "model_name": model_name,
            "source_py": py_path,
            "function_summaries": function_summaries,
            "across_heuristics_per_file": across_file_summary,
            "failed": failed,
            "save_txt": save_txt,
            "save_json": save_json,
            "save_fail_json": save_fail_json if failed else None,
            "global_overall": global_overall,
        }

    finally:
        sys.stdout = old_stdout
        model_tee.close()


# =========================================================
# Main
# =========================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    algo_dir = os.path.join(SAVED_CODES_DIR, ALGORITHM)
    if not os.path.isdir(algo_dir):
        raise NotADirectoryError(f"Algorithm folder not found: {algo_dir}")

    if not os.path.isdir(TESTING_DATA_DIR):
        raise NotADirectoryError(f"TestingData dir not found: {TESTING_DATA_DIR}")

    taillard_files = sorted(
        [str(p) for p in Path(TESTING_DATA_DIR).glob("t_j*_m*") if p.is_file()],
        key=lambda x: nm_from_name(Path(x))
    )
    if len(taillard_files) == 0:
        raise FileNotFoundError(f"No Taillard files found in {TESTING_DATA_DIR}")

    global_log_file = os.path.join(LOG_DIR, f"{ALGORITHM}_{timestamp}_multi_model_eval_log.txt")
    global_tee = Tee(global_log_file)
    old_stdout = sys.stdout
    sys.stdout = global_tee

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    all_model_summaries = []

    try:
        print(f"[*] Algorithm: {ALGORITHM}")
        print(f"[*] Models to evaluate: {MODEL_NAMES}")
        print(f"[*] Workers per model: {WORKERS}")
        print(f"[*] Time limit per block: {TIME_LIMIT}")
        print(f"[*] Iter max per block: {ITER_MAX}")
        print(f"[*] TestingData dir: {TESTING_DATA_DIR}")
        print(f"[*] Saved codes dir: {algo_dir}")
        print(f"[*] Global log file: {global_log_file}")

        # 不同 model 串行测试
        for model_name in MODEL_NAMES:
            print("\n" + "#" * 100)
            print(f"[*] Start evaluating model: {model_name}")
            print("#" * 100)

            summary = evaluate_one_model(model_name, timestamp, taillard_files)
            if summary is not None:
                all_model_summaries.append(summary)

        print("\n" + "#" * 100)
        print("[*] All model evaluations finished.")
        print("#" * 100)

        for summary in all_model_summaries:
            print(f"\n----- {summary['model_name']} -----")
            print(f"source_py: {summary['source_py']}")
            print(f"log_txt:   {summary['save_txt']}")
            print(f"json:      {summary['save_json']}")
            if summary["save_fail_json"] is not None:
                print(f"failed:    {summary['save_fail_json']}")

    finally:
        sys.stdout = old_stdout
        global_tee.close()


if __name__ == "__main__":
    main()