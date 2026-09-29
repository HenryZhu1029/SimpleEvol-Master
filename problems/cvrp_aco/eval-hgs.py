import os
import sys
import json
import time
import logging
from datetime import datetime

import numpy as np
import hygese as hgs


# =========================================================
# Config
# =========================================================
PROBLEM_SIZES = [50, 100, 200]
TIME_LIMIT = 60          # per instance, seconds
NB_GRANULAR = 20
MU = 25
LAMBDA = 40
NB_ELITE = 4
NB_ITER = 25000
TARGET_FEASIBLE = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "hgs_logs")
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


def build_hgs_solver():
    ap = hgs.AlgorithmParameters(
        timeLimit=TIME_LIMIT,
        nbGranular=NB_GRANULAR,
        mu=MU,
        lambda_=LAMBDA,
        nbElite=NB_ELITE,
        nbIter=NB_ITER,
        targetFeasible=TARGET_FEASIBLE,
    )
    return hgs.Solver(parameters=ap, verbose=False)


def solve_one_instance_with_hgs(instance: np.ndarray):
    demands = instance[:, 0].astype(int)
    coords = instance[:, 1:].astype(float) * 1000.0

    x = coords[:, 0]
    y = coords[:, 1]

    vehicle_capacity = 50
    num_vehicles = int(np.ceil(np.sum(demands[1:]) / vehicle_capacity))

    data_dict = {
        "x_coordinates": x,
        "y_coordinates": y,
        "demands": demands,
        "vehicle_capacity": vehicle_capacity,
        "num_vehicles": max(1, num_vehicles),
        "service_times": np.zeros(len(demands)),
        "depot": 0,
    }

    solver = build_hgs_solver()
    result = solver.solve_cvrp(data_dict)
    return float(result.cost)/1000, result.routes


def evaluate_one_size(problem_size: int):
    dataset_path = os.path.join(DATA_DIR, f"test{problem_size}_dataset.npy")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = np.load(dataset_path)
    n_instances = dataset.shape[0]

    print("\n" + "=" * 60)
    print(f"[*] Evaluating size={problem_size} | instances={n_instances}")

    costs = []
    failed = []

    t0 = time.time()

    for idx, instance in enumerate(dataset):
        try:
            cost, _ = solve_one_instance_with_hgs(instance)
            costs.append(cost)
        except Exception as e:
            failed.append(idx)

    total_elapsed = time.time() - t0

    avg_cost = float(np.mean(costs)) if costs else float("inf")
    std_cost = float(np.std(costs)) if costs else float("nan")

    print(f"[*] size={problem_size} | avg={avg_cost:.6f} | std={std_cost:.6f} | time={total_elapsed:.2f}s")

    return {
        "problem_size": problem_size,
        "n_instances": int(n_instances),
        "n_success": len(costs),
        "n_failed": len(failed),
        "average_cost": avg_cost,
        "std_cost": std_cost,
        "all_costs": costs,
        "elapsed_sec": total_elapsed,
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"hgs_eval_{timestamp}.txt")
    json_file = os.path.join(LOG_DIR, f"hgs_eval_{timestamp}.json")

    original_stdout = sys.stdout
    tee = Tee(log_file)
    sys.stdout = tee

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    total_start = time.time()
    all_results = []

    try:
        print(f"[*] Data dir: {DATA_DIR}")
        print(f"[*] Problem sizes: {PROBLEM_SIZES}")
        print(f"[*] Log file: {log_file}")

        for size in PROBLEM_SIZES:
            result = evaluate_one_size(size)
            all_results.append(result)

        summary_table = []
        print("\n" + "#" * 60)
        print("[*] Summary (HGS Reference)")
        print("#" * 60)

        print("size\tavg_cost\tstd")
        for r in all_results:
            print(
                f"{r['problem_size']}\t"
                f"{r['average_cost']:.6f}\t"
                f"{r['std_cost']:.6f}"
            )
            summary_table.append({
                "size": r["problem_size"],
                "average_cost": r["average_cost"],
                "std_cost": r["std_cost"],
                "n_success": r["n_success"],
                "n_failed": r["n_failed"],
                "elapsed_sec": r["elapsed_sec"],
            })

        total_elapsed = time.time() - total_start
        print(f"\n[*] Total elapsed: {total_elapsed:.2f}s")

        payload = {
            "timestamp": timestamp,
            "config": {
                "problem_sizes": PROBLEM_SIZES,
                "time_limit": TIME_LIMIT,
                "nb_granular": NB_GRANULAR,
                "mu": MU,
                "lambda": LAMBDA,
                "nb_elite": NB_ELITE,
                "nb_iter": NB_ITER,
                "target_feasible": TARGET_FEASIBLE,
            },
            "summary": summary_table,
            "results": all_results,
            "total_elapsed_sec": total_elapsed,
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[*] JSON results saved to: {json_file}")

    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()