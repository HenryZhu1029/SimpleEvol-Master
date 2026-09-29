# problems/fssp_gls/gen_inst.py

import random
from pathlib import Path


def generate_datasets(
    n_jobs=50,
    num_instances=64,
    seed=2024,
):
    """
    Generate FSSP instances (EOH-compatible format).
    """
    random.seed(seed)
    basepath = Path(__file__).resolve().parent
    out_dir = basepath / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for inst in range(num_instances):
        m = random.randint(2, 20)
        processing_times = [
            [random.randint(1, 100) for _ in range(m)]
            for _ in range(n_jobs)
        ]

        file_path = out_dir / f"{inst + 1}.txt"
        with open(file_path, "w") as f:
            f.write(f"{n_jobs} {m}\n")
            for i in range(n_jobs):
                for j in range(m):
                    f.write(f"{j} {processing_times[i][j]} ")
                f.write("\n")
                
if __name__ == "__main__":
    generate_datasets()