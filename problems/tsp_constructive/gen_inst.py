

import numpy as np
from pathlib import Path


def generate_datasets(mode: str = "train", data_path=None, sizes=None,
                      n_instances: int = 64, seed: int = 1234):
    root = Path(__file__).resolve().parent
    tsp_eval_dir = Path(data_path) if data_path is not None else root / "data"
    tsp_eval_dir.mkdir(parents=True, exist_ok=True)

    default_sizes = {"train": [50], "val": [20, 50, 100, 200],
                     "test": [50, 100, 200]}
    mode_ids = {"train": 0, "val": 1, "test": 2}
    if mode not in default_sizes:
        raise ValueError(f"Unknown mode: {mode}")
    if n_instances <= 0:
        raise ValueError("n_instances must be positive")
    sizes = default_sizes[mode] if sizes is None else sizes

    for problem_size in sizes:
        out_path = tsp_eval_dir / f"{mode}{problem_size}_dataset.npy"
        if out_path.exists():
            continue  # Preserve existing experiment datasets.
        # Separate reproducible streams for each split and problem size.
        rng = np.random.default_rng(np.random.SeedSequence([seed, mode_ids[mode], problem_size]))
        arr = rng.random((n_instances, problem_size, 2))
        np.save(out_path, arr)

    print("Datasets generated successfully.")
    print(" - NPY saved under:", tsp_eval_dir)


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        generate_datasets(mode=split)
