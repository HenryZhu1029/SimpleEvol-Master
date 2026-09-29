import hydra
import logging 
import os
from pathlib import Path
import subprocess
import warnings
from datetime import datetime
from utils.utils import init_client
from importlib import import_module

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)

    if cfg.algorithm == "simple_evol":
        from simple_evol import SimpleEvol as LHH
    elif  cfg.algorithm == "funsearch":
        from baseline.funsearch.funsearch import FunSearch as LHH
    else:
        raise NotImplementedError

    # Main algorithm
    lhh = LHH(cfg, ROOT_DIR, client)
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    # logging.info(f"Best Code Path Overall: {best_code_path_overall}")
    
    # save the best candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_cand_filename = f"best_cand_{timestamp}.py"
    # Run validation and redirect stdout to a file "best_code_overall_stdout.txt"
    save_dir = Path(ROOT_DIR) / "problems" / cfg.problem.problem_name / "best_candidates"
    save_dir.mkdir(exist_ok=True)

    best_cand_path = save_dir / best_cand_filename
    with open(best_cand_path, "w", encoding="utf-8") as file:
        prefix = (
            "import numpy as np\n"
            "import math\n"
            "import random\n"
            "import torch\n"
            "import scipy\n"
        )
        file.write(prefix+ best_code_overall + "\n")

    # Dynamically load evaluator
    if cfg.algorithm == "simple_evol":
        eval_module = import_module(f"problems.{cfg.problem.problem_name}.eval")
    elif cfg.algorithm == "funsearch":
        eval_module = import_module(f"baseline.funsearch.problems.{cfg.problem.problem_name}.eval")
    else:
        raise NotImplementedError("please create a valid evaluator first!")
    
    eval_tool_cls = None
    for name in dir(eval_module):
        obj = getattr(eval_module, name)
        if isinstance(obj, type) and name.endswith("EvalTool"):
            eval_tool_cls = obj
            break

    if eval_tool_cls is None:
        raise ValueError(
            f"Cannot find evaluator class ending with 'EvalTool' in problems/{cfg.problem.problem_name}/eval.py or(in ./evalfc.py)"
        )

    logging.info("Running final test evaluation...")

    # Let eval.py decide what 'test' means
    evaluator = eval_tool_cls(
        mode="test",
        problem_size=cfg.problem.problem_size,
        n_instances=cfg.problem.test_n_instances,
        timeout=(cfg.timeout if cfg.algorithm == "simple_evol"
                 and cfg.problem.problem_name == "fssp_gls" else 3600),
    )
    test_result = evaluator.evaluate(str(best_cand_path))

    logging.info(
        f"Final test result: obj={test_result.obj}, time={test_result.time}, error={test_result.error}"
    )
    logging.info(f"Final test details: {test_result.details}")

    if hasattr(client, "get_usage_summary"):
        usage = client.get_usage_summary()
        logging.info(
            f"[TOTAL LLM USAGE] "
            f"calls={usage['total_calls']}, "
            f"prompt={usage['prompt_tokens']}, "
            f"completion={usage['completion_tokens']}, "
            f"total={usage['total_tokens']}"
        )


if __name__ == "__main__":
    main()
