from __future__ import annotations

import logging
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from utils.utils import extract_candidate_from_response, repair_python_function_code, save_experiment_record

from . import code_manipulation
from .funsearch_utils import (
    build_scores_from_result,
    calls_ancestor,
    extract_function_names_from_spec,
    load_text_if_exists,
    make_default_seed_function,
    make_default_template,
    normalize_generated_function,
    sample_to_program,
)
from .programs_database import ProgramsDatabase
from .prompt_builder import build_chat_messages

logger = logging.getLogger(__name__)


class FunSearch:
    def __init__(self, cfg, root_dir, client):
        self.cfg = cfg
        self.root_dir = Path(root_dir)
        self.client = client

        self.problem_name = cfg.problem.problem_name
        self.problem_size = cfg.problem.problem_size
        self.n_instances = cfg.problem.n_instances
        self.test_n_instances = cfg.problem.test_n_instances
        self.func_name = cfg.problem.func_name
        self.obj_type = cfg.problem.obj_type
        self.problem_type = cfg.problem.problem_type
        self.time_out = cfg.timeout

        algo_cfg = cfg.get("baseline",cfg)
        self.max_experiments = int(algo_cfg.get("max_experiments", cfg.get("max_experiments", 50)))
        self.samples_per_prompt = int(algo_cfg.get("samples_per_prompt", 1))
        self.num_islands = int(algo_cfg.get("num_islands", 8))
        self.functions_per_prompt = int(algo_cfg.get("functions_per_prompt", 2))
        self.reset_period = int(algo_cfg.get("reset_period", 20))
        self.keep_ratio = float(algo_cfg.get("keep_ratio", 0.5))
        self.cluster_sampling_temperature_init = float(algo_cfg.get("cluster_sampling_temperature_init", 0.1))
        self.cluster_sampling_temperature_period = int(algo_cfg.get("cluster_sampling_temperature_period", 30))
        self.use_signature_clustering = bool(algo_cfg.get("use_signature_clustering", True))
        self.enable_test_eval = bool(algo_cfg.get("enable_test_eval", cfg.get("enable_test_eval", False)))

        self.baseline_dir = self.root_dir / "baseline" / "funsearch"
        self.problem_dir = self.baseline_dir / "problems" / self.problem_name
        self.candidates_dir = self.problem_dir / "llm_temp"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path.cwd()
        self.exp_records_dir = self.run_dir / "experiment_records"
        self.exp_records_dir.mkdir(parents=True, exist_ok=True)

        self.eval_tool_cls = self._load_problem_evaluator()
        self.eval_tool = self.eval_tool_cls(
            problem_size=self.problem_size,
            n_instances=self.n_instances,
            data_path=self.problem_dir / "data",
            timeout=self.time_out,
        )
        self.test_eval_tool = None
        if self.enable_test_eval:
            self.test_eval_tool = self.eval_tool_cls(
                problem_size=self.problem_size,
                n_instances=self.test_n_instances,
                data_path=self.problem_dir / "data",
                mode="test",
                timeout=self.time_out,
            )

        self.task_desc = self._load_problem_prompt()
        self.template_text = self._load_template_text()
        evolve_name, _ = extract_function_names_from_spec(self.template_text)
        self.function_to_evolve = evolve_name or self.func_name
        self.template_program = code_manipulation.text_to_program(self.template_text)
        self.seed_text = self._load_seed_text()
        self.seed_function, self.seed_program_text = sample_to_program(
            self.seed_text,
            version_generated=None,
            template=self.template_program,
            function_to_evolve=self.function_to_evolve,
        )

        self.database = ProgramsDatabase(
            template=self.template_program,
            function_to_evolve=self.function_to_evolve,
            num_islands=self.num_islands,
            functions_per_prompt=self.functions_per_prompt,
            cluster_sampling_temperature_init=self.cluster_sampling_temperature_init,
            cluster_sampling_temperature_period=self.cluster_sampling_temperature_period,
            obj_type=self.obj_type,
            use_signature_clustering=self.use_signature_clustering,
            seed=int(cfg.get("seed", 0)),
        )

        self.experiment_results: list[dict[str, Any]] = []
        self.best_result: dict[str, Any] | None = None
        self.experiment_count = 0
        self._init_seed_database()

    def _load_problem_evaluator(self):
        module = import_module(f"baseline.funsearch.problems.{self.problem_name}.eval")
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and name.endswith("EvalTool"):
                return obj
        raise ValueError(f"Cannot find evaluator class in problems/{self.problem_name}/eval.py")

    def _load_problem_prompt(self):
        module = import_module(f"baseline.funsearch.problems.{self.problem_name}.prompt")

        parts = []
        for key in ["TASK_DESC", "FUNCTION_DESCRIPTION", "FUNCTION_SIGNATURE", "PROMPT"]:
            val = getattr(module, key, "")
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())

        task_desc = "\n\n".join(parts)
        if not task_desc:
            raise ValueError(
                f"Cannot find prompt text in baseline/funsearch/problems/{self.problem_name}/prompt.py"
            )
        return task_desc

    def _load_template_text(self) -> str:
        template_path = self.problem_dir / "template.py"
        text = load_text_if_exists(template_path)
        if text:
            return text
        return make_default_template(self.func_name)

    def _load_seed_text(self) -> str:
        seed_path = self.problem_dir / "seed.py"
        text = load_text_if_exists(seed_path)
        if text:
            return text
        return make_default_seed_function(self.function_to_evolve)

    def _call_llm(self, messages: list[dict], n: int) -> list[str]:
        texts: list[str] = []

        for _ in range(n):
            resp = self.client.chat_completion(n=1, messages=messages)[0]

            content = resp.message.content or ""
            parsed = extract_candidate_from_response(content)
            code = parsed.get("code") or content
            code = normalize_generated_function(code)

            if code:
                code = repair_python_function_code(code)

            texts.append(code or "")

        return texts

    def _evaluate_program(self, program_text: str, code_path: Path) -> tuple[dict, Any]:
        code_path.write_text(program_text, encoding="utf-8")
        result = self.eval_tool.evaluate(str(code_path))
        return result.to_dict(), result

    def _init_seed_database(self):
        seed_path = self.candidates_dir / "seed_funsearch.py"
        result_dict, _ = self._evaluate_program(self.seed_program_text, seed_path)
        if result_dict.get("obj") is None:
            raise ValueError(f"Seed program is invalid: {result_dict.get('error')}")
        scores_per_test = build_scores_from_result(result_dict)
        for island_id in range(self.num_islands):
            self.database.register_program(
                self.seed_function,
                island_id=island_id,
                scores_per_test=scores_per_test,
                source_path=str(seed_path),
                raw_code=self.seed_program_text,
                experiment_id=0,
            )
        self.best_result = {
            "experiment": 0,
            "code": self.seed_program_text,
            "code_path": str(seed_path),
            "obj": result_dict.get("obj"),
            "time": result_dict.get("time"),
            "error": result_dict.get("error"),
            "description": "seed",
        }

    def _update_best(self, record: dict):
        if record.get("obj") is None:
            return
        if self.best_result is None:
            self.best_result = record
            return
        if self.obj_type == "min":
            if record["obj"] < self.best_result["obj"]:
                self.best_result = record
        else:
            if record["obj"] > self.best_result["obj"]:
                self.best_result = record

    def evolve(self):
        logger.info("=" * 70)
        logger.info("FunSearch started.")
        logger.info(f"Problem: {self.problem_name}")
        logger.info(f"Problem type: {self.problem_type}")
        logger.info(f"Problem size: {self.problem_size}")
        logger.info(f"Train instances: {self.n_instances}")
        logger.info(f"Test instances: {self.test_n_instances}")
        logger.info(f"Objective type: {self.obj_type}")
        logger.info(f"Function to evolve: {self.function_to_evolve}")
        logger.info(f"Timeout: {self.time_out}")
        logger.info(f"Max experiments: {self.max_experiments}")
        logger.info(f"Samples per prompt: {self.samples_per_prompt}")
        logger.info(f"Num islands: {self.num_islands}")
        logger.info(f"Functions per prompt: {self.functions_per_prompt}")
        logger.info(f"Reset period: {self.reset_period}")
        logger.info(f"Keep ratio: {self.keep_ratio}")
        logger.info(f"Cluster sampling temperature init: {self.cluster_sampling_temperature_init}")
        logger.info(f"Cluster sampling temperature period: {self.cluster_sampling_temperature_period}")
        logger.info(f"Use signature clustering: {self.use_signature_clustering}")
        logger.info(f"Enable test eval: {self.enable_test_eval}")
        logger.info(f"Problem dir: {self.problem_dir}")
        logger.info(f"Candidates dir: {self.candidates_dir}")
        logger.info("=" * 70)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        
        while self.experiment_count < self.max_experiments:
            prompt = self.database.get_prompt()
            messages = build_chat_messages(
                task_desc=self.task_desc,
                prompt_code=prompt.code,
                function_to_evolve=self.function_to_evolve,
            )
            candidates = self._call_llm(messages, n=self.samples_per_prompt)

            for candidate_code in candidates:
                if self.experiment_count >= self.max_experiments:
                    break
                if not candidate_code:
                    continue

                self.experiment_count += 1

                code_path = self.candidates_dir / f"funsearch_{timestamp}.py"

                try:
                    new_function, program_text = sample_to_program(
                        candidate_code,
                        version_generated=prompt.version_generated,
                        template=self.template_program,
                        function_to_evolve=self.function_to_evolve,
                    )
                except Exception as e:
                    logger.warning(f"Failed to assemble candidate: {e}")
                    continue

                if calls_ancestor(program_text, self.function_to_evolve):
                    logger.info("Reject candidate because it calls ancestor versions.")
                    continue

                result_dict, _ = self._evaluate_program(program_text, code_path)
                train_obj = result_dict.get("obj")
                exec_time = result_dict.get("time")
                error_msg = result_dict.get("error")
                test_obj = None
                if self.enable_test_eval and self.test_eval_tool is not None and train_obj is not None:
                    try:
                        test_result = self.test_eval_tool.evaluate(str(code_path))
                        test_obj = getattr(test_result, "obj", None)
                    except Exception as e:
                        logger.warning(f"Silent test evaluation failed: {e}")

                save_experiment_record(
                    experiment_id=self.experiment_count,
                    code=program_text,
                    code_path=str(code_path),
                    train_obj=train_obj,
                    exec_time=exec_time,
                    error_msg=error_msg,
                    problem_name=self.problem_name,
                    problem_size=self.problem_size,
                    exp_records_dir=self.exp_records_dir,
                    test_obj=test_obj,
                    description=f"island={prompt.island_id}, version={prompt.version_generated}",
                )
                record = {
                    "experiment": self.experiment_count,
                    "code": program_text,
                    "code_path": str(code_path),
                    "obj": train_obj,
                    "time": exec_time,
                    "error": error_msg,
                    "description": f"island={prompt.island_id}, version={prompt.version_generated}",
                }
                time_str = f"{exec_time:.4f}" if exec_time is not None else "None"

                logger.info(
                    f"[EVAL] exp={record['experiment']} "
                    f"island={prompt.island_id} "
                    f"obj={train_obj} "
                    f"time={time_str} "
                    f"error={error_msg}"
                )
                self.experiment_results.append(record)
                self._update_best(record)

                if train_obj is not None:
                    scores_per_test = build_scores_from_result(result_dict)
                    self.database.register_program(
                        new_function,
                        island_id=prompt.island_id,
                        scores_per_test=scores_per_test,
                        source_path=str(code_path),
                        raw_code=program_text,
                        experiment_id=self.experiment_count,
                    )

                if self.reset_period > 0 and self.experiment_count % self.reset_period == 0:
                    logger.info("Resetting weak islands...")
                    self.database.reset_islands(keep_ratio=self.keep_ratio)

        if self.best_result is None:
            raise RuntimeError("FunSearch did not produce any valid candidate.")
        return self.best_result["code"], self.best_result["code_path"]
