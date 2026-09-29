import json
import logging
import random
from pathlib import Path
from importlib import import_module
import time
from datetime import datetime
from utils.utils import *
from utils.common_prompts import PROMPT_REGISTRY
import re

logger = logging.getLogger(__name__)


class SimpleEvol:
    def __init__(self, cfg, root_dir, client):
        self.cfg = cfg
        self.root_dir = Path(root_dir)
        self.client = client

        # ========= Generic config =========
        self.problem_name = cfg.problem.problem_name
        self.problem_size = cfg.problem.problem_size
        self.n_instances = cfg.problem.n_instances
        self.test_n_instances = cfg.problem.test_n_instances
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type
        
        self.max_experiments = cfg.max_experiments
        self.compress_every = cfg.compress_every
        self.enable_test_eval = cfg.enable_test_eval
        self.time_out = cfg.timeout
        
        # ========= Paths =========
        self.problem_dir = self.root_dir / "problems" / self.problem_name
        self.test_data_path = self.problem_dir / "data"
        
        
        self.candidates_dir = self.problem_dir / "llm_temp"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_dir = Path.cwd()
        self.exp_records_dir = self.run_dir / "experiment_records"
        self.exp_records_dir.mkdir(parents=True, exist_ok=True)
        # ========= Problem-specific evaluator =========
        self.eval_tool_cls = self._load_problem_evaluator()

        self.eval_tool = self.eval_tool_cls(
            problem_size=self.problem_size,
            n_instances=self.n_instances,
            data_path = self.test_data_path,
            timeout = self.time_out
        )

        self.test_eval_tool = None
        if self.enable_test_eval:
            self.test_eval_tool = self.eval_tool_cls(
                problem_size=self.problem_size,
                n_instances=self.test_n_instances,
                data_path=str(self.test_data_path),
                mode="test",
                timeout = self.time_out
            )

        # ========= Prompts =========
        self.system_generator_prompt = PROMPT_REGISTRY.get("system_generator_prompt", "")
        self.summary_user_prompt = PROMPT_REGISTRY.get("summary_user_prompt", "").strip()
        
        if not self.system_generator_prompt or not self.summary_user_prompt:
            raise ValueError("PROMPT_REGISTRY['system_generator_prompt'] or ['summary_user_prompt'] is empty or missing.")
            
        self.task_desc = self._load_problem_prompt()
        self.seed_text = self._load_problem_seed()

        # ========= Runtime states =========
        self.messages = self._build_initial_messages()
        self.experiment_results = []
        self.score_history = []

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if self.seed_text:
            self._bootstrap_seed_experiment()
    # -------------------------------------------------------------------------
    # Problem-specific dynamic loading
    # -------------------------------------------------------------------------
    def _load_problem_evaluator(self):
        """
        Load evaluator class from:
            problems/<problem_name>/eval.py

        Expected:
            - evaluator class name ends with 'EvalTool'
        """
        module = import_module(f"problems.{self.problem_name}.eval")

        eval_tool_cls = None
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and name.endswith("EvalTool"):
                eval_tool_cls = obj
                break

        if eval_tool_cls is None:
            raise ValueError(
                f"Cannot find evaluator class ending with 'EvalTool' in problems/{self.problem_name}/eval.py"
            )

        return eval_tool_cls
    def _bootstrap_seed_experiment(self):
        logger.info("=" * 70)
        logger.info("[Bootstrap] Evaluating seed as experiment 0...")
        logger.info("=" * 70)

        code, description = self._extract_code_and_description(self.seed_text)

        if not code:
            logger.warning("SEED exists but no valid Python code block was extracted.")
            return

        if not description:
            description = "Seed heuristic description unavailable."

        code_path = self.candidates_dir / f"seed_{self.timestamp}.py"
        code_path.write_text(code, encoding="utf-8")

        eval_result = self.eval_tool.evaluate(str(code_path))
        result_dict = eval_result.to_dict()

        train_obj = result_dict.get("obj")
        exec_time = result_dict.get("time")
        error_msg = result_dict.get("error")

        test_obj = None
        if (
            self.enable_test_eval
            and self.test_eval_tool is not None
            and train_obj is not None
        ):
            try:
                test_result = self.test_eval_tool.evaluate(str(code_path))
                test_obj = getattr(test_result, "obj", None)
            except Exception as e:
                logger.warning(f"Seed silent test evaluation failed: {e}")

        filtered_code = filter_code(code)

        self.score_history.append(
            {
                "experiment": 0,
                "train_obj": train_obj,
                "test_obj": test_obj,
            }
        )

        save_experiment_record(
            experiment_id=0,
            code=code,
            code_path=str(code_path),
            train_obj=train_obj,
            exec_time=exec_time,
            error_msg=error_msg,
            problem_name=self.problem_name,
            problem_size=self.problem_size,
            exp_records_dir=self.exp_records_dir,
            test_obj=test_obj,
            description=description,
        )

        self.experiment_results.append(
            {
                "experiment": 0,
                "code": code,
                "filtered_code": filtered_code,
                "description": description,
                "code_path": str(code_path),
                "obj": train_obj,
                "time": exec_time,
                "error": error_msg,
            }
        )

        # 只把 description 和 eval feedback 放进消息，不放 seed code
        self.messages.append(
            {
                "role": "assistant",
                "content": f"[Seed Description | Experiment 0]\n{description}",
            }
        )

        self.messages.append(
            build_eval_feedback_message(
                experiment_id=0,
                result_dict=result_dict,
                test_obj=test_obj,
            )
        )

        logger.info("[Bootstrap] Seed experiment 0 completed.")
        logger.info(f"  obj: {train_obj}")
        logger.info(f"  time: {exec_time}")
        logger.info(f"  error: {error_msg}")
    # -------------------------------------------------------------------------
    # Prompt / message construction
    # -------------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        
        task_prompt = self.task_desc.strip()
        base_prompt = self.system_generator_prompt.format(
            max_experiments=self.max_experiments
        ).strip()

        format_suffix = """

        Important output rule:
        Your response must contain exactly two parts in the following order:
        1. A single Python code block formatted as:
        ```python
        # your code here 
        ```
        
        2. After designing the code, write a concise heuristic description after the code block (you may wrap it using <description> ... </description>).
        """.strip()
        return f"{base_prompt}\n\n{format_suffix}"


    def _load_problem_prompt(self):
        module = import_module(f"problems.{self.problem_name}.prompt")
        task_desc = getattr(module, "TASK_DESC", "")
        if not task_desc:
            raise ValueError(
                f"Cannot find TASK_DESC in problems/{self.problem_name}/prompt.py"
            )
        return task_desc
    def _load_problem_seed(self):
        module = import_module(f"problems.{self.problem_name}.prompt")
        seed_text = getattr(module, "SEED", "")
        return seed_text.strip() if isinstance(seed_text, str) else ""
    
    
    def _build_initial_messages(self):
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"{self.task_desc.strip()}\n\n"
                    f"Please start your {self.problem_name} heuristic exploration experiment. "
                    "Begin with a simple baseline method, then try different improvement strategies. "
                    "Output exactly one candidate heuristic using the following format:\n"
                    "```python\n"
                    "# your code here\n"
                    "```\n"
                    "Then provide a concise heuristic description after the code block (you may wrap it using <description> ... </description>). Keep within 50 words.\n"
                    "##Description requirements:\n"
                    "-Briefly summarize what structure/scoring rule is used.\n"
                    "-State the key mechanism and logic.\n"
                    "-Mention important parameters / thresholds if any.\n"
                    "-Keep this description informative and structured.\n"
                    "Note: Write all necessary imports inside functions (not at the top level of the file)."
                    "Do not include any natural-language explanation inside the code block."
                    "Do not write descriptions/docstrings/comments in natural language inside code."
                ),
            },
        ]
    def _build_retry_message_for_bad_format(self):
        return {
            "role": "user",
            "content": (
                "Your previous response did not follow the required format.\n"
                "Please output exactly:\n"
                "1. One Python code block formatted as:\n"
                "```python\n"
                "# your code here\n"
                "```\n"
                "2. A concise heuristic description after the code block.\n"
                "Do not omit the Python code block. Do not write/docstrings descriptions inside code."
            ),
        }
    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------
    def get_best_result(self,experiment_results):
        valid_results = [r for r in experiment_results if r.get("obj") is not None]
        if not valid_results:
            return None
        if self.obj_type == "min":
            return min(valid_results, key=lambda x: x["obj"])
        else:
            return max(valid_results, key=lambda x: x["obj"])


    def _call_llm_once(self, messages):
        """
        ReEvo-style normal chat completion, no tool calling.
        Assumes client.chat_completion(...) returns a list-like choices object.
        """
        responses = self.client.chat_completion(
            n=1,
            messages=messages,
        )
        return responses[0].message
    
    # -------------------------------------------------------------------------
    # Context compression
    # -------------------------------------------------------------------------
    def compress_context(self):
        
        def _strip_code_blocks(text):
            return re.sub(r"```[\s\S]*?```", "", text)
        logger.info("#" * 70)
        logger.info(f"[Context Compression] {self.experiment_count} experiments completed, generating intermediate summary...")
        logger.info("#" * 70)

        summary_request = {
            "role": "user",
            "content": self.summary_user_prompt.format(experiment_count=self.experiment_count),
        }

        summary_messages = list(self.messages) + [summary_request]
        summary_message = self._call_llm_once(summary_messages)
        summary_content = summary_message.content or ""
        summary_content = _strip_code_blocks(summary_content)
        
        logger.info("[Intermediate Summary]")
        logger.info("-" * 40)
        logger.info(summary_content)

        initial_messages = self.messages[:2]
        remaining_experiments = self.max_experiments - self.experiment_count
        best_result = self.get_best_result(self.experiment_results)

        # An lightweight prompting cues to encourage exploration.
        # It is intended only to diversify the agent's behaviour, not to replace search actions.
        if best_result is not None:
            prompt_strategy = random.choice(
                [
                    "Based on your findings, continue refining the current best strategy.",
                    "Focus on improving the strongest existing design.",
                    "Fine-tuning the current elite design with different parameters and thresholds settings.",
                    "Based on your findings, try new improvement strategies.",
                    "Modify the current design with different mechanisms.",
                    "Test alternative refinements around the current promising direction.",
                    "Try a radically different heuristic family unrelated to the current best.",
                    "Based on your findings, explore fundamentally different heuristic principles that are completely different from existing strategies.",
                    "Experiment with hybrid combinations of different heuristic strategies.",
                ]
            )
            best_code_section = (
                f"Current best/elite code core logic (Experiment {best_result['experiment']}, "
                f"obj={best_result['obj']:.6f}):\n"
                f"```python\n{best_result['filtered_code']}\n```"
                f"best/elite code description:{best_result.get('description', 'No description available.')}"
            )
        else:
            prompt_strategy = "No successful experiment yet. Focus on obtaining a valid working heuristic first."
            best_code_section = "No successful experiment yet."


        # best_code_section = ""
        compressed_messages = initial_messages + [
            {
                "role": "assistant",
                "content": (
                    f"[Intermediate Summary - {self.experiment_count}/{self.max_experiments} "
                    f"experiments completed]\n\n{summary_content}"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Based on your intermediate summary, please continue exploring.\n\n"
                    f"Experiment progress: {self.experiment_count}/{self.max_experiments} completed, "
                    f"{remaining_experiments} remaining.\n\n"
                    f"{best_code_section}\n\n"
                    f"Prompt strategy: {prompt_strategy}\n\n"
                    "Output the next candidate using exactly the required format: "
                    "one Python code block followed by a heuristic description."
                ),
            },
        ]

        logger.info(f"[Context compressed] messages: Length {len(self.messages)} -> length {len(compressed_messages)}")
        logger.info("#" * 70)

        self.messages = compressed_messages
        
    def _extract_description_from_response(self, response_text: str) -> str:

    
        if not response_text:
            return ""

        # 去掉 ```python ... ``` 代码块
        description = re.sub(r"```python\s*[\s\S]*?```", "", response_text, flags=re.IGNORECASE).strip()

        # 压缩多余空行
        description = re.sub(r"\n{3,}", "\n\n", description).strip()

        return description
    
    def _extract_code_and_description(self, text: str):
        parsed = extract_candidate_from_response(text)
        code = parsed.get("code")
        if code:
            code = repair_python_function_code(code)

        description = self._extract_description_from_response(text)
        description = self._normalize_description(description)

        return code, description
    
    def _normalize_description(self, text: str, max_chars: int = 2000) -> str:
        if not text:
            return ""
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n...[truncated]"
        return text
    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    def _request_final_summary(self):
        final_user_msg = {
            "role": "user",
            "content": (
                f"You have completed all {self.max_experiments} experiments. "
                "Please now provide your final summary:\n\n"
                "1. What strategies did you try?\n"
                "2. How did each strategy perform?\n"
                "3. What patterns or insights did you discover?\n"
                "4. What types of heuristic designs are more effective?\n"
                "5. What is your best solution?"
            ),
        }
        self.messages.append(final_user_msg)

        final_message = self._call_llm_once(self.messages)
        self.messages.append(
            {
                "role": "assistant",
                "content": final_message.content or "",
            }
        )

        logger.info("=" * 70)
        logger.info("[LLM Final Summary]")
        logger.info("=" * 70)
        logger.info(final_message.content if final_message.content else "")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def evolve(self):
        logger.info("=" * 70)
        logger.info("SimpleEvol started.")
        logger.info(f"Problem: {self.problem_name}")
        logger.info(f"Problem size: {self.problem_size}")
        logger.info(f"Max experiments: {self.max_experiments}")
        logger.info(f"Function Name: {self.func_name}")
        logger.info(f"Obj Type: {self.obj_type}")
        logger.info(f"Context compression: every {self.compress_every} evals" if self.compress_every > 0
                    else "Context compression: disabled")
        logger.info(f"Silent test evaluation: {'enabled' if self.enable_test_eval else 'disabled'}")

        logger.info(f"Timeout per evaluation: {self.time_out}")
        logger.info("=" * 70)

        self.experiment_count = 0
        last_compress_at = 0
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        while self.experiment_count < self.max_experiments:
            
            logger.info("=" * 70)
            logger.info(f"[Iteration {self.experiment_count}] Completed {self.experiment_count}/{self.max_experiments} evaluations")
            logger.info("=" * 70)

            assistant_message = self._call_llm_once(self.messages)
            assistant_content = assistant_message.content or ""

            # 记录 assistant 原始回复
            parsed = extract_candidate_from_response(assistant_content)
            code = parsed.get("code")
            if code:
                code = repair_python_function_code(code)

            description = self._extract_description_from_response(assistant_content)
            description = self._normalize_description(description)

            if not code:
                logger.warning("No valid Python code block extracted from model response.")
                self.messages.append(self._build_retry_message_for_bad_format())
                continue

            filtered_code = filter_code(code)

            if not description:
                description = "Heuristic description unavailable."

            self.messages.append(
                {
                    "role": "assistant",
                    "content": f"[Candidate Description]\n{description}",
                }
            )

            self.experiment_count += 1


            code_path = self.candidates_dir / f"gpt_{self.timestamp}.py"
            code_path.write_text(code, encoding="utf-8")
  
            
            

            logger.info("-" * 40)
            logger.info(
                f"[Experiment {self.experiment_count}/{self.max_experiments}] Running evaluation..."
            )


            # train eval
            eval_result = self.eval_tool.evaluate(str(code_path))
            result_dict = eval_result.to_dict()

            train_obj = result_dict.get("obj")
            exec_time = result_dict.get("time")
            error_msg = result_dict.get("error")

            logger.info("Evaluation results (train):")
            logger.info(f"  obj: {train_obj}")
            logger.info(f"  time: {exec_time}")
            logger.info(f"  error: {error_msg}")

            # silent test eval (not returned unless you want)
            test_obj = None
            if (
                self.enable_test_eval
                and self.test_eval_tool is not None
                and train_obj is not None
            ):
                try:
                    test_result = self.test_eval_tool.evaluate(str(code_path))
                    test_obj = getattr(test_result, "obj", None)
                    logger.info("Evaluation results (test, not returned to optimizer logic):")
                    logger.info(f"  obj: {test_obj}")
                except Exception as e:
                    logger.warning(f"Silent test evaluation failed: {e}")

            self.score_history.append(
                {
                    "experiment": self.experiment_count,
                    "train_obj": train_obj,
                    "test_obj": test_obj,
                }
            )
            save_experiment_record(
                experiment_id=self.experiment_count,
                code=code,
                code_path=str(code_path),
                train_obj=train_obj,
                exec_time=exec_time,
                error_msg=error_msg,
                problem_name=self.problem_name,
                problem_size=self.problem_size,
                exp_records_dir=self.exp_records_dir,
                test_obj=test_obj,
                description=description,
            )
            self.experiment_results.append(
                {
                    "experiment": self.experiment_count,
                    "code": code,
                    "filtered_code": filtered_code,
                    "description": description,
                    "code_path": str(code_path),
                    "obj": train_obj,
                    "time": exec_time,
                    "error": error_msg,
                }
            )

            # 把外部 evaluator 的结果作为普通 user feedback 喂回去
            self.messages.append(
                build_eval_feedback_message(
                    experiment_id=self.experiment_count,
                    result_dict=result_dict,
                    test_obj=test_obj,
                )
            )

            # compress context if needed
            if (
                self.compress_every > 0
                and self.experiment_count < self.max_experiments
                and self.experiment_count >= last_compress_at + self.compress_every
            ):
                self.compress_context()
                last_compress_at = self.experiment_count

        self._request_final_summary()

        # ------------------------------------------------------------------
        # Post-run logging
        # ------------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("Experimental Statistics")
        logger.info("=" * 70)

        best_result = self.get_best_result(self.experiment_results)

        if self.experiment_results:
            logger.info(f"Completed {len(self.experiment_results)} experiments.")
            for r in self.experiment_results:
                status = "✓" if r["obj"] is not None else "✗"
                obj_str = f"{r['obj']:.6f}" if r["obj"] is not None else "N/A"
                logger.info(f"[{status}] Experiment {r['experiment']}: obj={obj_str}")

            valid_results = [r for r in self.experiment_results if r["obj"] is not None]
            if valid_results:
                if self.obj_type == "min":
                    worst = max(valid_results, key=lambda x: x["obj"])
                else:
                    worst = min(valid_results, key=lambda x: x["obj"])
                logger.info(f"Best result: Exp. {best_result['experiment']}, obj={best_result['obj']:.6f}")
                logger.info(f"Worst result: Exp. {worst['experiment']}, obj={worst['obj']:.6f}")
                if worst["obj"] != 0:
                    if self.obj_type == "min":
                        improvement = ((worst["obj"] - best_result["obj"]) / worst["obj"]) * 100 
                    else:
                        improvement = -((worst["obj"] - best_result["obj"]) / worst["obj"]) * 100 
                    logger.info(f"Improvement span: {improvement:.2f}%")
        else:
            logger.info("No experiments were completed.")

        # Optional client-side token summary
        if hasattr(self.client, "get_usage_summary"):
            logger.info("=" * 70)
            logger.info("Client Usage Summary")
            logger.info("=" * 70)
            try:
                logger.info(json.dumps(self.client.get_usage_summary(), indent=2, ensure_ascii=False))
            except Exception:
                logger.info(str(self.client.get_usage_summary()))

        # logger.info("=" * 70)
        # logger.info("Training/Test Score History")
        # logger.info("=" * 70)
        # logger.info(json.dumps(self.score_history, indent=2, ensure_ascii=False))

        best_code_overall = best_result["code"] if best_result is not None else ""
        best_code_path_overall = best_result["code_path"] if best_result is not None else None
        

        return best_code_overall, best_code_path_overall
    
    
    
