import logging
import re
import inspect
import hydra
import logging
import json
from pathlib import Path
from typing import Optional

def init_client(cfg):
    global client
    if cfg.get("model", None): # for compatibility
        model: str = cfg.get("model")
        temperature: float = cfg.get("temperature", 1.0)
        if model.startswith("gpt"):
            from utils.llm_client.openai import OpenAIClient
            client = OpenAIClient(model, temperature)
        # elif cfg.model.startswith("GLM"):
        #     from utils.llm_client.zhipuai import ZhipuAIClient
        #     client = ZhipuAIClient(model, temperature)
        # else: # fall back to Llama API
        #     from utils.llm_client.llama_api import LlamaAPIClient
        #     client = LlamaAPIClient(model, temperature)
    else:
        client = hydra.utils.instantiate(cfg.llm_client)
    return client

def message_to_dict(msg):
    if isinstance(msg, dict):
        return msg

    result = {
        "role": getattr(msg, "role", ""),
        "content": getattr(msg, "content", "") or "",
    }

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        result["tool_calls"] = []
        for tc in tool_calls:
            result["tool_calls"].append({
                "id": getattr(tc, "id", ""),
                "type": getattr(tc, "type", "function"),
                "function": {
                    "name": getattr(tc.function, "name", ""),
                    "arguments": getattr(tc.function, "arguments", ""),
                }
            })

    return result


def extract_candidate_from_response(response_text: str) -> dict:
    """
    Parse a normal LLM response and extract the candidate heuristic code.

    Expected format:
    - Prefer a single ```python ... ``` block
    - Fallback to existing heuristic code extraction logic

    Returns:
        {
            "raw_response": str,
            "code": Optional[str],
        }
    """
    code = extract_code_from_generator(response_text)
    return {
        "raw_response": response_text,
        "code": code,
    }


def build_eval_feedback_message(
    experiment_id: int,
    result_dict: dict,
    test_obj=None,
) -> dict:
    """
    Build a framework-generated evaluation feedback message for the next round.

    The LLM does NOT output JSON.
    The framework constructs a JSON summary externally and sends it back as user feedback.
    """
    feedback = {
        "experiment": experiment_id,
        "train_obj": result_dict.get("obj"),
        "exec_time": result_dict.get("time"),
        "error": result_dict.get("error"),
    }
    if test_obj is not None:
        feedback["test_obj"] = test_obj

    return {
        "role": "user",
        "content": (
            "Your last candidate heuristic has been evaluated.\n\n"
            "Evaluation result:\n"
            f"```json\n{json.dumps(feedback, indent=2, ensure_ascii=False)}\n```\n\n"
            "Please analyze the result internally and output your next candidate heuristic.\n"
        ),
    }
    
def parse_tool_call_code(tool_call):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    code = function_args.get("code", "")
    return function_name, function_args, code


def save_experiment_record(
    experiment_id: int,
    code: str,
    code_path: str,
    train_obj,
    exec_time,
    error_msg,
    problem_name="",
    problem_size="",
    exp_records_dir="",
    test_obj=None,
    description:Optional[str]=None,
):
    record_path = exp_records_dir / f"exp_{experiment_id:03d}.txt"

    content = []
    content.append(f"Experiment: {experiment_id}")
    content.append(f"Problem: {problem_name}")
    content.append(f"Problem size: {problem_size}")
    content.append(f"Code path: {code_path}")
    content.append("")
    content.append("=== Evaluation Result ===")
    content.append(f"train_obj: {train_obj}")
    content.append(f"exec_time: {exec_time}")
    content.append(f"test_obj: {test_obj}")
    content.append(f"error: {error_msg}")
    content.append("")
    content.append("=== Description ===")
    content.append(description if description is not None else "")
    content.append("")
    content.append("=== Code ===")
    content.append(code)

    record_path.write_text("\n".join(content), encoding="utf-8")


########################### reevo helpers ##########################
def extract_code_from_generator(content: str) -> str:
    pattern_code = r'```python(.*?)```'
    match = re.search(pattern_code, content, re.DOTALL)
    code_string = match.group(1).strip() if match else None

    # fallback
    if code_string is None:
        lines = content.split('\n')
        start = None

        for i, line in enumerate(lines):
            if line.strip().startswith('def'):
                start = i
                break

        if start is not None:
            collected = []
            for line in lines[start:]:
                if line.strip().startswith("```"):
                    break
                collected.append(line)

            if collected:
                code_string = "\n".join(collected).strip()

    if code_string is None:
        return None

    imports = set()
    if 'torch' in code_string:
        imports.add("import torch")
    if 'np.' in code_string or 'numpy' in code_string:
        imports.add("import numpy as np")

    import_header = '\n'.join(sorted(imports)) + '\n\n' if imports else ''
    return import_header + code_string
def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    # assert code_string.count("return") <= 1, "生成的函数 return 太多了，检查 prompt 或手动截断"

    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


import textwrap

def repair_python_function_code(code: str) -> str:
    if code is None:
        return None

    code = textwrap.dedent(code).strip()

    lines = code.splitlines()
    if not lines:
        return code

    # 找到第一个 def
    def_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            def_idx = i
            break

    if def_idx is None:
        return code

    # 如果 def 后面的非空行没有缩进，则整体补 4 空格
    repaired = lines[:def_idx + 1]
    body = lines[def_idx + 1:]

    has_body = False
    need_indent = False
    for line in body:
        if not line.strip():
            continue
        has_body = True
        if len(line) - len(line.lstrip()) == 0:
            need_indent = True
        break

    if has_body and need_indent:
        new_body = []
        for line in body:
            if line.strip():
                new_body.append("    " + line)
            else:
                new_body.append(line)
        repaired.extend(new_body)
        return "\n".join(repaired)

    return "\n".join(lines)