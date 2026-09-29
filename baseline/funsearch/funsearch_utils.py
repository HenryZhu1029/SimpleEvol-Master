from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import code_manipulation

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


class _FunctionLineVisitor(ast.NodeVisitor):
    def __init__(self, target_function_name: str) -> None:
        self._target_function_name = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node):  # noqa: N802
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        if self._function_end_line is None:
            raise ValueError("Function end line not found.")
        return self._function_end_line


def extract_function_names_from_spec(specification: str) -> tuple[str | None, str | None]:
    run_fns = list(code_manipulation.yield_decorated(specification, "funsearch", "run"))
    evolve_fns = list(code_manipulation.yield_decorated(specification, "funsearch", "evolve"))
    run_name = run_fns[0] if len(run_fns) == 1 else None
    evolve_name = evolve_fns[0] if len(evolve_fns) == 1 else None
    return evolve_name, run_name


def _extract_function_block(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fns:
        return None
    node = fns[0]
    seg = ast.get_source_segment(text, node)
    return seg.strip() if seg else None


def normalize_generated_function(raw_text: str) -> str | None:
    """Try to recover one standalone function from model output."""
    if not raw_text:
        return None
    block = _extract_function_block(raw_text)
    if block:
        return block

    lines = raw_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            start = i
            break
    if start is None:
        return None

    candidate = "\n".join(lines[start:]).strip()
    block = _extract_function_block(candidate)
    return block.strip() if block else candidate


def trim_generated_function_body(generated_code: str) -> str:
    if not generated_code:
        return ""
    code = f"def __temp__():\n{generated_code}"
    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            lines = code.splitlines()
            code = "\n".join(lines[: max(1, (e.lineno or len(lines)) - 1)])
            if code.strip() == "def __temp__():":
                return ""
    visitor = _FunctionLineVisitor("__temp__")
    visitor.visit(tree)
    body_lines = code.splitlines()[1: visitor.function_end_line]
    return "\n".join(body_lines).rstrip() + "\n"


def calls_ancestor(program: str, function_to_evolve: str) -> bool:
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f"{function_to_evolve}_v"):
            return True
    return False


def reduce_score(scores_per_test: ScoresPerTest, obj_type: str = "max") -> float:
    if not scores_per_test:
        raise ValueError("scores_per_test cannot be empty.")
    vals = [float(scores_per_test[k]) for k in sorted(scores_per_test.keys(), key=str)]
    score = sum(vals) / len(vals)
    return -score if obj_type == "min" else score


def make_signature(scores_per_test: ScoresPerTest, ndigits: int = 8) -> Signature:
    return tuple(round(float(scores_per_test[k]), ndigits) for k in sorted(scores_per_test.keys(), key=str))


def build_scores_from_result(result_dict: dict, fallback_key: str = "obj") -> dict[str, float]:
    details = result_dict.get("details") or {}
    for key in ("per_size_obj", "scores_per_test", "per_case_obj"):
        val = details.get(key)
        if isinstance(val, dict) and val:
            return {str(k): float(v) for k, v in val.items()}
    obj = result_dict.get(fallback_key)
    if obj is None:
        return {}
    return {"aggregate": float(obj)}


def sample_to_program(
    generated_function_code: str,
    version_generated: int | None,
    template: code_manipulation.Program,
    function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """Insert one generated function into the fixed template and return full program."""
    generated_function_code = normalize_generated_function(generated_function_code) or generated_function_code
    new_function = code_manipulation.text_to_function(generated_function_code)

    body = trim_generated_function_body(new_function.body)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            body,
            source_name=f"{function_to_evolve}_v{version_generated}",
            target_name=function_to_evolve,
        )
    new_function.name = function_to_evolve
    new_function.body = body
    new_function.decorators = []

    program = copy.deepcopy(template)
    program.replace_function(new_function)
    return new_function, str(program)


def load_text_if_exists(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def make_default_seed_function(func_name: str) -> str:
    return (
        f"def {func_name}(*args, **kwargs):\n"
        f"    unvisited_nodes = kwargs.get('unvisited_nodes', None)\n"
        f"    if unvisited_nodes:\n"
        f"        return min(unvisited_nodes)\n"
        f"    for value in args:\n"
        f"        if isinstance(value, (set, list, tuple)) and len(value) > 0:\n"
        f"            return min(value) if not isinstance(value, tuple) else value[0]\n"
        f"    raise ValueError('No feasible move found in default seed.')\n"
    )


def make_default_template(func_name: str) -> str:
    return (
        "import math\n"
        "import random\n"
        "import numpy as np\n\n"
        f"def {func_name}(*args, **kwargs):\n"
        "    raise NotImplementedError\n"
    )
