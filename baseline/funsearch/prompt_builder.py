from __future__ import annotations

from typing import Sequence

from . import code_manipulation


def _ensure_versioned_functions(
    sampled_functions: Sequence[code_manipulation.Function],
    function_to_evolve: str,
) -> list[code_manipulation.Function]:
    versioned = []
    for i, fn in enumerate(sampled_functions):
        renamed = fn.with_name(f"{function_to_evolve}_v{i}")
        renamed.body = code_manipulation.rename_function_calls(
            renamed.body,
            source_name=function_to_evolve,
            target_name=renamed.name,
        )
        versioned.append(renamed)
    return versioned


def build_prompt_code(
    template: code_manipulation.Program,
    sampled_functions: Sequence[code_manipulation.Function],
    function_to_evolve: str,
) -> tuple[str, int]:
    if not sampled_functions:
        raise ValueError("sampled_functions cannot be empty.")

    versioned = _ensure_versioned_functions(sampled_functions, function_to_evolve)
    next_version = len(versioned)
    header_source = sampled_functions[-1]
    header = code_manipulation.Function(
        name=f"{function_to_evolve}_v{next_version}",
        args=header_source.args,
        body="",
        returns=header_source.returns,
        decorators=[],
        docstring=f"Improved version of {function_to_evolve}_v{next_version - 1}",
    )
    prompt_program = code_manipulation.Program(preface=template.preface, functions=versioned + [header])
    return str(prompt_program), next_version


def build_chat_messages(
    task_desc: str,
    prompt_code: str,
    function_to_evolve: str,
) -> list[dict]:
    system = (
        "You are improving a Python heuristic function inside a fixed template. "
        "Return exactly one Python code block containing only the full improved function definition. "
        "Do not explain anything outside the code block."
    )
    user = (
        f"{task_desc.strip()}\n\n"
        f"Below are previous versions of `{function_to_evolve}` followed by the header of a new version to complete.\n"
        "Write a better implementation for the last function.\n"
        "Requirements:\n"
        "- Output exactly one complete Python function.\n"
        "- Do not call older versioned functions unless you intentionally want the candidate rejected.\n"
        "- Keep the same signature.\n"
        "- The function must be executable.\n\n"
        f"```python\n{prompt_code.rstrip()}\n```"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
