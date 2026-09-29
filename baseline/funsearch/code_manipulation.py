from __future__ import annotations

import ast
import copy
import dataclasses
import io
import tokenize
from typing import Iterator, List, MutableSet


@dataclasses.dataclass
class Function:
    """Lightweight representation of a Python function."""

    name: str
    args: str
    body: str
    decorators: List[str] = dataclasses.field(default_factory=list)
    returns: str | None = None
    docstring: str | None = None

    def header(self, override_name: str | None = None) -> str:
        name = override_name or self.name
        ret = f" -> {self.returns}" if self.returns else ""
        return f"def {name}({self.args}){ret}:"

    def with_name(self, new_name: str) -> "Function":
        new_fn = copy.deepcopy(self)
        new_fn.name = new_name
        return new_fn

    def __str__(self) -> str:
        lines: list[str] = []
        for deco in self.decorators:
            lines.append(deco)
        lines.append(self.header())

        body = self.body.rstrip("\n")
        if not body.strip():
            lines.append("    pass")
        else:
            for line in body.splitlines():
                lines.append(line if line.strip() else "")
        return "\n".join(lines)


@dataclasses.dataclass
class Program:
    """Python source split into preface and top-level functions."""

    preface: str
    functions: List[Function]

    def get_function(self, name: str) -> Function:
        for fn in self.functions:
            if fn.name == name:
                return fn
        raise ValueError(f"Function `{name}` not found in program.")

    def replace_function(self, function: Function) -> None:
        for i, fn in enumerate(self.functions):
            if fn.name == function.name:
                self.functions[i] = function
                return
        self.functions.append(function)

    def remove_function(self, name: str) -> None:
        self.functions = [fn for fn in self.functions if fn.name != name]

    def __str__(self) -> str:
        chunks: list[str] = []
        preface = self.preface.rstrip()
        if preface:
            chunks.append(preface)
        for fn in self.functions:
            chunks.append(str(fn).rstrip())
        return "\n\n".join(chunks).rstrip() + "\n"


class _TopLevelFunctionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit_FunctionDef(self, node):  # noqa: N802
        self.functions.append(node)

    def visit_AsyncFunctionDef(self, node):  # noqa: N802
        self.functions.append(node)

    def generic_visit(self, node):
        if isinstance(node, ast.Module):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.visit(child)
        # skip nested traversal


def _get_source_segment(text: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(text, node)
    return seg if seg is not None else ""


def _node_to_function(text: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Function:
    decorators = [_get_source_segment(text, deco).strip() for deco in node.decorator_list]
    args = ast.unparse(node.args)

    returns = ast.unparse(node.returns) if node.returns is not None else None

    lines = text.splitlines()
    doc = ast.get_docstring(node)

    function_end_line = node.end_lineno
    body_start_line = node.body[0].lineno - 1 if node.body else function_end_line

    # 如果第一个语句是 docstring，就把 body 起点移到 docstring 后面
    if node.body:
        doc_node = node.body[0]
        if (
            isinstance(doc_node, ast.Expr)
            and isinstance(getattr(doc_node, "value", None), ast.Constant)
            and isinstance(doc_node.value.value, str)
        ):
            if len(node.body) > 1:
                body_start_line = node.body[1].lineno - 1
            else:
                body_start_line = function_end_line

    body = "\n".join(lines[body_start_line:function_end_line]).rstrip()

    return Function(
        name=node.name,
        args=args,
        body=body,
        decorators=decorators,
        returns=returns,
        docstring=doc,
    )


def text_to_program(text: str) -> Program:
    tree = ast.parse(text)
    collector = _TopLevelFunctionCollector()
    collector.visit(tree)

    functions = [_node_to_function(text, node) for node in collector.functions]
    if collector.functions:
        first_lineno = min(node.lineno for node in collector.functions)
        preface = "\n".join(text.splitlines()[: first_lineno - 1]).rstrip()
    else:
        preface = text.rstrip()
    return Program(preface=preface, functions=functions)


def text_to_function(text: str) -> Function:
    program = text_to_program(text)
    if len(program.functions) != 1:
        raise ValueError("Expected exactly one function in text.")
    return program.functions[0]


def program_to_text(program: Program) -> str:
    return str(program)


def _untokenize(tokens: list[tokenize.TokenInfo]) -> str:
    return tokenize.untokenize(tokens)


def _yield_token_and_is_call(code: str):
    token_stream = tokenize.generate_tokens(io.StringIO(code).readline)
    tokens = list(token_stream)
    for i, tok in enumerate(tokens[:-1]):
        is_call = tok.type == tokenize.NAME and tokens[i + 1].string == "("
        yield tok, is_call
    if tokens:
        yield tokens[-1], False


def get_functions_called(code: str) -> MutableSet[str]:
    return {tok.string for tok, is_call in _yield_token_and_is_call(code) if is_call}


def rename_function_calls(code: str, source_name: str, target_name: str) -> str:
    modified: list[tokenize.TokenInfo] = []
    for tok, is_call in _yield_token_and_is_call(code):
        if is_call and tok.string == source_name:
            modified.append(
                tokenize.TokenInfo(tok.type, target_name, tok.start, tok.end, tok.line)
            )
        else:
            modified.append(tok)
    return _untokenize(modified)


def rename_function_definition(code: str, new_name: str) -> str:
    tree = ast.parse(code)
    fn_nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(fn_nodes) != 1:
        raise ValueError("Expected exactly one top-level function.")
    fn = fn_nodes[0]
    lines = code.splitlines()
    def_line_idx = fn.lineno - 1
    line = lines[def_line_idx]
    prefix, _, suffix = line.partition(fn.name)
    lines[def_line_idx] = prefix + new_name + suffix
    return "\n".join(lines)


def replace_function(program_text: str, new_function_text: str, function_name: str) -> str:
    program = text_to_program(program_text)
    new_function = text_to_function(new_function_text)
    if new_function.name != function_name:
        new_function.name = function_name
    program.replace_function(new_function)
    return str(program)


def yield_decorated(code: str, module: str, name: str) -> Iterator[str]:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                attribute = None
                if isinstance(decorator, ast.Attribute):
                    attribute = decorator
                elif isinstance(decorator, ast.Call):
                    attribute = decorator.func
                if isinstance(attribute, ast.Attribute) and isinstance(attribute.value, ast.Name):
                    if attribute.value.id == module and attribute.attr == name:
                        yield node.name
