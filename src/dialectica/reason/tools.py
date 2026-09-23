"""Tools exposed to the agent. Pure functions, no side effects, no network.

Author: 晨星
"""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime

from ..core.errors import ToolError

_ALLOWED_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FN = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "pow": math.pow,
}


class Calculator:
    """Safe arithmetic evaluator -- AST whitelist, never `eval`."""

    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. 12*(3+4)"

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN:
            return _ALLOWED_BIN[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UN:
            return _ALLOWED_UN[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname in _ALLOWED_FN and not node.keywords:
                return float(_ALLOWED_FN[fname](*[self._eval(a) for a in node.args]))
        raise ToolError(f"disallowed expression element: {type(node).__name__}")

    def invoke(self, argument: str) -> str:
        expr = argument.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
        expr = expr.replace(",", "")
        if not expr:
            raise ToolError("empty expression")
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"bad expression: {exc}") from exc
        value = self._eval(tree.body)
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"


class Clock:
    """Current local time. Deterministic enough for tests via injection."""

    name = "clock"
    description = "Return the current local date and time"

    def __init__(self, now_fn=None) -> None:
        self._now = now_fn or (lambda: datetime.now())

    def invoke(self, argument: str = "") -> str:
        return self._now().strftime("%Y-%m-%d %H:%M:%S")


class UnitConverter:
    """Length / weight / time unit conversion with an explicit factor table."""

    name = "unit"
    description = "Convert between units, e.g. '12 km to m'"

    FACTORS = {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "mi": 1609.344,
        "ft": 0.3048,
        "g": 1.0,
        "kg": 1000.0,
        "mg": 0.001,
        "lb": 453.59237,
        "s": 1.0,
        "min": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }

    def invoke(self, argument: str) -> str:
        import re

        m = re.match(r"\s*([-+0-9.]+)\s*([a-zA-Z]+)\s*(?:to|->|到|为)\s*([a-zA-Z]+)\s*$", argument)
        if not m:
            raise ToolError(f"cannot parse conversion: {argument}")
        value, src, dst = float(m.group(1)), m.group(2).lower(), m.group(3).lower()
        if src not in self.FACTORS or dst not in self.FACTORS:
            raise ToolError(f"unknown unit: {src} or {dst}")
        result = value * self.FACTORS[src] / self.FACTORS[dst]
        return f"{result:.6g} {dst}"


def default_tools() -> list:
    return [Calculator(), Clock(), UnitConverter()]


def find_tool(tools, name: str):
    for t in tools:
        if t.name == name:
            return t
    return None
