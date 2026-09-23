"""Deterministic pre-routing.

A 0.5B model cannot be trusted to decide whether `12*(3+4)` is arithmetic, and
it certainly cannot be trusted to *not* re-derive the result after a tool
already produced it. So the routing decision is made by regex, before the model
ever sees the query.

Author: 晨星
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ARITH = re.compile(r"[\d\.\s]*[\d\.\s+\-*/×÷^()]+[+\-*/×÷^][\d\.\s+\-*/×÷^()]*\d")
_TIME = re.compile(r"(现在几点|当前时间|今天几号|今天是什么日子|what time|current time|today's date)")
_UNIT = re.compile(r"([-+0-9.]+)\s*([a-zA-Z]+)\s*(?:to|->|到|为)\s*([a-zA-Z]+)")


@dataclass(frozen=True)
class Route:
    kind: str  # calculator | clock | unit | rag
    tool: str = ""
    argument: str = ""

    @property
    def is_tool(self) -> bool:
        return self.kind != "rag"


def route(query: str) -> Route:
    """Classify a query. Pure function, no model involved."""
    q = query.strip()
    if _TIME.search(q):
        return Route(kind="clock", tool="clock", argument="")
    m = _UNIT.search(q)
    if m and not _ARITH.fullmatch(q or " "):
        return Route(kind="unit", tool="unit", argument=m.group(0))
    if _ARITH.search(q) and re.search(r"\d", q):
        # Only treat as arithmetic when the string is dominated by maths.
        body = q.strip(" ?？。.了是多少等于")
        if _ARITH.fullmatch(body) or re.fullmatch(r"[\d\.\s+\-*/×÷^()]+", body):
            return Route(kind="calculator", tool="calculator", argument=body)
    return Route(kind="rag")
