"""Reasoning layer: deterministic tool routing, ReAct loop, self-consistency.

Author: 晨星
"""
from .react import REACT_SYSTEM, ReActAgent, self_consistent
from .router import Route, route
from .tools import Calculator, Clock, UnitConverter, default_tools, find_tool

__all__ = [
    "route",
    "Route",
    "ReActAgent",
    "REACT_SYSTEM",
    "self_consistent",
    "Calculator",
    "Clock",
    "UnitConverter",
    "default_tools",
    "find_tool",
]
