"""Offline LangGraph agent for 3D-printing defect inspection."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["build_defect_agent"]


def __getattr__(name: str) -> Any:
    if name != "build_defect_agent":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(".agent_graph", __name__).build_defect_agent
    globals()[name] = value
    return value
