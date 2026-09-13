"""LangGraph-ready adapters for the local 3D-printing inspection pipeline.

Exports are loaded lazily so ``python -m langGraph.video.video_graph`` and
``python -m langGraph.video.video_tools`` do not import their target module twice.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "VideoDetectionNodes",
    "VideoSamRenderingTool",
    "VideoSamSettings",
    "VideoWorkflowState",
    "VideoYoloDetectionTool",
    "VideoYoloSettings",
    "build_video_detection_graph",
]


_EXPORT_MODULES = {
    "VideoDetectionNodes": ".video_nodes",
    "VideoWorkflowState": ".video_nodes",
    "VideoSamRenderingTool": ".video_tools",
    "VideoSamSettings": ".video_tools",
    "VideoYoloDetectionTool": ".video_tools",
    "VideoYoloSettings": ".video_tools",
    "build_video_detection_graph": ".video_graph",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
