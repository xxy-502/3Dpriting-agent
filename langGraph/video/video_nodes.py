"""LangGraph node adapters for the two-stage video inspection workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict

from .video_tools import VideoSamRenderingTool, VideoYoloDetectionTool


class VideoWorkflowState(TypedDict, total=False):
    """Serializable state shared by the video detection graph."""

    video_path: str
    output_dir: str | None
    render_result_video: bool
    output_video_name: str

    yolo_result: dict[str, Any]
    sam_result: dict[str, Any]
    detection_result: dict[str, Any]
    video_result: dict[str, Any]
    result_video_path: str | None
    error: str


def _failure(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


class VideoDetectionNodes:
    """Bound node methods sharing resident YOLO and SAM tool instances."""

    def __init__(
        self,
        yolo_tool: VideoYoloDetectionTool | None = None,
        sam_tool: VideoSamRenderingTool | None = None,
    ) -> None:
        self.yolo_tool = yolo_tool or VideoYoloDetectionTool()
        self.sam_tool = sam_tool or VideoSamRenderingTool()

    def video_yolo_detection(self, state: VideoWorkflowState) -> dict[str, Any]:
        """Run temporal YOLO and persist full events for the optional SAM node."""

        try:
            result = self.yolo_tool.detect(
                state["video_path"],
                output_dir=state.get("output_dir"),
            )
        except Exception as exc:
            result = _failure("video_yolo_detection", exc)

        return {
            "yolo_result": result,
            "detection_result": result,
            # Clear data that may remain when a persistent thread is reused.
            "sam_result": {},
            "result_video_path": None,
            "error": result.get("error", ""),
        }

    def route_after_yolo(
        self, state: VideoWorkflowState
    ) -> Literal["render", "finish"]:
        """Render only when YOLO succeeded, found defects, and output was requested."""

        result = state.get("yolo_result", {})
        should_render = (
            result.get("success") is True
            and result.get("defect_exists") is True
            and bool(state.get("render_result_video", False))
        )
        return "render" if should_render else "finish"

    def video_sam_rendering(self, state: VideoWorkflowState) -> dict[str, Any]:
        """Load the persisted YOLO events and create a SAM mask-overlay video."""

        yolo_result = state.get("yolo_result", {})
        try:
            json_path = yolo_result["artifacts"]["yolo_events_json_path"]
            output_dir = state.get("output_dir") or str(Path(json_path).parent)
            result = self.sam_tool.render(
                state["video_path"],
                json_path,
                output_dir=output_dir,
                output_video_name=state.get("output_video_name"),
            )
        except Exception as exc:
            result = _failure("video_sam_rendering", exc)

        return {
            "sam_result": result,
            "result_video_path": result.get("result_video_path"),
            "error": result.get("error", ""),
        }

    def finalize_video_result(self, state: VideoWorkflowState) -> dict[str, Any]:
        """Expose one compact output field to the parent multimodal graph."""

        yolo_result = dict(state.get("yolo_result", {}))
        sam_result = state.get("sam_result") or None
        if sam_result is not None:
            yolo_result["sam_rendering"] = sam_result
            if sam_result.get("success"):
                yolo_result["result_video_path"] = sam_result.get(
                    "result_video_path"
                )
        yolo_result["render_result_video"] = bool(
            state.get("render_result_video", False)
        )
        return {"video_result": yolo_result}
