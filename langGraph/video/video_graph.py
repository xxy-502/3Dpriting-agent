"""Two-stage LangGraph workflow for local video defect inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .video_nodes import VideoDetectionNodes, VideoWorkflowState
from .video_tools import (
    VideoSamRenderingTool,
    VideoSamSettings,
    VideoYoloDetectionTool,
    VideoYoloSettings,
)


def build_video_detection_graph(
    *,
    nodes: VideoDetectionNodes | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the reusable YOLO -> optional SAM subgraph."""

    graph_nodes = nodes or VideoDetectionNodes()
    builder = StateGraph(VideoWorkflowState)

    builder.add_node("video_yolo_detection", graph_nodes.video_yolo_detection)
    builder.add_node("video_sam_rendering", graph_nodes.video_sam_rendering)
    builder.add_node("finalize_video_result", graph_nodes.finalize_video_result)

    builder.add_edge(START, "video_yolo_detection")
    builder.add_conditional_edges(
        "video_yolo_detection",
        graph_nodes.route_after_yolo,
        {
            "render": "video_sam_rendering",
            "finish": "finalize_video_result",
        },
    )
    builder.add_edge("video_sam_rendering", "finalize_video_result")
    builder.add_edge("finalize_video_result", END)
    return builder.compile(checkpointer=checkpointer)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--render-result-video", action="store_true")
    parser.add_argument("--output-video-name", default="defect_mask_tracking.mp4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--yolo-checkpoint", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--coarse-stride", type=int, default=7)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    nodes = VideoDetectionNodes(
        yolo_tool=VideoYoloDetectionTool(
            VideoYoloSettings(
                checkpoint=args.yolo_checkpoint,
                device=args.device,
                coarse_stride=args.coarse_stride,
            )
        ),
        sam_tool=VideoSamRenderingTool(
            VideoSamSettings(
                checkpoint=args.sam_checkpoint,
                device=args.device,
                output_video_name=args.output_video_name,
            )
        ),
    )
    graph = build_video_detection_graph(nodes=nodes)
    result = graph.invoke(
        {
            "video_path": str(args.video_path),
            "output_dir": str(args.output_dir) if args.output_dir else None,
            "render_result_video": args.render_result_video,
            "output_video_name": args.output_video_name,
        }
    )
    print(json.dumps(result["video_result"], ensure_ascii=False, indent=2))
    return 0 if result["video_result"].get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
