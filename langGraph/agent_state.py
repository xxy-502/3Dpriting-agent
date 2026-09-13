"""Shared state schema for the offline 3D-printing defect agent."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class DefectAgentState(TypedDict, total=False):
    user_query: str
    media_path: str | None
    media_type: Literal["image", "video", "none"]
    print_context: dict[str, Any]
    detection_result: dict[str, Any] | None

    output_dir: str | None
    render_result_video: bool
    output_video_name: str

    video_result: dict[str, Any]
    rag_queries: list[str]
    rag_documents: list[dict[str, Any]]
    sources: list[dict[str, Any]]

    result_image_path: str | None
    result_video_path: str | None
    final_answer: str
    guardrail_issues: list[str]
    error: str
