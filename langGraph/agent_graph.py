"""Build the complete offline LangGraph 3D-printing defect agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .agent_nodes import AgentServices, DefectAgentNodes
from .agent_state import DefectAgentState
from .image_defect.defect_tool import DefectDetectionTool
from .local_qwen import DEFAULT_QWEN_ROOT, LocalQwenRuntime
from .model_paths import BGE_MODEL_DIR, SAM_MODEL_DIR, YOLO_MODEL_DIR
from .rag_retriever import DEFAULT_COLLECTION_DIR, OfflineHybridRetriever
from .video.video_graph import build_video_detection_graph
from .video.video_nodes import VideoDetectionNodes
from .video.video_tools import (
    VideoSamRenderingTool,
    VideoSamSettings,
    VideoYoloDetectionTool,
    VideoYoloSettings,
)


LANGGRAPH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LANGGRAPH_ROOT.parent
DEFAULT_RUN_ROOT = LANGGRAPH_ROOT / "runs" / "agent"
DEFAULT_YOLO_CHECKPOINT = YOLO_MODEL_DIR
DEFAULT_SAM_CHECKPOINT = SAM_MODEL_DIR


def create_default_services(
    *,
    qwen_model_path: Path | str = DEFAULT_QWEN_ROOT,
    bge_model_path: Path | str = BGE_MODEL_DIR,
    collection_dir: Path | str = DEFAULT_COLLECTION_DIR,
    output_dir: Path | str = DEFAULT_RUN_ROOT,
    llm_device: str = "auto",
    vision_device: str = "auto",
    rag_device: str = "cpu",
    max_new_tokens: int = 768,
    video_coarse_stride: int = 7,
) -> AgentServices:
    output_root = Path(output_dir).expanduser().resolve()
    image_tool = DefectDetectionTool(
        output_root=output_root / "image",
        eager_load=False,
        yolo_checkpoint=DEFAULT_YOLO_CHECKPOINT,
        sam_checkpoint=DEFAULT_SAM_CHECKPOINT,
        device=vision_device,
    )

    video_nodes = VideoDetectionNodes(
        yolo_tool=VideoYoloDetectionTool(
            VideoYoloSettings(
                checkpoint=DEFAULT_YOLO_CHECKPOINT,
                device=vision_device,
                coarse_stride=video_coarse_stride,
            )
        ),
        sam_tool=VideoSamRenderingTool(
            VideoSamSettings(
                checkpoint=DEFAULT_SAM_CHECKPOINT,
                device=vision_device,
            )
        ),
    )
    video_graph = build_video_detection_graph(nodes=video_nodes)
    retriever = OfflineHybridRetriever(
        collection_dir,
        bge_model_path=bge_model_path,
        device=rag_device,
    )
    llm = LocalQwenRuntime(
        qwen_model_path,
        device=llm_device,
        max_new_tokens=max_new_tokens,
        eager_load=False,
    )
    return AgentServices(
        image_tool=image_tool,
        video_graph=video_graph,
        retriever=retriever,
        llm=llm,
    )


def build_defect_agent(
    *,
    services: AgentServices | None = None,
    checkpointer: Any | None = None,
    rag_limit: int = 6,
    **service_kwargs: Any,
) -> Any:
    """Compile the deterministic media detection -> RAG -> Qwen workflow."""

    if services is not None and service_kwargs:
        raise ValueError("service_kwargs cannot be used with an explicit services object")
    runtime_services = services or create_default_services(**service_kwargs)
    nodes = DefectAgentNodes(runtime_services, rag_limit=rag_limit)

    builder = StateGraph(DefectAgentState)
    builder.add_node("prepare_request", nodes.prepare_request)
    builder.add_node("image_detection", nodes.image_detection)
    builder.add_node("video_detection", nodes.video_detection)
    builder.add_node("retrieve_knowledge", nodes.retrieve_knowledge)
    builder.add_node("generate_answer", nodes.generate_answer)
    builder.add_node("error_response", nodes.error_response)

    builder.add_edge(START, "prepare_request")
    builder.add_conditional_edges(
        "prepare_request",
        nodes.route_request,
        {
            "image": "image_detection",
            "video": "video_detection",
            "rag": "retrieve_knowledge",
            "error": "error_response",
        },
    )
    builder.add_conditional_edges(
        "image_detection",
        nodes.route_after_detection,
        {"rag": "retrieve_knowledge", "error": "error_response"},
    )
    builder.add_conditional_edges(
        "video_detection",
        nodes.route_after_detection,
        {"rag": "retrieve_knowledge", "error": "error_response"},
    )
    builder.add_conditional_edges(
        "retrieve_knowledge",
        nodes.route_after_rag,
        {"generate": "generate_answer", "error": "error_response"},
    )
    builder.add_edge("generate_answer", END)
    builder.add_edge("error_response", END)
    return builder.compile(checkpointer=checkpointer)
