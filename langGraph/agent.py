"""Command-line entry point for the offline 3D-printing defect agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agent_graph import build_defect_agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="Run one request; omit for interactive mode")
    parser.add_argument("--media-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--render-result-video", action="store_true")
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument("--vision-device", default="auto")
    parser.add_argument("--rag-device", default="cpu")
    parser.add_argument("--bge-model-path", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--video-coarse-stride", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Print the complete state")
    return parser


def _input_state(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    return {
        "user_query": prompt,
        "media_path": str(args.media_path) if args.media_path else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "render_result_video": bool(args.render_result_video),
    }


def _print_result(result: dict[str, Any], *, complete_json: bool) -> None:
    if complete_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["final_answer"])
    if result.get("result_image_path"):
        print(f"结果图片：{result['result_image_path']}")
    if result.get("result_video_path"):
        print(f"结果视频：{result['result_video_path']}")


def main() -> int:
    args = _build_parser().parse_args()
    graph = build_defect_agent(
        output_dir=args.output_dir or Path(__file__).resolve().parent / "runs" / "agent",
        llm_device=args.llm_device,
        vision_device=args.vision_device,
        rag_device=args.rag_device,
        **({"bge_model_path": args.bge_model_path} if args.bge_model_path else {}),
        max_new_tokens=args.max_new_tokens,
        video_coarse_stride=args.video_coarse_stride,
    )

    if args.prompt:
        result = graph.invoke(_input_state(args, args.prompt))
        _print_result(result, complete_json=args.json)
        return 0 if not result.get("error") else 1

    print("输入问题开始分析；输入 /exit 退出。")
    while True:
        try:
            prompt = input("用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit"}:
            break
        result = graph.invoke(_input_state(args, prompt))
        _print_result(result, complete_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
