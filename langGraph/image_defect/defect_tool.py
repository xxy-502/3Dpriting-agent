"""Agent-facing adapter for the resident YOLO-OBB + SAM2 detector.

This module owns parameter validation and converts the detailed inference
record into concise JSON that a language model can safely summarize.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

if __package__:  # Package import: python -m agent.qwen_agent
    from .inference import (
        DEFAULT_OUTPUT_PATH,
        IMAGE_SUFFIXES,
        YoloSamInferencer,
        get_inferencer,
        infer_single_image,
    )
else:  # Direct script: python agent/defect_tool.py
    from inference import (
        DEFAULT_OUTPUT_PATH,
        IMAGE_SUFFIXES,
        YoloSamInferencer,
        get_inferencer,
        infer_single_image,
    )


DETECT_DEFECTS_TOOL_NAME = "detect_defects"
DETECT_DEFECTS_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DETECT_DEFECTS_TOOL_NAME,
        "description": (
            "检测一张本地图片中的3D打印缺陷。工具会执行YOLO定位和SAM分割，"
            "返回是否存在缺陷、缺陷数量、类别统计、置信度以及结果图片路径。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "待检测图片的本地绝对路径。必须原样保留路径内容。",
                }
            },
            "required": ["image_path"],
            "additionalProperties": False,
        },
    },
}


class ToolValidationError(ValueError):
    """Raised when language-model-provided tool arguments are invalid."""


def _strip_path_wrappers(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def validate_image_path(image_path: Any) -> Path:
    """Normalize and validate one image path on both Windows and Linux."""

    if not isinstance(image_path, str) or not image_path.strip():
        raise ToolValidationError("image_path 必须是非空字符串")

    raw_path = _strip_path_wrappers(image_path)
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()

    # Chat/Markdown input sometimes escapes underscores with a backslash. Only try
    # this recovery when the exact path does not exist, then validate again.
    if not candidate.exists() and "\\_" in raw_path:
        recovered = Path(raw_path.replace("\\_", "_")).expanduser()
        if not recovered.is_absolute():
            recovered = Path.cwd() / recovered
        recovered = recovered.resolve()
        if recovered.exists():
            candidate = recovered

    if not candidate.exists():
        raise ToolValidationError(f"图片路径不存在：{candidate}")
    if not candidate.is_file():
        raise ToolValidationError(f"必须传入单张图片文件，不能传入目录：{candidate}")
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ToolValidationError(
            f"不支持的图片格式 {candidate.suffix!r}，支持格式：{supported}"
        )

    try:
        with Image.open(candidate) as image:
            image.verify()
    except Exception as exc:
        raise ToolValidationError(f"图片文件无法读取或已经损坏：{candidate}") from exc
    return candidate


def _compact_detection(detection: dict[str, Any]) -> dict[str, Any]:
    """Remove verbose OBB vertices before returning data to the LLM."""

    return {
        "rank": detection["rank"],
        "class_id": detection["class_id"],
        "class_name": detection["class_name"],
        "confidence": detection["confidence"],
        "box_xyxy": detection["box_xyxy"],
        "sam_score": detection.get("sam_score"),
        "mask_area_pixels": detection.get("mask_area_pixels"),
        "mask_area_ratio": detection.get("mask_area_ratio"),
    }


class DefectDetectionTool:
    """Validated, reusable tool exposed to the Qwen agent."""

    name = DETECT_DEFECTS_TOOL_NAME
    schema = DETECT_DEFECTS_TOOL_SCHEMA

    def __init__(
        self,
        inferencer: YoloSamInferencer | None = None,
        output_root: Path | str = DEFAULT_OUTPUT_PATH,
        box_expand: float = 10,
        max_masks: int | None = None,
        eager_load: bool = True,
        **inferencer_kwargs: Any,
    ) -> None:
        if box_expand < 0:
            raise ValueError("box_expand must be >= 0")
        if max_masks is not None and max_masks < 1:
            raise ValueError("max_masks must be >= 1 or None")
        self.output_root = Path(output_root).expanduser().resolve()
        self.box_expand = float(box_expand)
        self.max_masks = max_masks
        self._inferencer = inferencer
        self._inferencer_kwargs = inferencer_kwargs
        self._load_lock = threading.Lock()
        if eager_load:
            self.load()

    def load(self) -> YoloSamInferencer:
        """Load YOLO/SAM once and keep the service resident."""

        if self._inferencer is None:
            with self._load_lock:
                if self._inferencer is None:
                    self._inferencer = get_inferencer(**self._inferencer_kwargs)
        return self._inferencer

    def detect_defects(self, image_path: Any) -> dict[str, Any]:
        """Validate, infer, summarize and return an Agent-safe result."""

        try:
            validated_path = validate_image_path(image_path)
            raw_result = infer_single_image(
                image_path=validated_path,
                output_root=self.output_root,
                box_expand=self.box_expand,
                max_masks=self.max_masks,
                inferencer=self.load(),
            )
        except (ToolValidationError, ValueError, FileNotFoundError) as exc:
            return {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "image_path": str(image_path) if image_path is not None else None,
            }
        except Exception as exc:
            return {
                "success": False,
                "error_type": type(exc).__name__,
                "error": f"缺陷检测执行失败：{exc}",
                "image_path": str(image_path) if image_path is not None else None,
            }

        detections = raw_result["detections"]
        class_counts = dict(
            sorted(Counter(item["class_name"] for item in detections).items())
        )
        compact_detections = [_compact_detection(item) for item in detections]
        defect_count = int(raw_result["defect_count"])
        segmented_count = int(raw_result["segmented_count"])

        if defect_count == 0:
            summary = "未检测到达到置信度阈值的缺陷。"
        else:
            categories = "、".join(
                f"{name} {count}处" for name, count in class_counts.items()
            )
            summary = f"检测到{defect_count}处缺陷（{categories}），已分割{segmented_count}处。"

        return {
            "success": True,
            "image_path": raw_result["image_path"],
            "defect_exists": bool(raw_result["defect_exists"]),
            "defect_count": defect_count,
            "segmented_count": segmented_count,
            "class_counts": class_counts,
            "detections": compact_detections,
            "result_image_path": raw_result["result_image_path"],
            "step2_output_path": raw_result["step2_output_path"],
            "metadata_path": raw_result["metadata_path"],
            "run_directory": raw_result["run_directory"],
            "summary": summary,
        }

    def execute(self, tool_name: str, arguments: Any) -> dict[str, Any]:
        """Dispatch only explicitly registered tools."""

        if tool_name != self.name:
            return {
                "success": False,
                "error_type": "UnknownToolError",
                "error": f"不允许调用未知工具：{tool_name}",
            }
        if not isinstance(arguments, dict):
            return {
                "success": False,
                "error_type": "ToolValidationError",
                "error": "工具 arguments 必须是 JSON 对象",
            }
        unexpected = set(arguments) - {"image_path"}
        if unexpected:
            return {
                "success": False,
                "error_type": "ToolValidationError",
                "error": f"工具包含不支持的参数：{sorted(unexpected)}",
            }
        return self.detect_defects(arguments.get("image_path"))


_DEFAULT_TOOL: DefectDetectionTool | None = None
_DEFAULT_TOOL_LOCK = threading.Lock()


def get_default_tool() -> DefectDetectionTool:
    """Return a process-wide resident defect tool."""

    global _DEFAULT_TOOL
    if _DEFAULT_TOOL is None:
        with _DEFAULT_TOOL_LOCK:
            if _DEFAULT_TOOL is None:
                _DEFAULT_TOOL = DefectDetectionTool()
    return _DEFAULT_TOOL


def detect_defects(image_path: str) -> dict[str, Any]:
    """Function-style entry point used by integrations and tests."""

    return get_default_tool().detect_defects(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Agent defect-detection tool once.")
    parser.add_argument("image_path", help="Local image path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--box-expand", type=float, default=10)
    parser.add_argument(
        "--max-masks",
        type=int,
        default=None,
        help="Default is None, which segments every detection.",
    )
    args = parser.parse_args()

    tool = DefectDetectionTool(
        device=args.device,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.image_size,
        box_expand=args.box_expand,
        max_masks=args.max_masks,
    )
    result = tool.detect_defects(args.image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
