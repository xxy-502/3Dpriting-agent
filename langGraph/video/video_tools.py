"""Reusable video detection and SAM rendering tools.

The heavy algorithms live in the sibling ``video_pipeline.py`` module. This
module provides small, validated, JSON-serializable adapters suitable for LangGraph
nodes, CLIs, Gradio applications, and tests.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..model_paths import (
    SAM_MODEL_DIR,
    YOLO_MODEL_DIR,
    resolve_sam_checkpoint,
    resolve_yolo_checkpoint,
)


PACKAGE_DIR = Path(__file__).resolve().parent
LANGGRAPH_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = LANGGRAPH_ROOT.parent
DEFAULT_OUTPUT_ROOT = PACKAGE_DIR / "runs" / "video_detection"

DEFAULT_YOLO_CHECKPOINT = YOLO_MODEL_DIR
DEFAULT_SAM_CHECKPOINT = SAM_MODEL_DIR
DEFAULT_SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_b+"

VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


class VideoToolValidationError(ValueError):
    """Raised when a caller supplies an invalid video-tool argument."""


def _load_video_pipeline() -> Any:
    """Import the pipeline bundled in this package only when video is used."""

    from . import video_pipeline

    return video_pipeline


def _strip_path_wrappers(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`":
        text = text[1:-1].strip()
    if len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def _resolve_path(value: Any, *, label: str, must_exist: bool = True) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise VideoToolValidationError(f"{label} must be a non-empty path")

    raw_path = _strip_path_wrappers(str(value))
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()

    # Recover paths copied from Markdown, where underscores may be escaped.
    if must_exist and not candidate.exists() and "\\_" in raw_path:
        recovered = Path(raw_path.replace("\\_", "_")).expanduser()
        if not recovered.is_absolute():
            recovered = Path.cwd() / recovered
        recovered = recovered.resolve()
        if recovered.exists():
            candidate = recovered

    if must_exist and not candidate.exists():
        raise VideoToolValidationError(f"{label} does not exist: {candidate}")
    return candidate


def validate_video_path(video_path: Any) -> Path:
    path = _resolve_path(video_path, label="Video")
    if not path.is_file():
        raise VideoToolValidationError(f"Video path is not a file: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        supported = ", ".join(sorted(VIDEO_SUFFIXES))
        raise VideoToolValidationError(
            f"Unsupported video suffix {path.suffix!r}; supported: {supported}"
        )
    return path


def validate_json_path(json_path: Any) -> Path:
    path = _resolve_path(json_path, label="YOLO event JSON")
    if not path.is_file() or path.suffix.lower() != ".json":
        raise VideoToolValidationError(f"Expected a JSON file: {path}")
    return path


def resolve_output_dir(output_dir: Any, video_path: Path) -> Path:
    if output_dir is None:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._")
        output = DEFAULT_OUTPUT_ROOT / (safe_stem or "video")
    else:
        output = _resolve_path(output_dir, label="Output directory", must_exist=False)
    if output.exists() and not output.is_dir():
        raise VideoToolValidationError(f"Output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


@dataclass(frozen=True)
class VideoYoloSettings:
    """Settings reproduced from video_yolo_skip_frame_detection.ipynb."""

    checkpoint: Path | str | None = None
    device: str = "auto"
    confidence: float = 0.25
    nms_iou: float = 0.70
    image_size: int | None = None
    coarse_stride: int = 7
    match_iou: float = 0.50
    confirm_frames: int = 3
    missing_patience: int = 3
    box_expand: float = 10.0

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou must be in [0, 1]")
        if not 0 <= self.match_iou <= 1:
            raise ValueError("match_iou must be in [0, 1]")
        if self.coarse_stride < 1:
            raise ValueError("coarse_stride must be >= 1")
        if self.confirm_frames < 1 or self.missing_patience < 1:
            raise ValueError("confirm_frames and missing_patience must be >= 1")


@dataclass(frozen=True)
class VideoSamSettings:
    """Settings reproduced from video_defect_detection_yolo_sam2.ipynb."""

    checkpoint: Path | str | None = None
    sam_config: str = DEFAULT_SAM_CONFIG
    device: str = "auto"
    output_video_name: str = "defect_mask_tracking.mp4"
    mask_alpha: float = 0.45
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True
    keep_extracted_frames: bool = False
    jpeg_quality: int = 95

    def __post_init__(self) -> None:
        if not 0 <= self.mask_alpha <= 1:
            raise ValueError("mask_alpha must be in [0, 1]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        output_name = Path(self.output_video_name)
        if output_name.name != self.output_video_name or output_name.suffix.lower() != ".mp4":
            raise ValueError("output_video_name must be a plain .mp4 filename")


def _compact_event(event: Any, fps: float) -> dict[str, Any]:
    duration_frames = int(event.end_frame) - int(event.start_frame) + 1
    return {
        "event_id": int(event.event_id),
        "class_id": int(event.class_id),
        "class_name": str(event.class_name),
        "start_frame": int(event.start_frame),
        "confirmation_frame": int(event.confirmation_frame),
        "end_frame": int(event.end_frame),
        "disappear_frame": int(event.disappear_frame),
        "start_time_seconds": event.start_time_seconds,
        "start_timecode": event.start_timecode,
        "disappear_time_seconds": event.disappear_time_seconds,
        "disappear_timecode": event.disappear_timecode,
        "duration_frames": duration_frames,
        "duration_seconds": round(duration_frames / fps, 3),
        "detection_frame_count": len(event.detections),
        "first_confidence": round(float(event.first_confidence), 6),
        "max_confidence": round(float(event.max_confidence), 6),
        "mean_confidence": round(float(event.mean_confidence), 6),
        "prompt_box_xyxy": [round(float(value), 2) for value in event.prompt_box_xyxy],
    }


def _class_counts(events: Sequence[Any]) -> dict[str, int]:
    counts = Counter(str(event.class_name) for event in events)
    return dict(sorted(counts.items()))


class VideoYoloDetectionTool:
    """Resident temporal YOLO detector returning compact agent-facing JSON."""

    def __init__(
        self,
        settings: VideoYoloSettings | None = None,
        *,
        eager_load: bool = False,
    ) -> None:
        self.settings = settings or VideoYoloSettings()
        self._detector: Any | None = None
        self._lock = threading.RLock()
        if eager_load:
            self._get_detector()

    def _get_detector(self) -> Any:
        if self._detector is None:
            checkpoint = resolve_yolo_checkpoint(self.settings.checkpoint)
            pipeline = _load_video_pipeline()
            self._detector = pipeline.TemporalYoloOBBDetector(
                checkpoint=checkpoint,
                device=self.settings.device,
                confidence=self.settings.confidence,
                nms_iou=self.settings.nms_iou,
                image_size=self.settings.image_size,
                coarse_stride=self.settings.coarse_stride,
                match_iou=self.settings.match_iou,
                confirm_frames=self.settings.confirm_frames,
                missing_patience=self.settings.missing_patience,
                box_expand=self.settings.box_expand,
            )
        return self._detector

    def detect(
        self,
        video_path: Path | str,
        *,
        output_dir: Path | str | None = None,
        progress: bool = True,
    ) -> dict[str, Any]:
        video = validate_video_path(video_path)
        output = resolve_output_dir(output_dir, video)
        pipeline = _load_video_pipeline()

        with self._lock:
            detector = self._get_detector()
            events, report = detector.detect_events(video, progress=progress)
            json_path, csv_path = pipeline.save_yolo_results(output, events, report)

        fps = float(report["video"]["fps"])
        compact_events = [_compact_event(event, fps) for event in events]
        return {
            "success": True,
            "stage": "video_yolo_detection",
            "video_path": str(video),
            "defect_exists": bool(events),
            # A video defect count is the number of temporally confirmed events,
            # not the sum of per-frame bounding boxes.
            "defect_count": len(events),
            "event_count": len(events),
            "class_counts": _class_counts(events),
            "events": compact_events,
            "video": report["video"],
            "inferred_frame_count": int(report["inferred_frame_count"]),
            "settings": report["settings"],
            "artifacts": {
                "yolo_events_json_path": str(json_path),
                "yolo_events_csv_path": str(csv_path),
            },
        }


class VideoSamRenderingTool:
    """Resident SAM2 renderer consuming a persisted YOLO event JSON."""

    def __init__(
        self,
        settings: VideoSamSettings | None = None,
        *,
        eager_load: bool = False,
    ) -> None:
        self.settings = settings or VideoSamSettings()
        self._predictor: Any | None = None
        self._resolved_checkpoint: Path | None = None
        self._lock = threading.RLock()
        if eager_load:
            self._get_predictor()

    def _get_predictor(self) -> Any:
        if self._predictor is None:
            checkpoint = resolve_sam_checkpoint(self.settings.checkpoint)
            self._resolved_checkpoint = checkpoint
            pipeline = _load_video_pipeline()
            self._predictor = pipeline.build_video_predictor(
                sam_config=self.settings.sam_config,
                sam_checkpoint=checkpoint,
                device=self.settings.device,
            )
        return self._predictor

    def render(
        self,
        video_path: Path | str,
        yolo_events_json_path: Path | str,
        *,
        output_dir: Path | str | None = None,
        output_video_name: str | None = None,
    ) -> dict[str, Any]:
        video = validate_video_path(video_path)
        json_path = validate_json_path(yolo_events_json_path)
        output = resolve_output_dir(output_dir or json_path.parent, video)
        pipeline = _load_video_pipeline()
        events, report = pipeline.load_yolo_results(json_path)

        recorded_video = report.get("video", {}).get("path")
        if recorded_video and Path(recorded_video).resolve() != video:
            raise VideoToolValidationError(
                "YOLO event JSON belongs to a different source video: "
                f"{recorded_video}"
            )

        if not events:
            return {
                "success": True,
                "stage": "video_sam_rendering",
                "video_path": str(video),
                "skipped": True,
                "skip_reason": "No confirmed YOLO defect events",
                "defect_exists": False,
                "defect_count": 0,
                "event_count": 0,
                "class_counts": {},
                "result_video_path": None,
                "artifacts": {
                    "yolo_events_json_path": str(json_path),
                },
            }

        name = output_video_name or self.settings.output_video_name
        name_path = Path(name)
        if name_path.name != name or name_path.suffix.lower() != ".mp4":
            raise VideoToolValidationError(
                "output_video_name must be a plain .mp4 filename"
            )

        with self._lock:
            predictor = self._get_predictor()
            metadata = pipeline.track_events_with_sam(
                video_path=video,
                events=events,
                predictor=predictor,
                output_dir=output,
                output_video_name=name,
                mask_alpha=self.settings.mask_alpha,
                offload_video_to_cpu=self.settings.offload_video_to_cpu,
                offload_state_to_cpu=self.settings.offload_state_to_cpu,
                keep_extracted_frames=self.settings.keep_extracted_frames,
                jpeg_quality=self.settings.jpeg_quality,
            )

        fps = float(report["video"]["fps"])
        checkpoint = self._resolved_checkpoint or resolve_sam_checkpoint(
            self.settings.checkpoint
        )
        return {
            "success": True,
            "stage": "video_sam_rendering",
            "video_path": str(video),
            "skipped": False,
            "defect_exists": True,
            "defect_count": len(events),
            "event_count": len(events),
            "class_counts": _class_counts(events),
            "events": [_compact_event(event, fps) for event in events],
            "result_video_path": metadata["output_video"],
            "groups": metadata["groups"],
            "audio_preserved": bool(metadata["audio_preserved"]),
            "settings": {
                "checkpoint": str(checkpoint),
                "sam_config": self.settings.sam_config,
                "device": str(predictor.device),
                "mask_alpha": self.settings.mask_alpha,
                "offload_video_to_cpu": self.settings.offload_video_to_cpu,
                "offload_state_to_cpu": self.settings.offload_state_to_cpu,
                "keep_extracted_frames": self.settings.keep_extracted_frames,
                "jpeg_quality": self.settings.jpeg_quality,
            },
            "artifacts": {
                "yolo_events_json_path": str(json_path),
                "sam_metadata_json_path": str(output / "sam_mask_metadata.json"),
                "result_video_path": metadata["output_video"],
            },
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    yolo = subparsers.add_parser("yolo", help="Run temporal YOLO detection")
    yolo.add_argument("video_path", type=Path)
    yolo.add_argument("--output-dir", type=Path)
    yolo.add_argument("--checkpoint", type=Path)
    yolo.add_argument("--device", default="auto")
    yolo.add_argument("--confidence", type=float, default=0.25)
    yolo.add_argument("--nms-iou", type=float, default=0.70)
    yolo.add_argument("--coarse-stride", type=int, default=7)
    yolo.add_argument("--match-iou", type=float, default=0.50)
    yolo.add_argument("--confirm-frames", type=int, default=3)
    yolo.add_argument("--missing-patience", type=int, default=3)
    yolo.add_argument("--box-expand", type=float, default=10.0)
    yolo.add_argument("--quiet", action="store_true")

    sam = subparsers.add_parser("sam", help="Render persisted YOLO events with SAM2")
    sam.add_argument("video_path", type=Path)
    sam.add_argument("yolo_events_json_path", type=Path)
    sam.add_argument("--output-dir", type=Path)
    sam.add_argument("--checkpoint", type=Path)
    sam.add_argument("--device", default="auto")
    sam.add_argument("--sam-config", default=DEFAULT_SAM_CONFIG)
    sam.add_argument("--output-video-name", default="defect_mask_tracking.mp4")
    sam.add_argument("--mask-alpha", type=float, default=0.45)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "yolo":
        settings = VideoYoloSettings(
            checkpoint=args.checkpoint,
            device=args.device,
            confidence=args.confidence,
            nms_iou=args.nms_iou,
            coarse_stride=args.coarse_stride,
            match_iou=args.match_iou,
            confirm_frames=args.confirm_frames,
            missing_patience=args.missing_patience,
            box_expand=args.box_expand,
        )
        result = VideoYoloDetectionTool(settings).detect(
            args.video_path,
            output_dir=args.output_dir,
            progress=not args.quiet,
        )
    else:
        settings = VideoSamSettings(
            checkpoint=args.checkpoint,
            sam_config=args.sam_config,
            device=args.device,
            output_video_name=args.output_video_name,
            mask_alpha=args.mask_alpha,
        )
        result = VideoSamRenderingTool(settings).render(
            args.video_path,
            args.yolo_events_json_path,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
