"""YOLO26-OBB + SAM2 video defect detection and tracking pipeline.

The module is intentionally notebook-friendly: each expensive stage has a small,
serializable result and can be called independently.  YOLO performs a coarse scan
and switches to dense, bidirectional per-frame inspection around every detection.
SAM2 then tracks only the confirmed defect intervals, avoiding the memory cost of
loading an entire long video into ``SAM2VideoPredictor.init_state``.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
LANGGRAPH_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = LANGGRAPH_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ultralytics otherwise writes to a per-user settings directory, which may be
# unavailable in managed Windows environments.
ULTRALYTICS_CONFIG_ROOT = SCRIPT_DIR / ".ultralytics"
(ULTRALYTICS_CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_ROOT))

from sam2.build_sam import build_sam2_video_predictor  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from langGraph.model_paths import (  # noqa: E402
    SAM_MODEL_DIR,
    YOLO_MODEL_DIR,
    resolve_sam_checkpoint,
    resolve_yolo_checkpoint,
)


DEFAULT_VIDEO_PATH = (
    PROJECT_ROOT
    / "vos_dataset"
    / "video"
    / "fdm_3d_printing_defect.mp4"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs" / "video_defect"
DEFAULT_YOLO_CHECKPOINT = YOLO_MODEL_DIR
DEFAULT_SAM_CHECKPOINT = SAM_MODEL_DIR
DEFAULT_SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_b+"

COLORS_RGB = (
    (255, 82, 82),
    (66, 165, 245),
    (102, 187, 106),
    (255, 167, 38),
    (171, 71, 188),
    (38, 198, 218),
    (255, 238, 88),
    (236, 64, 122),
)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _fourcc_text(value: int) -> str:
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


@dataclass
class VideoInfo:
    path: str
    container: str
    codec_fourcc: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
    size_bytes: int
    size_mib: float


def inspect_video(video_path: Path | str) -> VideoInfo:
    """Read container, timing, frame-count, geometry and file-size metadata."""

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = _fourcc_text(int(capture.get(cv2.CAP_PROP_FOURCC)))
    finally:
        capture.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Invalid video metadata: fps={fps}, frames={frame_count}, size={width}x{height}"
        )
    size_bytes = path.stat().st_size
    return VideoInfo(
        path=str(path),
        container=path.suffix.lower().lstrip("."),
        codec_fourcc=fourcc,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps,
        width=width,
        height=height,
        size_bytes=size_bytes,
        size_mib=size_bytes / (1024**2),
    )


@dataclass
class FrameDetection:
    frame_idx: int
    class_id: int
    class_name: str
    confidence: float
    obb_points: list[list[float]]
    box_xyxy: list[float]


@dataclass
class DefectEvent:
    event_id: int
    ann_obj_id: int
    class_id: int
    class_name: str
    start_frame: int
    confirmation_frame: int
    end_frame: int
    disappear_frame: int
    disappearance_confirmation_frame: int | None
    disappearance_confirmed: bool
    prompt_frame: int
    prompt_box_xyxy: list[float]
    prompt_obb_points: list[list[float]]
    first_confidence: float
    max_confidence: float
    mean_confidence: float
    detection_frames: list[int]
    detections: list[FrameDetection] = field(repr=False)
    start_time_seconds: float | None = None
    start_timecode: str | None = None
    disappear_time_seconds: float | None = None
    disappear_timecode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["detections"] = [asdict(item) for item in self.detections]
        return output


def box_iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Axis-aligned IoU for two ``[x1, y1, x2, y2]`` boxes."""

    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def frame_to_time(frame_idx: int, fps: float) -> tuple[float, str]:
    """Convert a zero-based frame index to seconds and ``HH:MM:SS.mmm``."""

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    seconds = float(frame_idx) / float(fps)
    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    timecode = f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"
    return seconds, timecode


def _populate_event_times(event: DefectEvent, fps: float) -> None:
    start_seconds, start_timecode = frame_to_time(event.start_frame, fps)
    disappear_seconds, disappear_timecode = frame_to_time(event.disappear_frame, fps)
    event.start_time_seconds = start_seconds
    event.start_timecode = start_timecode
    event.disappear_time_seconds = disappear_seconds
    event.disappear_timecode = disappear_timecode


class VideoFrameReader:
    """Small random-access OpenCV reader; frames are returned in BGR order."""

    def __init__(self, video_path: Path | str) -> None:
        self.path = Path(video_path).expanduser().resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next_idx = 0

    def read(self, frame_idx: int) -> np.ndarray:
        if not 0 <= frame_idx < self.frame_count:
            raise IndexError(f"frame_idx {frame_idx} outside [0, {self.frame_count - 1}]")
        if frame_idx != self._next_idx:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to decode frame {frame_idx} from {self.path}")
        self._next_idx = frame_idx + 1
        return frame

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoFrameReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class TemporalYoloOBBDetector:
    """Coarse YOLO scan plus dense temporal verification around every hit.

    A detection is considered the same instance only when the class is equal and
    adjacent matched boxes reach ``match_iou``.  An event becomes valid after
    ``confirm_frames`` consecutive detections.  Up to ``missing_patience - 1``
    isolated YOLO misses are tolerated; ``missing_patience`` misses confirm that
    the object disappeared.
    """

    def __init__(
        self,
        checkpoint: Path | str = DEFAULT_YOLO_CHECKPOINT,
        device: str = "auto",
        confidence: float = 0.25,
        nms_iou: float = 0.7,
        image_size: int | None = None,
        coarse_stride: int = 15,
        match_iou: float = 0.5,
        confirm_frames: int = 3,
        missing_patience: int = 3,
        box_expand: float = 10.0,
    ) -> None:
        checkpoint = resolve_yolo_checkpoint(checkpoint)
        if coarse_stride < 1:
            raise ValueError("coarse_stride must be >= 1")
        if confirm_frames < 1 or missing_patience < 1:
            raise ValueError("confirm_frames and missing_patience must be >= 1")
        if not 0 <= match_iou <= 1:
            raise ValueError("match_iou must be in [0, 1]")

        self.checkpoint = checkpoint
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
            if device == "auto"
            else device
        )
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.image_size = image_size
        self.coarse_stride = coarse_stride
        self.match_iou = match_iou
        self.confirm_frames = confirm_frames
        self.missing_patience = missing_patience
        self.box_expand = box_expand
        self.model = YOLO(str(checkpoint), task="obb")
        if self.model.task != "obb":
            raise ValueError(f"Expected an OBB model, got task={self.model.task!r}")

        self._cache: dict[int, list[FrameDetection]] = {}
        self._reader: VideoFrameReader | None = None
        self._video_info: VideoInfo | None = None
        self.coarse_frames: list[int] = []

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "device": self.device,
            "confidence": self.confidence,
            "nms_iou": self.nms_iou,
            "image_size": self.image_size,
            "coarse_stride": self.coarse_stride,
            "match_iou": self.match_iou,
            "confirm_frames": self.confirm_frames,
            "missing_patience": self.missing_patience,
            "box_expand": self.box_expand,
        }

    def _predict_frame(self, frame_idx: int) -> list[FrameDetection]:
        if frame_idx in self._cache:
            return self._cache[frame_idx]
        if self._reader is None or self._video_info is None:
            raise RuntimeError("detect_events must initialize the video reader")

        frame_bgr = self._reader.read(frame_idx)
        kwargs: dict[str, Any] = {
            "source": frame_bgr,
            "conf": self.confidence,
            "iou": self.nms_iou,
            "device": self.device,
            "verbose": False,
        }
        if self.image_size is not None:
            kwargs["imgsz"] = self.image_size
        result = self.model.predict(**kwargs)[0]
        obb = getattr(result, "obb", None)
        detections: list[FrameDetection] = []
        if obb is not None and len(obb) > 0:
            polygons = obb.xyxyxyxy.detach().cpu().numpy().astype(np.float32)
            confidences = obb.conf.detach().cpu().numpy().astype(np.float32)
            class_ids = obb.cls.detach().cpu().numpy().astype(np.int64)
            max_x = float(self._video_info.width - 1)
            max_y = float(self._video_info.height - 1)
            for polygon, confidence, class_id in zip(polygons, confidences, class_ids):
                x1 = float(np.clip(polygon[:, 0].min() - self.box_expand, 0, max_x))
                y1 = float(np.clip(polygon[:, 1].min() - self.box_expand, 0, max_y))
                x2 = float(np.clip(polygon[:, 0].max() + self.box_expand, 0, max_x))
                y2 = float(np.clip(polygon[:, 1].max() + self.box_expand, 0, max_y))
                detections.append(
                    FrameDetection(
                        frame_idx=frame_idx,
                        class_id=int(class_id),
                        class_name=_class_name(result.names, int(class_id)),
                        confidence=float(confidence),
                        obb_points=polygon.astype(float).tolist(),
                        box_xyxy=[x1, y1, x2, y2],
                    )
                )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        self._cache[frame_idx] = detections
        return detections

    def detect_frames(
        self,
        video_path: Path | str,
        frame_indices: Sequence[int],
    ) -> dict[int, list[FrameDetection]]:
        """Run YOLO on explicitly selected frames and return per-frame detections.

        Results already produced by ``detect_events`` on the same video are reused.
        This public method is intended for notebook inspection and avoids exposing
        the detector's temporary OpenCV reader state.
        """

        video_info = inspect_video(video_path)
        normalized_indices = sorted({int(index) for index in frame_indices})
        invalid = [
            index
            for index in normalized_indices
            if not 0 <= index < video_info.frame_count
        ]
        if invalid:
            raise IndexError(
                f"Frame indices outside [0, {video_info.frame_count - 1}]: {invalid}"
            )
        if self._reader is not None:
            raise RuntimeError("detect_frames cannot run while another video scan is active")

        same_video = (
            self._video_info is not None
            and Path(self._video_info.path).resolve() == Path(video_info.path).resolve()
        )
        if not same_video:
            self._cache.clear()
        self._video_info = video_info
        self._reader = VideoFrameReader(video_path)
        try:
            return {index: list(self._predict_frame(index)) for index in normalized_indices}
        finally:
            self._reader.close()
            self._reader = None

    def _best_match(
        self,
        reference: FrameDetection,
        candidates: Sequence[FrameDetection],
    ) -> FrameDetection | None:
        eligible = [
            (box_iou_xyxy(reference.box_xyxy, item.box_xyxy), item)
            for item in candidates
            if item.class_id == reference.class_id
        ]
        eligible = [pair for pair in eligible if pair[0] >= self.match_iou]
        if not eligible:
            return None
        return max(eligible, key=lambda pair: (pair[0], pair[1].confidence))[1]

    def _trace_direction(
        self,
        seed: FrameDetection,
        direction: int,
    ) -> tuple[dict[int, FrameDetection], int | None, bool]:
        assert direction in (-1, 1)
        assert self._video_info is not None
        matches: dict[int, FrameDetection] = {}
        reference = seed
        misses = 0
        frame_idx = seed.frame_idx + direction
        disappearance_confirmation: int | None = None
        boundary_confirmed = False

        while 0 <= frame_idx < self._video_info.frame_count:
            match = self._best_match(reference, self._predict_frame(frame_idx))
            if match is None:
                misses += 1
                if misses >= self.missing_patience:
                    disappearance_confirmation = frame_idx
                    boundary_confirmed = True
                    break
            else:
                matches[frame_idx] = match
                reference = match
                misses = 0
            frame_idx += direction
        return matches, disappearance_confirmation, boundary_confirmed

    def _already_covered(
        self,
        seed: FrameDetection,
        event_detections: Sequence[dict[int, FrameDetection]],
    ) -> bool:
        for track in event_detections:
            existing = track.get(seed.frame_idx)
            if (
                existing is not None
                and existing.class_id == seed.class_id
                and box_iou_xyxy(existing.box_xyxy, seed.box_xyxy) >= self.match_iou
            ):
                return True
        return False

    def _first_consecutive_run(self, frames: Sequence[int]) -> tuple[int, int] | None:
        if not frames:
            return None
        run_start = frames[0]
        previous = frames[0]
        for current in list(frames[1:]) + [math.inf]:
            if current == previous + 1:
                previous = int(current)
                continue
            if previous - run_start + 1 >= self.confirm_frames:
                return run_start, run_start + self.confirm_frames - 1
            run_start = int(current) if current != math.inf else 0
            previous = int(current) if current != math.inf else 0
        return None

    def _make_event(
        self,
        track: dict[int, FrameDetection],
        forward_confirmation: int | None,
        forward_confirmed: bool,
        event_id: int,
    ) -> DefectEvent | None:
        assert self._video_info is not None
        frames = sorted(track)
        run = self._first_consecutive_run(frames)
        if run is None:
            return None
        start_frame, confirmation_frame = run

        # Discard isolated detections preceding the first confirmed run.  After
        # confirmation, one or two YOLO misses remain part of the same lifecycle.
        retained = {idx: det for idx, det in track.items() if idx >= start_frame}
        retained_frames = sorted(retained)
        end_frame = retained_frames[-1]
        disappear_frame = min(end_frame + 1, self._video_info.frame_count)
        detections = [retained[idx] for idx in retained_frames]
        prompt = retained[start_frame]
        confidences = [item.confidence for item in detections]
        event = DefectEvent(
            event_id=event_id,
            ann_obj_id=event_id,
            class_id=prompt.class_id,
            class_name=prompt.class_name,
            start_frame=start_frame,
            confirmation_frame=confirmation_frame,
            end_frame=end_frame,
            disappear_frame=disappear_frame,
            disappearance_confirmation_frame=forward_confirmation,
            disappearance_confirmed=forward_confirmed,
            prompt_frame=start_frame,
            prompt_box_xyxy=prompt.box_xyxy,
            prompt_obb_points=prompt.obb_points,
            first_confidence=prompt.confidence,
            max_confidence=max(confidences),
            mean_confidence=float(np.mean(confidences)),
            detection_frames=retained_frames,
            detections=detections,
        )
        _populate_event_times(event, self._video_info.fps)
        return event

    def _tracks_are_duplicates(
        self,
        left: dict[int, FrameDetection],
        right: dict[int, FrameDetection],
    ) -> bool:
        """Return True when two seed traces represent the same temporal instance."""

        if not left or not right:
            return False
        left_class = next(iter(left.values())).class_id
        right_class = next(iter(right.values())).class_id
        if left_class != right_class:
            return False
        common_frames = set(left).intersection(right)
        return any(
            box_iou_xyxy(left[idx].box_xyxy, right[idx].box_xyxy) >= self.match_iou
            for idx in common_frames
        )

    def _merge_duplicate_tracks(
        self,
        tracks: Sequence[dict[int, FrameDetection]],
        track_meta: Sequence[tuple[int | None, bool]],
    ) -> tuple[list[dict[int, FrameDetection]], list[tuple[int | None, bool]]]:
        """Merge duplicate bidirectional traces while retaining separate instances."""

        merged_tracks: list[dict[int, FrameDetection]] = []
        merged_meta: list[tuple[int | None, bool]] = []
        for incoming, incoming_meta in zip(tracks, track_meta):
            duplicate_idx = next(
                (
                    idx
                    for idx, existing in enumerate(merged_tracks)
                    if self._tracks_are_duplicates(existing, incoming)
                ),
                None,
            )
            if duplicate_idx is None:
                merged_tracks.append(dict(incoming))
                merged_meta.append(incoming_meta)
                continue

            existing = merged_tracks[duplicate_idx]
            existing_last = max(existing)
            incoming_last = max(incoming)
            for frame_idx, detection in incoming.items():
                previous = existing.get(frame_idx)
                if previous is None or detection.confidence > previous.confidence:
                    existing[frame_idx] = detection
            # The trace reaching furthest forward owns the disappearance evidence.
            if incoming_last > existing_last:
                merged_meta[duplicate_idx] = incoming_meta

        # A merge can bridge two earlier traces, so repeat until the relation is stable.
        changed = True
        while changed:
            changed = False
            for left_idx in range(len(merged_tracks)):
                for right_idx in range(left_idx + 1, len(merged_tracks)):
                    if not self._tracks_are_duplicates(
                        merged_tracks[left_idx], merged_tracks[right_idx]
                    ):
                        continue
                    left_last = max(merged_tracks[left_idx])
                    right_last = max(merged_tracks[right_idx])
                    for frame_idx, detection in merged_tracks[right_idx].items():
                        previous = merged_tracks[left_idx].get(frame_idx)
                        if previous is None or detection.confidence > previous.confidence:
                            merged_tracks[left_idx][frame_idx] = detection
                    if right_last > left_last:
                        merged_meta[left_idx] = merged_meta[right_idx]
                    merged_tracks.pop(right_idx)
                    merged_meta.pop(right_idx)
                    changed = True
                    break
                if changed:
                    break
        return merged_tracks, merged_meta

    def detect_events(
        self,
        video_path: Path | str,
        progress: bool = True,
    ) -> tuple[list[DefectEvent], dict[str, Any]]:
        """Detect and temporally validate all defect instances in a video."""

        self._cache.clear()
        self.coarse_frames.clear()
        self._video_info = inspect_video(video_path)
        self._reader = VideoFrameReader(video_path)
        tracks: list[dict[int, FrameDetection]] = []
        track_meta: list[tuple[int | None, bool]] = []

        try:
            coarse_indices = list(range(0, self._video_info.frame_count, self.coarse_stride))
            if coarse_indices[-1] != self._video_info.frame_count - 1:
                coarse_indices.append(self._video_info.frame_count - 1)

            for coarse_number, frame_idx in enumerate(coarse_indices, start=1):
                self.coarse_frames.append(frame_idx)
                detections = self._predict_frame(frame_idx)
                if progress and (detections or coarse_number % 20 == 0):
                    print(
                        f"YOLO coarse [{coarse_number}/{len(coarse_indices)}] "
                        f"frame={frame_idx}, detections={len(detections)}, "
                        f"dense_frames_cached={len(self._cache)}"
                    )
                for seed in detections:
                    if self._already_covered(seed, tracks):
                        continue
                    backward, _, _ = self._trace_direction(seed, direction=-1)
                    forward, forward_confirmation, forward_confirmed = self._trace_direction(
                        seed, direction=1
                    )
                    track = {seed.frame_idx: seed, **backward, **forward}
                    tracks.append(track)
                    track_meta.append((forward_confirmation, forward_confirmed))
        finally:
            self._reader.close()
            self._reader = None

        tracks, track_meta = self._merge_duplicate_tracks(tracks, track_meta)
        events: list[DefectEvent] = []
        for track, (confirmation, confirmed) in zip(tracks, track_meta):
            event = self._make_event(track, confirmation, confirmed, event_id=len(events) + 1)
            if event is not None:
                events.append(event)

        events.sort(key=lambda item: (item.start_frame, item.class_id, item.ann_obj_id))
        # Reassign stable chronological IDs after rejecting unconfirmed tracks.
        for event_id, event in enumerate(events, start=1):
            event.event_id = event_id
            event.ann_obj_id = event_id

        report = {
            "video": asdict(self._video_info),
            "settings": self.settings,
            "coarse_frames": self.coarse_frames,
            "inferred_frame_count": len(self._cache),
            "confirmed_event_count": len(events),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return events, report


def save_yolo_results(
    output_dir: Path | str,
    events: Sequence[DefectEvent],
    report: dict[str, Any],
) -> tuple[Path, Path]:
    """Save complete YOLO temporal metadata as JSON and an event-level CSV."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "yolo_defect_events.json"
    csv_path = output_dir / "yolo_defect_events.csv"
    fps = float(report["video"]["fps"])
    for event in events:
        _populate_event_times(event, fps)
    payload = {**report, "events": [event.to_dict() for event in events]}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    columns = [
        "event_id",
        "ann_obj_id",
        "class_id",
        "class_name",
        "start_frame",
        "start_time_seconds",
        "start_timecode",
        "confirmation_frame",
        "end_frame",
        "disappear_frame",
        "disappear_time_seconds",
        "disappear_timecode",
        "disappearance_confirmation_frame",
        "disappearance_confirmed",
        "first_confidence",
        "max_confidence",
        "mean_confidence",
        "prompt_box_xyxy",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        import csv

        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for event in events:
            row = {key: getattr(event, key) for key in columns}
            row["prompt_box_xyxy"] = json.dumps(event.prompt_box_xyxy)
            writer.writerow(row)
    return json_path, csv_path


def load_yolo_results(
    json_path: Path | str,
) -> tuple[list[DefectEvent], dict[str, Any]]:
    """Restore events/report so SAM can be rerun without repeating YOLO inference."""

    path = Path(json_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    events: list[DefectEvent] = []
    for raw_event in payload.get("events", []):
        raw_event = dict(raw_event)
        raw_event["detections"] = [
            FrameDetection(**item) for item in raw_event.get("detections", [])
        ]
        events.append(DefectEvent(**raw_event))
    report = {key: value for key, value in payload.items() if key != "events"}
    return events, report


@dataclass
class EventGroup:
    start_frame: int
    end_frame: int
    events: list[DefectEvent]


def group_overlapping_events(events: Sequence[DefectEvent]) -> list[EventGroup]:
    """Merge overlapping lifecycles so concurrent objects share one SAM state."""

    groups: list[EventGroup] = []
    for event in sorted(events, key=lambda item: (item.start_frame, item.end_frame)):
        if not groups or event.start_frame > groups[-1].end_frame:
            groups.append(EventGroup(event.start_frame, event.end_frame, [event]))
        else:
            groups[-1].end_frame = max(groups[-1].end_frame, event.end_frame)
            groups[-1].events.append(event)
    return groups


def extract_frame_range(
    video_path: Path | str,
    start_frame: int,
    end_frame: int,
    output_dir: Path | str,
    jpeg_quality: int = 95,
) -> Path:
    """Extract an inclusive frame interval as SAM-compatible numbered JPEGs."""

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with VideoFrameReader(video_path) as reader:
        for local_idx, global_idx in enumerate(range(start_frame, end_frame + 1)):
            frame = reader.read(global_idx)
            path = output_dir / f"{local_idx:06d}.jpg"
            if not cv2.imwrite(
                str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            ):
                raise RuntimeError(f"Failed to write extracted frame: {path}")
    return output_dir


def build_video_predictor(
    sam_config: str = DEFAULT_SAM_CONFIG,
    sam_checkpoint: Path | str = DEFAULT_SAM_CHECKPOINT,
    device: str = "auto",
) -> Any:
    """Build the fine-tuned SAM2 video predictor used by the notebook."""

    checkpoint = resolve_sam_checkpoint(sam_checkpoint)
    resolved_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        if device == "auto"
        else device
    )
    return build_sam2_video_predictor(
        sam_config,
        str(checkpoint),
        device=resolved_device,
        # This also protects correction prompts if the notebook is extended to
        # allow manual refinement after the first propagation.
        hydra_overrides_extra=["++model.add_all_frames_to_correct_as_cond=true"],
    )


def _autocast_context(device: torch.device) -> Any:
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _overlay_masks_bgr(
    frame_bgr: np.ndarray,
    obj_ids: Sequence[int],
    masks: np.ndarray,
    event_by_obj_id: dict[int, DefectEvent],
    alpha: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    canvas = frame_bgr.copy()
    frame_records: list[dict[str, Any]] = []
    for index, obj_id in enumerate(obj_ids):
        event = event_by_obj_id.get(int(obj_id))
        if event is None:
            continue
        mask = np.asarray(masks[index]).squeeze().astype(bool)
        color_rgb = COLORS_RGB[(event.event_id - 1) % len(COLORS_RGB)]
        color_bgr = np.asarray(color_rgb[::-1], dtype=np.float32)
        if np.any(mask):
            canvas[mask] = np.clip(
                canvas[mask].astype(np.float32) * (1 - alpha) + color_bgr * alpha,
                0,
                255,
            ).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            mask_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            area = int(mask.sum())
            cv2.rectangle(
                canvas,
                (mask_box[0], mask_box[1]),
                (mask_box[2], mask_box[3]),
                tuple(map(int, color_bgr)),
                2,
                cv2.LINE_AA,
            )
            anchor = (mask_box[0], max(22, mask_box[1] - 7))
        else:
            mask_box = None
            area = 0
            anchor = (10, 28 + 26 * index)
        label = f"ID {event.event_id} {event.class_name} YOLO {event.first_confidence:.2f}"
        cv2.putText(
            canvas,
            label,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            tuple(map(int, color_bgr)),
            2,
            cv2.LINE_AA,
        )
        frame_records.append(
            {
                "obj_id": int(obj_id),
                "event_id": event.event_id,
                "class_id": event.class_id,
                "class_name": event.class_name,
                "mask_area": area,
                "mask_box_xyxy": mask_box,
            }
        )
    return canvas, frame_records


def _write_until(
    reader: VideoFrameReader,
    writer: cv2.VideoWriter,
    next_frame: int,
    stop_before: int,
) -> int:
    while next_frame < stop_before:
        writer.write(reader.read(next_frame))
        next_frame += 1
    return next_frame


def track_events_with_sam(
    video_path: Path | str,
    events: Sequence[DefectEvent],
    predictor: Any,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    output_video_name: str = "defect_mask_tracking.mp4",
    mask_alpha: float = 0.45,
    offload_video_to_cpu: bool = True,
    offload_state_to_cpu: bool = True,
    keep_extracted_frames: bool = False,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """Track confirmed events with SAM and render a mask-overlay MP4.

    Only each confirmed event group is loaded into SAM.  Concurrent events are
    represented by distinct ``ann_obj_id`` values; objects are removed immediately
    after their YOLO lifecycle ends.
    """

    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_cache_root = output_dir / "sam_frame_cache"
    frame_cache_root.mkdir(parents=True, exist_ok=True)
    video_info = inspect_video(video_path)
    output_video_path = output_dir / output_video_name
    mask_metadata_path = output_dir / "sam_mask_metadata.json"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        video_info.fps,
        (video_info.width, video_info.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_video_path}")

    groups = group_overlapping_events(events)
    event_by_obj_id = {event.ann_obj_id: event for event in events}
    mask_records: list[dict[str, Any]] = []
    next_frame_to_write = 0
    source_reader = VideoFrameReader(video_path)
    compute_device = predictor.device

    try:
        for group_index, group in enumerate(groups, start=1):
            print(
                f"SAM group [{group_index}/{len(groups)}] "
                f"frames={group.start_frame}..{group.end_frame}, "
                f"events={[item.event_id for item in group.events]}"
            )
            next_frame_to_write = _write_until(
                source_reader, writer, next_frame_to_write, group.start_frame
            )

            group_dir = frame_cache_root / f"group_{group_index:03d}"
            extract_frame_range(
                video_path,
                group.start_frame,
                group.end_frame,
                group_dir,
                jpeg_quality=jpeg_quality,
            )
            state = predictor.init_state(
                video_path=str(group_dir),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                # Keep False: the current async loader retains float64 JPEG tensors.
                async_loading_frames=False,
            )

            starts: dict[int, list[DefectEvent]] = {}
            removals: dict[int, list[DefectEvent]] = {}
            for event in group.events:
                starts.setdefault(event.start_frame, []).append(event)
                removals.setdefault(event.end_frame + 1, []).append(event)

            boundaries = sorted(
                {group.start_frame, group.end_frame + 1, *starts.keys(), *removals.keys()}
            )
            active: set[int] = set()
            for boundary_index, boundary in enumerate(boundaries[:-1]):
                for event in removals.get(boundary, []):
                    if event.ann_obj_id in active:
                        predictor.remove_object(
                            state, event.ann_obj_id, strict=True, need_output=False
                        )
                        active.remove(event.ann_obj_id)
                for event in starts.get(boundary, []):
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=event.prompt_frame - group.start_frame,
                        obj_id=event.ann_obj_id,
                        box=np.asarray(event.prompt_box_xyxy, dtype=np.float32),
                    )
                    active.add(event.ann_obj_id)

                next_boundary = boundaries[boundary_index + 1]
                if not active or next_boundary <= boundary:
                    continue
                local_start = boundary - group.start_frame
                local_end = next_boundary - 1 - group.start_frame
                max_offset = local_end - local_start
                with torch.inference_mode(), _autocast_context(compute_device):
                    iterator = predictor.propagate_in_video(
                        state,
                        start_frame_idx=local_start,
                        max_frame_num_to_track=max_offset,
                        reverse=False,
                    )
                    for local_idx, obj_ids, mask_logits in iterator:
                        global_idx = group.start_frame + local_idx
                        next_frame_to_write = _write_until(
                            source_reader, writer, next_frame_to_write, global_idx
                        )
                        frame = source_reader.read(global_idx)
                        masks = (mask_logits > 0).detach().cpu().numpy()
                        rendered, per_object = _overlay_masks_bgr(
                            frame,
                            obj_ids,
                            masks,
                            event_by_obj_id,
                            alpha=mask_alpha,
                        )
                        # Display the original YOLO prompt box on its prompt frame.
                        for event in starts.get(global_idx, []):
                            color = COLORS_RGB[(event.event_id - 1) % len(COLORS_RGB)][::-1]
                            x1, y1, x2, y2 = map(round, event.prompt_box_xyxy)
                            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
                        writer.write(rendered)
                        next_frame_to_write = global_idx + 1
                        if per_object:
                            mask_records.append(
                                {"frame_idx": global_idx, "objects": per_object}
                            )

            # The last boundary is not propagated, but its removals are still part
            # of the requested object lifecycle and release per-object state cleanly.
            final_boundary = boundaries[-1]
            for event in removals.get(final_boundary, []):
                if event.ann_obj_id in active:
                    predictor.remove_object(
                        state, event.ann_obj_id, strict=True, need_output=False
                    )
                    active.remove(event.ann_obj_id)

            del state
            if compute_device.type == "cuda":
                torch.cuda.empty_cache()
            if not keep_extracted_frames:
                shutil.rmtree(group_dir, ignore_errors=True)

        next_frame_to_write = _write_until(
            source_reader, writer, next_frame_to_write, video_info.frame_count
        )
    finally:
        source_reader.close()
        writer.release()
        if not keep_extracted_frames:
            shutil.rmtree(frame_cache_root, ignore_errors=True)

    metadata = {
        "source_video": str(video_path),
        "output_video": str(output_video_path),
        "event_count": len(events),
        "groups": [
            {
                "start_frame": group.start_frame,
                "end_frame": group.end_frame,
                "event_ids": [event.event_id for event in group.events],
            }
            for group in groups
        ],
        "mask_frames": mask_records,
        "audio_preserved": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    mask_metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def read_frame_rgb(video_path: Path | str, frame_idx: int) -> np.ndarray:
    """Convenience helper for notebook visualization."""

    with VideoFrameReader(video_path) as reader:
        return cv2.cvtColor(reader.read(frame_idx), cv2.COLOR_BGR2RGB)


def event_rows(events: Sequence[DefectEvent]) -> list[dict[str, Any]]:
    """Compact event table suitable for pandas/display in a notebook."""

    return [
        {
            "event_id": item.event_id,
            "ann_obj_id": item.ann_obj_id,
            "class": item.class_name,
            "start_frame": item.start_frame,
            "start_time_seconds": (
                round(item.start_time_seconds, 3)
                if item.start_time_seconds is not None
                else None
            ),
            "start_timecode": item.start_timecode,
            "confirmation_frame": item.confirmation_frame,
            "end_frame": item.end_frame,
            "disappear_frame": item.disappear_frame,
            "disappear_time_seconds": (
                round(item.disappear_time_seconds, 3)
                if item.disappear_time_seconds is not None
                else None
            ),
            "disappear_timecode": item.disappear_timecode,
            "duration_frames": item.end_frame - item.start_frame + 1,
            "detections": len(item.detections),
            "first_conf": round(item.first_confidence, 4),
            "max_conf": round(item.max_confidence, 4),
            "prompt_box": [round(value, 1) for value in item.prompt_box_xyxy],
        }
        for item in events
    ]
