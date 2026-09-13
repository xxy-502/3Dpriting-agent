"""YOLO26m-OBB + SAM2.1 Hiera B+ image inference pipeline.

Pipeline:
1. Run YOLO OBB inference.
2. Convert every OBB to an expanded, image-clipped horizontal box and sort by
   YOLO confidence.
3. Use every detected box as a SAM2 box prompt by default. An optional
   ``max_masks`` limit can be used when explicitly requested.

The models can remain resident through ``YoloSamInferencer`` or the cached
``infer_single_image`` API. Both a single image and a directory are supported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
LANGGRAPH_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = LANGGRAPH_ROOT.parent

# Allow this script to use the SAM2 source tree without relying on the current
# working directory. This is also compatible with an editable SAM2 install.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep Ultralytics settings local to this project. This avoids permission
# problems with the per-user settings directory on managed Windows machines.
_ultralytics_config_root = SCRIPT_DIR / ".ultralytics"
(_ultralytics_config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_config_root))

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from langGraph.model_paths import (  # noqa: E402
    SAM_MODEL_DIR,
    YOLO_MODEL_DIR,
    resolve_sam_checkpoint,
    resolve_yolo_checkpoint,
)


DEFAULT_PICTURE_PATH = PROJECT_ROOT / "vos_dataset" / "433" / "box" / "images"
DEFAULT_OUTPUT_PATH = LANGGRAPH_ROOT / "runs" / "image_detection"
DEFAULT_YOLO_CHECKPOINT = YOLO_MODEL_DIR
DEFAULT_SAM_CHECKPOINT = SAM_MODEL_DIR
DEFAULT_SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_b+"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# RGB colors. The same detection gets the same color in step 2 and step 3.
COLORS = (
    (255, 82, 82),
    (66, 165, 245),
    (102, 187, 106),
    (255, 167, 38),
    (171, 71, 188),
    (38, 198, 218),
    (255, 238, 88),
    (236, 64, 122),
)


@dataclass
class Detection:
    """One YOLO OBB detection and its processed horizontal prompt box."""

    rank: int
    class_id: int
    class_name: str
    confidence: float
    obb_points: list[list[float]]
    box_xyxy: list[float]
    sam_score: float | None = None
    mask_area_pixels: int | None = None
    mask_area_ratio: float | None = None


@dataclass
class ImageInferenceOutput:
    """In-memory result for one image.

    The two NumPy images are intentionally kept outside the JSON metadata. Use
    :meth:`YoloSamInferencer.infer_and_save` for a serializable result.
    """

    image_path: Path
    image_width: int
    image_height: int
    box_expand: float
    max_masks: int | None
    detections: list[Detection]
    segmented_count: int
    step2_image: np.ndarray
    result_image: np.ndarray

    def to_metadata(self) -> dict[str, Any]:
        return {
            "success": True,
            "image_path": str(self.image_path),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "defect_exists": bool(self.detections),
            "defect_count": len(self.detections),
            "segmented_count": self.segmented_count,
            "box_expand": self.box_expand,
            "max_masks": self.max_masks,
            "detections": [asdict(detection) for detection in self.detections],
        }


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def collect_images(path: Path, recursive: bool = False) -> list[Path]:
    """Return one image or the supported images immediately under a directory."""

    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    candidates: Iterable[Path] = path.rglob("*") if recursive else path.iterdir()
    images = sorted(
        (item for item in candidates if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: str(item).lower(),
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {path}")
    return images


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image as contiguous uint8 RGB while honoring EXIF orientation."""

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        # ``np.asarray(PIL_image)`` can be read-only. SAM2 eventually wraps the
        # array with ``torch.from_numpy``, so make an explicitly writable copy.
        return np.asarray(image, dtype=np.uint8).copy()


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def process_yolo_obbs(
    result: Any,
    image_width: int,
    image_height: int,
    box_expand: float,
) -> list[Detection]:
    """Convert OBB vertices to expanded/clipped XYXY boxes and sort by confidence."""

    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return []

    points = obb.xyxyxyxy.detach().cpu().numpy().astype(np.float32)
    confidences = obb.conf.detach().cpu().numpy().astype(np.float32)
    class_ids = obb.cls.detach().cpu().numpy().astype(np.int64)
    names = getattr(result, "names", {})

    unsorted: list[Detection] = []
    max_x = float(max(0, image_width - 1))
    max_y = float(max(0, image_height - 1))
    for polygon, confidence, class_id in zip(points, confidences, class_ids):
        x1 = float(np.clip(polygon[:, 0].min() - box_expand, 0.0, max_x))
        y1 = float(np.clip(polygon[:, 1].min() - box_expand, 0.0, max_y))
        x2 = float(np.clip(polygon[:, 0].max() + box_expand, 0.0, max_x))
        y2 = float(np.clip(polygon[:, 1].max() + box_expand, 0.0, max_y))
        unsorted.append(
            Detection(
                rank=0,
                class_id=int(class_id),
                class_name=class_name(names, int(class_id)),
                confidence=float(confidence),
                obb_points=polygon.astype(float).tolist(),
                box_xyxy=[x1, y1, x2, y2],
            )
        )

    detections = sorted(unsorted, key=lambda detection: detection.confidence, reverse=True)
    for rank, detection in enumerate(detections, start=1):
        detection.rank = rank
    return detections


def draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a readable OpenCV label on an RGB image."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(image.shape[:2]) / 1200.0)
    thickness = max(1, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(np.clip(anchor[0], 0, max(0, image.shape[1] - text_width - 5)))
    y = int(np.clip(anchor[1], text_height + baseline + 5, max(text_height + baseline + 5, image.shape[0] - 1)))
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 5),
        (x + text_width + 5, y + 2),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 2, y - baseline - 1),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_processed_boxes(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Visualize every confidence-sorted, expanded horizontal box (step 2)."""

    canvas = image.copy()
    line_width = max(2, round(min(canvas.shape[:2]) / 400))
    for detection in detections:
        color = COLORS[(detection.rank - 1) % len(COLORS)]
        x1, y1, x2, y2 = (round(value) for value in detection.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        label = f"#{detection.rank} {detection.class_name} {detection.confidence:.3f}"
        draw_label(canvas, label, (x1, y1), color)
    if not detections:
        draw_label(canvas, "No YOLO detections", (10, 30), (90, 90, 90))
    return canvas


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha blend one boolean mask onto an RGB image."""

    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}")
    output = image.copy()
    binary_mask = mask.astype(bool)
    if np.any(binary_mask):
        color_array = np.asarray(color, dtype=np.float32)
        output[binary_mask] = np.clip(
            output[binary_mask].astype(np.float32) * (1.0 - alpha) + color_array * alpha,
            0,
            255,
        ).astype(np.uint8)
    return output


def draw_sam_results(
    image: np.ndarray,
    selected: Sequence[Detection],
    masks: Sequence[np.ndarray],
) -> np.ndarray:
    """Visualize selected SAM masks together with YOLO class/confidence (step 3)."""

    canvas = image.copy()
    for detection, mask in zip(selected, masks):
        color = COLORS[(detection.rank - 1) % len(COLORS)]
        canvas = overlay_mask(canvas, mask, color)

    line_width = max(2, round(min(canvas.shape[:2]) / 400))
    for detection in selected:
        color = COLORS[(detection.rank - 1) % len(COLORS)]
        x1, y1, x2, y2 = (round(value) for value in detection.box_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
        label = f"{detection.class_name} {detection.confidence:.3f}"
        draw_label(canvas, label, (x1, y1), color)
    if not selected:
        draw_label(canvas, "No masks generated", (10, 30), (90, 90, 90))
    return canvas


class YoloSamInferencer:
    """Resident inference service for the combined YOLO-OBB and SAM2 models.

    Construct this class once and reuse it for every request. The internal lock
    protects the stateful ``SAM2ImagePredictor`` when an application receives
    concurrent requests.
    """

    def __init__(
        self,
        yolo_checkpoint: Path | str = DEFAULT_YOLO_CHECKPOINT,
        sam_checkpoint: Path | str = DEFAULT_SAM_CHECKPOINT,
        sam_config: str = DEFAULT_SAM_CONFIG,
        device: str = "auto",
        confidence: float = 0.25,
        iou: float = 0.7,
        image_size: int | None = None,
    ) -> None:
        self.yolo_checkpoint = resolve_yolo_checkpoint(yolo_checkpoint)
        self.sam_checkpoint = resolve_sam_checkpoint(sam_checkpoint)
        self.sam_config = sam_config
        self.device = resolve_device(device)
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self._inference_lock = threading.RLock()

        print(f"Loading YOLO OBB model: {self.yolo_checkpoint}")
        self.yolo_model = YOLO(str(self.yolo_checkpoint), task="obb")
        if self.yolo_model.task != "obb":
            raise ValueError(f"Expected an OBB checkpoint, got task={self.yolo_model.task!r}")

        print(f"Loading SAM2 model: {self.sam_checkpoint}")
        sam2_model = build_sam2(
            self.sam_config,
            str(self.sam_checkpoint),
            device=self.device,
        )
        self.sam_predictor = SAM2ImagePredictor(sam2_model)
        print(f"Models ready on device: {self.device}")

    def _autocast_context(self):
        if not self.device.startswith("cuda"):
            return nullcontext()
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    def infer_image(
        self,
        image_path: Path | str,
        box_expand: float = 10,
        max_masks: int | None = None,
    ) -> ImageInferenceOutput:
        """Run the full pipeline for one image without writing files.

        ``max_masks=None`` is the default and segments every YOLO detection.
        The detections are already sorted by YOLO confidence.
        """

        if box_expand < 0:
            raise ValueError("box_expand must be >= 0")
        if max_masks is not None and max_masks < 1:
            raise ValueError("max_masks must be >= 1 or None")

        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {image_path.suffix}")
        image_rgb = load_rgb_image(image_path)
        image_height, image_width = image_rgb.shape[:2]

        # SAM2ImagePredictor stores per-image state, so the complete detection
        # operation must be serialized when this instance is shared by a server.
        with self._inference_lock:
            yolo_kwargs: dict[str, Any] = {
                "source": np.ascontiguousarray(image_rgb[:, :, ::-1]),
                "conf": self.confidence,
                "iou": self.iou,
                "device": self.device,
                "verbose": False,
            }
            if self.image_size is not None:
                yolo_kwargs["imgsz"] = self.image_size
            yolo_result = self.yolo_model.predict(**yolo_kwargs)[0]
            detections = process_yolo_obbs(
                yolo_result,
                image_width=image_width,
                image_height=image_height,
                box_expand=box_expand,
            )
            step2_image = draw_processed_boxes(image_rgb, detections)

            selected = detections if max_masks is None else detections[:max_masks]
            best_masks: list[np.ndarray] = []
            if selected:
                with torch.inference_mode(), self._autocast_context():
                    self.sam_predictor.set_image(image_rgb)
                    for detection in selected:
                        masks, scores, _ = self.sam_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=np.asarray(detection.box_xyxy, dtype=np.float32),
                            multimask_output=True,
                        )
                        best_index = int(np.argmax(scores))
                        best_mask = np.asarray(masks[best_index], dtype=bool)
                        detection.sam_score = float(scores[best_index])
                        detection.mask_area_pixels = int(best_mask.sum())
                        detection.mask_area_ratio = float(
                            detection.mask_area_pixels / (image_width * image_height)
                        )
                        best_masks.append(best_mask)

            result_image = draw_sam_results(image_rgb, selected, best_masks)

        return ImageInferenceOutput(
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            box_expand=float(box_expand),
            max_masks=max_masks,
            detections=detections,
            segmented_count=len(best_masks),
            step2_image=step2_image,
            result_image=result_image,
        )

    def infer_and_save(
        self,
        image_path: Path | str,
        output_root: Path | str = DEFAULT_OUTPUT_PATH,
        box_expand: float = 10,
        max_masks: int | None = None,
    ) -> dict[str, Any]:
        """Infer one image, save it in a unique directory, and return JSON-safe metadata."""

        prediction = self.infer_image(
            image_path=image_path,
            box_expand=box_expand,
            max_masks=max_masks,
        )
        run_directory = create_unique_run_directory(
            output_root=output_root,
            prefix=prediction.image_path.stem,
        )
        step2_path = run_directory / "step2_boxes.png"
        result_path = run_directory / "result.png"
        metadata_path = run_directory / "result.json"
        save_rgb_image(step2_path, prediction.step2_image)
        save_rgb_image(result_path, prediction.result_image)

        record = prediction.to_metadata()
        record.update(
            {
                "run_directory": str(run_directory),
                "step2_output_path": str(step2_path),
                "result_image_path": str(result_path),
                "metadata_path": str(metadata_path),
            }
        )
        metadata_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record


def safe_run_prefix(value: str) -> str:
    """Create a portable directory-name fragment from an image stem."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned[:80] or "image"


def create_unique_run_directory(
    output_root: Path | str = DEFAULT_OUTPUT_PATH,
    prefix: str = "inference",
) -> Path:
    """Create an independent output directory without platform-specific syntax."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{timestamp}_{safe_run_prefix(prefix)}_{uuid.uuid4().hex[:8]}"
    run_directory = root / run_name
    run_directory.mkdir(parents=False, exist_ok=False)
    return run_directory


_INFERENCER_CACHE: dict[tuple[Any, ...], YoloSamInferencer] = {}
_INFERENCER_CACHE_LOCK = threading.Lock()


def get_inferencer(
    yolo_checkpoint: Path | str = DEFAULT_YOLO_CHECKPOINT,
    sam_checkpoint: Path | str = DEFAULT_SAM_CHECKPOINT,
    sam_config: str = DEFAULT_SAM_CONFIG,
    device: str = "auto",
    confidence: float = 0.25,
    iou: float = 0.7,
    image_size: int | None = 1024,
) -> YoloSamInferencer:
    """Return a process-wide cached YOLO/SAM service for the given configuration."""

    resolved_yolo = resolve_yolo_checkpoint(yolo_checkpoint)
    resolved_sam = resolve_sam_checkpoint(sam_checkpoint)
    key = (
        str(resolved_yolo),
        str(resolved_sam),
        sam_config,
        resolve_device(device),
        float(confidence),
        float(iou),
        image_size,
    )
    with _INFERENCER_CACHE_LOCK:
        inferencer = _INFERENCER_CACHE.get(key)
        if inferencer is None:
            inferencer = YoloSamInferencer(
                yolo_checkpoint=resolved_yolo,
                sam_checkpoint=resolved_sam,
                sam_config=sam_config,
                device=device,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
            )
            _INFERENCER_CACHE[key] = inferencer
        return inferencer


def infer_single_image(
    image_path: Path | str,
    output_root: Path | str = DEFAULT_OUTPUT_PATH,
    box_expand: float = 10,
    max_masks: int | None = None,
    inferencer: YoloSamInferencer | None = None,
    **inferencer_kwargs: Any,
) -> dict[str, Any]:
    """Public structured single-image API used by the defect-detection tool.

    If no service instance is supplied, a process-wide cached instance is used,
    so repeated calls do not reload YOLO or SAM.
    """

    resident_inferencer = inferencer or get_inferencer(**inferencer_kwargs)
    return resident_inferencer.infer_and_save(
        image_path=image_path,
        output_root=output_root,
        box_expand=box_expand,
        max_masks=max_masks,
    )


def run_pipeline(
    picture_path: Path | str = DEFAULT_PICTURE_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    yolo_checkpoint: Path | str = DEFAULT_YOLO_CHECKPOINT,
    sam_checkpoint: Path | str = DEFAULT_SAM_CHECKPOINT,
    sam_config: str = DEFAULT_SAM_CONFIG,
    box_expand: float = 10,
    max_masks: int | None = None,
    device: str = "auto",
    confidence: float = 0.25,
    iou: float = 0.7,
    image_size: int | None = None,
    recursive: bool = False,
) -> list[dict[str, Any]]:
    """Batch-compatible CLI API; one unique batch directory is created per call."""

    input_path = Path(picture_path).expanduser().resolve()
    output_root = Path(output_path).expanduser().resolve()
    batch_directory = create_unique_run_directory(output_root, "batch")
    images = collect_images(input_path, recursive=recursive)
    inferencer = get_inferencer(
        yolo_checkpoint=yolo_checkpoint,
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        device=device,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
    )

    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path}")
        try:
            record = inferencer.infer_and_save(
                image_path=image_path,
                output_root=batch_directory,
                box_expand=box_expand,
                max_masks=max_masks,
            )
            print(
                f"  detections={record['defect_count']}, masks={record['segmented_count']}"
            )
        except Exception as exc:
            record = {
                "success": False,
                "image_path": str(image_path),
                "detections": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {record['error']}")
        records.append(record)

    metadata_path = batch_directory / "batch_results.json"
    metadata_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    succeeded = sum(record.get("success", False) for record in records)
    print(f"Finished: {succeeded}/{len(records)} images succeeded")
    print(f"Batch output: {batch_directory}")
    print(f"Batch metadata: {metadata_path}")
    return records


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO26m-OBB coarse localization followed by SAM2.1 mask inference."
    )
    parser.add_argument("--picture-path", type=Path, default=DEFAULT_PICTURE_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--yolo-checkpoint", type=Path, default=DEFAULT_YOLO_CHECKPOINT)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-config", default=DEFAULT_SAM_CONFIG)
    parser.add_argument("--box-expand", type=float, default=10)
    parser.add_argument(
        "--max-masks",
        "--output-mask",
        dest="max_masks",
        type=int,
        default=None,
        help="Maximum number of detections to segment; default segments all detections.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--confidence", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument(
        "--image-size",
        type=int,
        default=1024,
        help="Optional YOLO input size; omitted means use the checkpoint/model default.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search image directories recursively")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    records = run_pipeline(
        picture_path=args.picture_path,
        output_path=args.output_path,
        yolo_checkpoint=args.yolo_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam_config=args.sam_config,
        box_expand=args.box_expand,
        max_masks=args.max_masks,
        device=args.device,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.image_size,
        recursive=args.recursive,
    )
    if any(not record.get("success", False) for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
