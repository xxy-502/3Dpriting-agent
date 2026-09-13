"""Deterministic extraction of media paths and common FDM printing context."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal


IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
MEDIA_SUFFIX_PATTERN = "|".join(
    re.escape(value.lstrip(".")) for value in sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES)
)

DEFECT_ALIASES = {
    "blobs": ("blobs", "blob", "zits", "凸点", "疙瘩", "料瘤"),
    "cracks": ("cracks", "crack", "layer separation", "裂纹", "开裂", "层分离"),
    "spaghetti": ("spaghetti", "spaghetti monster", "炒面", "意大利面"),
    "stringing": ("stringing", "strings", "拉丝", "牵丝"),
}


def _clean_path(value: str) -> str:
    text = value.strip().strip("`\"'").rstrip("，。；;、)）]】")
    return text.replace("\\_", "_")


def extract_media_path(text: str) -> str | None:
    """Extract a quoted, Windows, Linux, or explicit relative media path."""

    suffixes = MEDIA_SUFFIX_PATTERN
    patterns = (
        rf"[`\"']([^`\"'\r\n]+\.(?:{suffixes}))[`\"']",
        rf"([A-Za-z]:[\\/][^\r\n<>|?*\"]+?\.(?:{suffixes}))(?=$|[\s，。；;、)）\]】])",
        rf"(/[^\r\n`\"']+?\.(?:{suffixes}))(?=$|[\s，。；;、)）\]】])",
        rf"((?:\.{1,2}[\\/])[^\r\n`\"']+?\.(?:{suffixes}))(?=$|[\s，。；;、)）\]】])",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_path(match.group(1))
    return None


def media_type_from_path(path: str | Path | None) -> Literal["image", "video", "none", "unsupported"]:
    if path is None:
        return "none"
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "unsupported"


def _first_number(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def extract_print_context(text: str) -> dict[str, Any]:
    """Extract common settings without asking the LLM to control routing."""

    context: dict[str, Any] = {}
    material_match = re.search(
        r"(?<![A-Za-z0-9])(PLA(?:\+| Pro)?|PETG|ABS|ASA|TPU(?:\s*95A)?|PA(?:6|12)?|NYLON|PC)(?![A-Za-z0-9])",
        text,
        flags=re.IGNORECASE,
    )
    if material_match:
        context["material"] = material_match.group(1).upper()

    printer_patterns = (
        r"(?:打印机|printer(?:\s+model)?)\s*(?:为|是|[:：])\s*([^，。,;；\n]+)",
        r"(?:机型|型号)\s*(?:为|是|[:：])\s*([^，。,;；\n]+)",
    )
    for pattern in printer_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            context["printer_model"] = match.group(1).strip()
            break

    numeric_fields = {
        "nozzle_diameter_mm": (
            r"(?:喷嘴(?:直径|口径)?|nozzle(?:\s+diameter)?)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*mm",
        ),
        "nozzle_temperature_c": (
            r"(?:喷嘴温度|打印温度|nozzle\s+temperature)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*(?:°?c|摄氏度)?",
        ),
        "bed_temperature_c": (
            r"(?:热床温度|底板温度|bed\s+temperature)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*(?:°?c|摄氏度)?",
        ),
        "print_speed_mm_s": (
            r"(?:打印速度|print\s+speed)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*mm/s",
        ),
        "retraction_distance_mm": (
            r"(?:回抽距离|回退距离|retraction\s+distance)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*mm",
        ),
        "retraction_speed_mm_s": (
            r"(?:回抽速度|回退速度|retraction\s+speed)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*mm/s",
        ),
        "layer_height_mm": (
            r"(?:层高|layer\s+height)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*mm",
        ),
        "fan_speed_percent": (
            r"(?:风扇(?:速度)?|fan\s+speed)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*%",
        ),
    }
    for name, patterns in numeric_fields.items():
        value = _first_number(text, patterns)
        if value is not None:
            context[name] = value
    return context


def extract_supplied_detection(text: str) -> dict[str, Any] | None:
    """Recognize an explicitly supplied defect result when no media is present."""

    result_markers = (
        "检测结果",
        "检测到",
        "发现",
        "存在缺陷",
        "defect result",
        "detected",
        "found",
    )
    lowered = text.lower()
    if not any(marker in lowered for marker in result_markers):
        return None

    counts: dict[str, int] = {}
    for canonical, aliases in DEFECT_ALIASES.items():
        matched_alias = next(
            (alias for alias in aliases if alias.lower() in lowered), None
        )
        if matched_alias is None:
            continue
        escaped = re.escape(matched_alias)
        patterns = (
            rf"(\d+)\s*(?:处|个|次|events?)?\s*{escaped}",
            rf"{escaped}[^\d\r\n]{{0,12}}(\d+)\s*(?:处|个|次|events?)?",
        )
        count = next(
            (
                int(match.group(1))
                for pattern in patterns
                if (match := re.search(pattern, lowered, flags=re.IGNORECASE))
            ),
            1,
        )
        counts[canonical] = count

    if not counts:
        return None
    total = sum(counts.values())
    return {
        "success": True,
        "stage": "user_supplied_detection",
        "source": "user",
        "defect_exists": total > 0,
        "defect_count": total,
        "class_counts": counts,
        "detections": [],
        "summary": "用户在问题中直接提供了缺陷结果，未运行视觉检测工具。",
    }


def parse_user_request(
    user_query: str,
    *,
    media_path: str | Path | None = None,
    print_context: dict[str, Any] | None = None,
    detection_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")

    resolved_media = str(media_path) if media_path is not None else extract_media_path(user_query)
    media_type = media_type_from_path(resolved_media)
    if media_type == "unsupported":
        raise ValueError(f"Unsupported media path: {resolved_media}")

    extracted_context = extract_print_context(user_query)
    extracted_context.update(print_context or {})
    supplied = detection_result
    if supplied is None and media_type == "none":
        supplied = extract_supplied_detection(user_query)

    return {
        "user_query": user_query.strip(),
        "media_path": resolved_media,
        "media_type": media_type,
        "print_context": extracted_context,
        "detection_result": supplied,
    }
