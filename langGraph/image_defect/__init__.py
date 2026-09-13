"""Local YOLO-OBB + SAM2 image defect detector."""

from .defect_tool import DefectDetectionTool, detect_defects

__all__ = ["DefectDetectionTool", "detect_defects"]
