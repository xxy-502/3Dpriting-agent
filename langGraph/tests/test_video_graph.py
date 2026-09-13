from __future__ import annotations

import unittest
from typing import Any

from langGraph.video.video_graph import build_video_detection_graph
from langGraph.video.video_nodes import VideoDetectionNodes


class FakeYoloTool:
    def __init__(self, *, defect_exists: bool = True, success: bool = True) -> None:
        self.defect_exists = defect_exists
        self.success = success
        self.calls = 0

    def detect(self, video_path: str, *, output_dir: str | None = None) -> dict[str, Any]:
        self.calls += 1
        if not self.success:
            raise RuntimeError("synthetic YOLO failure")
        event_count = 2 if self.defect_exists else 0
        return {
            "success": True,
            "stage": "video_yolo_detection",
            "video_path": video_path,
            "defect_exists": self.defect_exists,
            "defect_count": event_count,
            "event_count": event_count,
            "class_counts": {"cracks": 2} if self.defect_exists else {},
            "events": [],
            "artifacts": {
                "yolo_events_json_path": "C:/synthetic/yolo_defect_events.json",
                "yolo_events_csv_path": "C:/synthetic/yolo_defect_events.csv",
            },
        }


class FakeSamTool:
    def __init__(self) -> None:
        self.calls = 0

    def render(
        self,
        video_path: str,
        yolo_events_json_path: str,
        *,
        output_dir: str | None = None,
        output_video_name: str | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "success": True,
            "stage": "video_sam_rendering",
            "video_path": video_path,
            "skipped": False,
            "defect_exists": True,
            "defect_count": 2,
            "event_count": 2,
            "class_counts": {"cracks": 2},
            "result_video_path": "C:/synthetic/defect_mask_tracking.mp4",
            "artifacts": {},
        }


class VideoGraphTests(unittest.TestCase):
    def build(self, *, defect_exists: bool = True, success: bool = True):
        yolo = FakeYoloTool(defect_exists=defect_exists, success=success)
        sam = FakeSamTool()
        nodes = VideoDetectionNodes(yolo_tool=yolo, sam_tool=sam)
        return build_video_detection_graph(nodes=nodes), yolo, sam

    def test_render_runs_when_requested_and_defect_exists(self) -> None:
        graph, yolo, sam = self.build()
        state = graph.invoke(
            {
                "video_path": "C:/synthetic/input.mp4",
                "render_result_video": True,
            }
        )
        self.assertEqual(yolo.calls, 1)
        self.assertEqual(sam.calls, 1)
        self.assertEqual(
            state["video_result"]["result_video_path"],
            "C:/synthetic/defect_mask_tracking.mp4",
        )

    def test_render_is_skipped_when_not_requested(self) -> None:
        graph, yolo, sam = self.build()
        state = graph.invoke(
            {
                "video_path": "C:/synthetic/input.mp4",
                "render_result_video": False,
            }
        )
        self.assertEqual(yolo.calls, 1)
        self.assertEqual(sam.calls, 0)
        self.assertNotIn("sam_rendering", state["video_result"])

    def test_render_is_skipped_without_defects(self) -> None:
        graph, yolo, sam = self.build(defect_exists=False)
        state = graph.invoke(
            {
                "video_path": "C:/synthetic/input.mp4",
                "render_result_video": True,
            }
        )
        self.assertEqual(yolo.calls, 1)
        self.assertEqual(sam.calls, 0)
        self.assertFalse(state["video_result"]["defect_exists"])

    def test_yolo_failure_is_structured_and_stops_rendering(self) -> None:
        graph, yolo, sam = self.build(success=False)
        state = graph.invoke(
            {
                "video_path": "C:/synthetic/input.mp4",
                "render_result_video": True,
            }
        )
        self.assertEqual(yolo.calls, 1)
        self.assertEqual(sam.calls, 0)
        self.assertFalse(state["video_result"]["success"])
        self.assertEqual(state["video_result"]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
