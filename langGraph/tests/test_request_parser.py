from __future__ import annotations

import unittest

from langGraph.request_parser import parse_user_request


class RequestParserTests(unittest.TestCase):
    def test_windows_video_path_and_print_context(self) -> None:
        result = parse_user_request(
            r"请检测 D:\data\print video.mp4，打印机为 Bambu H2D，材料PLA，"
            r"喷嘴0.4mm，喷嘴温度210C，打印速度200mm/s"
        )
        self.assertEqual(result["media_path"], r"D:\data\print video.mp4")
        self.assertEqual(result["media_type"], "video")
        self.assertEqual(result["print_context"]["material"], "PLA")
        self.assertEqual(result["print_context"]["nozzle_diameter_mm"], 0.4)
        self.assertEqual(result["print_context"]["print_speed_mm_s"], 200.0)

    def test_linux_quoted_image_path(self) -> None:
        result = parse_user_request('检测 "/data/print samples/a_01.jpg" 是否开裂')
        self.assertEqual(result["media_path"], "/data/print samples/a_01.jpg")
        self.assertEqual(result["media_type"], "image")

    def test_user_supplied_defect_result(self) -> None:
        result = parse_user_request("检测结果发现2处拉丝和1处裂纹，材料PETG")
        detection = result["detection_result"]
        self.assertIsNotNone(detection)
        self.assertEqual(detection["defect_count"], 3)
        self.assertEqual(detection["class_counts"], {"cracks": 1, "stringing": 2})
        self.assertEqual(result["print_context"]["material"], "PETG")

    def test_general_question_does_not_fake_a_detection(self) -> None:
        result = parse_user_request("什么是3D打印拉丝？")
        self.assertEqual(result["media_type"], "none")
        self.assertIsNone(result["detection_result"])


if __name__ == "__main__":
    unittest.main()
