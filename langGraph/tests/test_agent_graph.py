from __future__ import annotations

import unittest
from typing import Any

from langGraph.agent_graph import build_defect_agent
from langGraph.agent_nodes import AgentServices, _numeric_guardrail_issues


class FakeImageTool:
    def __init__(self, success: bool = True) -> None:
        self.success = success
        self.calls = 0

    def detect_defects(self, image_path: str) -> dict[str, Any]:
        self.calls += 1
        if not self.success:
            return {"success": False, "error": "synthetic image failure"}
        return {
            "success": True,
            "image_path": image_path,
            "defect_exists": True,
            "defect_count": 2,
            "class_counts": {"stringing": 2},
            "detections": [],
            "result_image_path": "/tmp/result.png",
        }


class FakeVideoGraph:
    def __init__(self) -> None:
        self.calls = 0
        self.last_input: dict[str, Any] = {}

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.last_input = state
        detection = {
            "success": True,
            "defect_exists": True,
            "defect_count": 1,
            "class_counts": {"cracks": 1},
            "events": [],
        }
        return {
            "detection_result": detection,
            "video_result": {**detection, "result_video_path": "/tmp/result.mp4"},
        }


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        self.queries.append(query)
        return [
            {
                "id": str(len(self.queries)),
                "chunk_id": f"chunk-{len(self.queries)}",
                "chunk_type": "entity",
                "score": 1.0,
                "page_content": "Inspect retraction, temperature, and print speed.",
                "metadata": {"source_path": "knowledge/defect.md"},
            }
        ]


class MixedMaterialRetriever(FakeRetriever):
    def search(self, query: str, *, limit: int) -> list[dict[str, Any]]:
        self.queries.append(query)
        if query.startswith("Official 3D printer material process profile"):
            return [
                {
                    "id": "abs-profile",
                    "chunk_id": "abs-profile",
                    "chunk_type": "entity",
                    "score": 1.0,
                    "page_content": "ABS nozzle temperature 250 C.",
                    "metadata": {
                        "source_path": "knowledge/abs.md",
                        "material_family": "ABS",
                    },
                },
                {
                    "id": "pla-profile",
                    "chunk_id": "pla-profile",
                    "chunk_type": "entity",
                    "score": 0.8,
                    "page_content": "PLA nozzle temperature 210-230 C.",
                    "metadata": {
                        "source_path": "knowledge/pla.md",
                        "material_family": "PLA",
                    },
                },
            ]
        return [
            {
                "id": "academic",
                "chunk_id": "academic",
                "chunk_type": "narrative",
                "score": 1.0,
                "page_content": "An unrelated academic temperature passage.",
                "metadata": {
                    "source_path": "knowledge/academic.pdf",
                    "category": "06_academic_and_standards",
                },
            },
            {
                "id": "stringing",
                "chunk_id": "stringing",
                "chunk_type": "entity",
                "score": 0.9,
                "page_content": "Stringing can relate to retraction and heat.",
                "metadata": {
                    "source_path": "knowledge/stringing.md",
                    "category": "01_defect_ontology",
                    "defect_types": ["stringing"],
                },
            },
        ]


class FakeLlm:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[dict[str, str]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.messages = messages
        return "基于检测结果和知识库的测试回答 [S1]"


class CorrectingFakeLlm(FakeLlm):
    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.messages = messages
        if self.calls == 1:
            return "喷嘴温度215°C，略低于推荐的210-230°C。"
        return "喷嘴温度215°C位于210-230°C范围内 [S1]。"


class AgentGraphTests(unittest.TestCase):
    def build(self, *, image_success: bool = True):
        image = FakeImageTool(success=image_success)
        video = FakeVideoGraph()
        retriever = FakeRetriever()
        llm = FakeLlm()
        graph = build_defect_agent(
            services=AgentServices(
                image_tool=image,
                video_graph=video,
                retriever=retriever,
                llm=llm,
            )
        )
        return graph, image, video, retriever, llm

    def test_image_detection_then_rag_then_llm(self) -> None:
        graph, image, video, retriever, llm = self.build()
        result = graph.invoke(
            {
                "user_query": "检测图片并分析",
                "media_path": "/tmp/input.jpg",
                "print_context": {"material": "PLA"},
            }
        )
        self.assertEqual(image.calls, 1)
        self.assertEqual(video.calls, 0)
        self.assertGreaterEqual(len(retriever.queries), 1)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result["result_image_path"], "/tmp/result.png")
        self.assertIn("[S1]", result["final_answer"])

    def test_video_render_flag_reaches_video_subgraph(self) -> None:
        graph, image, video, retriever, llm = self.build()
        result = graph.invoke(
            {
                "user_query": "检测视频并分析",
                "media_path": "/tmp/input.mp4",
                "render_result_video": True,
            }
        )
        self.assertEqual(image.calls, 0)
        self.assertEqual(video.calls, 1)
        self.assertTrue(video.last_input["render_result_video"])
        self.assertEqual(result["result_video_path"], "/tmp/result.mp4")
        self.assertEqual(llm.calls, 1)

    def test_supplied_result_skips_vision_tools(self) -> None:
        graph, image, video, retriever, llm = self.build()
        result = graph.invoke(
            {"user_query": "检测结果发现2处拉丝，材料PLA，请分析"}
        )
        self.assertEqual(image.calls, 0)
        self.assertEqual(video.calls, 0)
        self.assertEqual(result["detection_result"]["defect_count"], 2)
        self.assertEqual(llm.calls, 1)

    def test_detection_error_stops_before_rag(self) -> None:
        graph, image, video, retriever, llm = self.build(image_success=False)
        result = graph.invoke(
            {"user_query": "检测图片", "media_path": "/tmp/input.jpg"}
        )
        self.assertEqual(image.calls, 1)
        self.assertEqual(retriever.queries, [])
        self.assertEqual(llm.calls, 0)
        self.assertIn("synthetic image failure", result["final_answer"])

    def test_parameter_retrieval_rejects_other_material_profiles(self) -> None:
        image = FakeImageTool()
        video = FakeVideoGraph()
        retriever = MixedMaterialRetriever()
        llm = FakeLlm()
        graph = build_defect_agent(
            services=AgentServices(
                image_tool=image,
                video_graph=video,
                retriever=retriever,
                llm=llm,
            )
        )
        result = graph.invoke(
            {"user_query": "检测结果发现2处拉丝，材料PLA，请分析"}
        )
        source_paths = {
            source["source_path"] for source in result["sources"]
        }
        self.assertIn("knowledge/pla.md", source_paths)
        self.assertNotIn("knowledge/abs.md", source_paths)
        self.assertNotIn("knowledge/academic.pdf", source_paths)

    def test_guardrail_finds_range_and_duration_contradictions(self) -> None:
        issues = _numeric_guardrail_issues(
            "喷嘴温度215°C，略低于推荐的210-230°C。事件持续1.267秒。",
            {"nozzle_temperature_c": 215.0},
            {"events": [{"event_id": 1, "duration_seconds": 0.667}]},
        )
        self.assertEqual(len(issues), 2)

        directional_issue = _numeric_guardrail_issues(
            "建议降低喷嘴温度至210-230°C范围内。",
            {"nozzle_temperature_c": 215.0},
            {},
        )
        self.assertEqual(len(directional_issue), 1)

    def test_generation_is_rewritten_when_guardrail_finds_error(self) -> None:
        llm = CorrectingFakeLlm()
        graph = build_defect_agent(
            services=AgentServices(
                image_tool=FakeImageTool(),
                video_graph=FakeVideoGraph(),
                retriever=FakeRetriever(),
                llm=llm,
            )
        )
        result = graph.invoke(
            {
                "user_query": "检测结果发现1处拉丝，材料PLA，喷嘴温度215C。"
            }
        )
        self.assertEqual(llm.calls, 2)
        self.assertTrue(result["guardrail_issues"])
        self.assertIn("位于210-230", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
