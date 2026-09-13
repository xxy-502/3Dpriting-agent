"""Business nodes for the deterministic multimodal LangGraph workflow."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .agent_state import DefectAgentState
from .request_parser import parse_user_request


DEFECT_QUERY_TERMS = {
    "blobs": "blobs zits excess material seam extrusion restart",
    "cracks": "cracks layer separation splitting interlayer bonding",
    "spaghetti": "spaghetti failure detached print extrusion in free space",
    "stringing": "stringing oozing travel strands retraction",
}

CHINESE_QUERY_TERMS = {
    "拉丝": "stringing retraction travel oozing",
    "裂纹": "cracks layer separation interlayer bonding",
    "开裂": "cracks layer separation interlayer bonding",
    "炒面": "spaghetti print failure detachment",
    "意大利面": "spaghetti print failure detachment",
    "凸点": "blobs zits seam extrusion",
    "温度": "temperature",
    "回抽": "retraction",
    "热床": "bed temperature",
    "速度": "print speed",
    "风扇": "fan cooling",
}

KNOWN_MATERIAL_FAMILIES = (
    "PLA",
    "PETG",
    "ABS",
    "ASA",
    "TPU",
    "PA",
    "NYLON",
    "PC",
    "PVA",
    "HIPS",
    "PEEK",
)


@dataclass
class AgentServices:
    image_tool: Any
    video_graph: Any
    retriever: Any
    llm: Any


def _failure(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _print_context_text(context: dict[str, Any]) -> str:
    if not context:
        return "printer, material, nozzle, and current settings are unspecified"
    return ", ".join(f"{key}={value}" for key, value in sorted(context.items()))


def _material_family(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).upper().replace("-", " ").replace("_", " ")
    for family in KNOWN_MATERIAL_FAMILIES:
        if family in normalized.split() or normalized.startswith(family):
            return family
    return None


def _document_matches_material(
    row: dict[str, Any], target_material: str, *, require_match: bool
) -> bool:
    """Reject parameter records that explicitly belong to another material.

    Defect ontology and academic records are material-neutral and remain usable for
    causal analysis.  A process-profile query is stricter and only accepts a
    document that explicitly identifies the requested material.
    """

    target_family = _material_family(target_material)
    if target_family is None:
        return not require_match

    metadata = row.get("metadata", {})
    explicit_family = _material_family(metadata.get("material_family"))
    if explicit_family is not None:
        return explicit_family == target_family

    identifying_text = " ".join(
        str(metadata.get(key, ""))
        for key in (
            "material_product",
            "entity_title",
            "document_title",
            "source_path",
        )
    )
    mentioned_families = {
        family
        for family in KNOWN_MATERIAL_FAMILIES
        if _material_family(identifying_text) == family
        or family in identifying_text.upper().replace("-", " ").split()
    }
    if mentioned_families:
        return target_family in mentioned_families
    return not require_match


def _document_matches_defect(row: dict[str, Any], target_defect: str) -> bool:
    metadata = row.get("metadata", {})
    category = metadata.get("category")
    if category and category != "01_defect_ontology":
        return False
    # Synthetic/custom retrievers may not expose the collection's category
    # metadata; retain those rows so the graph stays adapter-friendly.
    if not category:
        return True

    normalized_target = target_defect.lower().strip()
    defect_types = {
        str(value).lower().strip() for value in metadata.get("defect_types", [])
    }
    canonical = str(metadata.get("canonical_defect_id") or "").lower()
    if normalized_target in defect_types or canonical.endswith(
        "." + normalized_target
    ):
        return True

    identifiers = " ".join(
        str(metadata.get(key, ""))
        for key in ("entity_title", "document_title", "source_path")
    ).lower()
    return bool(re.search(rf"\b{re.escape(normalized_target)}\b", identifiers))


def _detection_fact_text(detection: dict[str, Any]) -> str:
    """Create a compact authoritative summary so the LLM need not do arithmetic."""

    count = int(detection.get("defect_count", 0) or 0)
    counts = detection.get("class_counts") or {}
    lines = [
        f"defect_exists={bool(detection.get('defect_exists', False))}",
        f"defect_count={count}",
        "class_counts=" + json.dumps(counts, ensure_ascii=False, sort_keys=True),
    ]
    for event in detection.get("events") or []:
        lines.append(
            "event "
            f"{event.get('event_id')}: class={event.get('class_name')}, "
            f"frames={event.get('start_frame')}..{event.get('end_frame')}, "
            f"start_seconds={event.get('start_time_seconds')}, "
            f"disappear_seconds={event.get('disappear_time_seconds')}, "
            f"duration_seconds={event.get('duration_seconds')}"
        )
    return "\n".join(lines)


def _numeric_guardrail_issues(
    answer: str,
    print_context: dict[str, Any],
    detection: dict[str, Any],
) -> list[str]:
    """Find explicit contradictions that a small local model can overlook."""

    issues: list[str] = []
    contradictory_words = (
        "低于",
        "高于",
        "超出",
        "不在",
        "略低",
        "略高",
        "降低",
        "提高",
        "调低",
        "调高",
    )
    sentences = [part.strip() for part in re.split(r"[。！？\n]+", answer) if part.strip()]
    numeric_fields = {
        "nozzle_temperature_c": "喷嘴温度",
        "bed_temperature_c": "热床温度",
        "print_speed_mm_s": "打印速度",
        "retraction_distance_mm": "回抽距离",
        "retraction_speed_mm_s": "回抽速度",
        "layer_height_mm": "层高",
        "nozzle_diameter_mm": "喷嘴直径",
    }
    range_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:[-–—~～至到])\s*(\d+(?:\.\d+)?)"
    )
    for field, label in numeric_fields.items():
        if field not in print_context:
            continue
        value = float(print_context[field])
        for sentence in sentences:
            if label not in sentence or not any(word in sentence for word in contradictory_words):
                continue
            for low_text, high_text in range_pattern.findall(sentence):
                low, high = sorted((float(low_text), float(high_text)))
                if low <= value <= high:
                    issues.append(
                        f"{label}={value:g} 位于句中范围 {low:g}–{high:g} 内，"
                        "不能描述为低于、高于或超出该范围，也不能把整个范围"
                        "直接写成降低/提高的目标；应先说明当前值在范围内。"
                    )
                    break

    for event in detection.get("events") or []:
        duration = event.get("duration_seconds")
        if duration is None:
            continue
        duration_text = f"{float(duration):g}"
        for sentence in sentences:
            if "持续" in sentence and re.search(r"\d+(?:\.\d+)?\s*秒", sentence):
                if duration_text not in sentence:
                    issues.append(
                        f"事件 {event.get('event_id')} 的持续时间应为 "
                        f"{duration_text} 秒；不得使用消失时间代替。"
                    )
                break

    missing_section = ""
    if "需要补充的信息" in answer:
        missing_section = answer.split("需要补充的信息", 1)[1]
        missing_section = missing_section.split("证据来源", 1)[0]
    known_labels = {
        "material": ("材料",),
        "printer_model": ("打印机型号", "机型"),
        "nozzle_diameter_mm": ("喷嘴直径", "喷嘴尺寸"),
        "nozzle_temperature_c": ("喷嘴温度",),
        "bed_temperature_c": ("热床温度", "床温"),
        "print_speed_mm_s": ("打印速度",),
        "retraction_distance_mm": ("回抽距离",),
        "retraction_speed_mm_s": ("回抽速度",),
        "layer_height_mm": ("层高",),
        "fan_speed_percent": ("风扇速度",),
    }
    for field, labels in known_labels.items():
        if field in print_context and any(label in missing_section for label in labels):
            issues.append(f"{field} 已由用户提供，不能列为需要补充的信息。")
    return list(dict.fromkeys(issues))


def build_rag_queries(state: DefectAgentState) -> list[str]:
    detection = state.get("detection_result") or {}
    class_counts = detection.get("class_counts") or {}
    print_context = _print_context_text(state.get("print_context", {}))

    queries: list[str] = []
    for class_name, count in class_counts.items():
        normalized = str(class_name).lower().strip()
        defect_terms = DEFECT_QUERY_TERMS.get(normalized, normalized)
        queries.append(
            "3D printing FFF FDM defect analysis. "
            f"Detected defect={defect_terms}, confirmed count={count}. "
            f"Printing context: {print_context}. "
            "Retrieve observable evidence, likely causes, distinguishing signs, "
            "parameters to inspect, and safe corrective actions."
        )

    if class_counts:
        queries.append(
            "Official 3D printer material process profile and parameter facts. "
            f"Printing context: {print_context}. "
            "Retrieve applicable nozzle temperature, bed temperature, print speed, "
            "cooling, retraction, layer height, and first-layer settings."
        )
    else:
        original = state["user_query"]
        expansions = " ".join(
            english for chinese, english in CHINESE_QUERY_TERMS.items() if chinese in original
        )
        queries.append(
            "3D printing FFF FDM knowledge and process parameter analysis. "
            f"Keywords: {expansions}. Printing context: {print_context}. "
            f"User question: {original}"
        )
    return queries


class DefectAgentNodes:
    def __init__(self, services: AgentServices, *, rag_limit: int = 6) -> None:
        if rag_limit < 1:
            raise ValueError("rag_limit must be >= 1")
        self.services = services
        self.rag_limit = rag_limit

    def prepare_request(self, state: DefectAgentState) -> dict[str, Any]:
        try:
            parsed = parse_user_request(
                state["user_query"],
                media_path=state.get("media_path"),
                print_context=state.get("print_context"),
                detection_result=state.get("detection_result"),
            )
            return {
                **parsed,
                "rag_documents": [],
                "sources": [],
                "video_result": {},
                "result_image_path": None,
                "result_video_path": None,
                "final_answer": "",
                "guardrail_issues": [],
                "error": "",
            }
        except Exception as exc:
            failure = _failure("prepare_request", exc)
            return {"error": failure["error"], "detection_result": failure}

    def route_request(
        self, state: DefectAgentState
    ) -> Literal["image", "video", "rag", "error"]:
        if state.get("error"):
            return "error"
        media_type = state.get("media_type", "none")
        if media_type == "image":
            return "image"
        if media_type == "video":
            return "video"
        return "rag"

    def image_detection(self, state: DefectAgentState) -> dict[str, Any]:
        try:
            result = self.services.image_tool.detect_defects(state["media_path"])
        except Exception as exc:
            result = _failure("image_detection", exc)
        return {
            "detection_result": result,
            "result_image_path": result.get("result_image_path"),
            "error": result.get("error", ""),
        }

    def video_detection(self, state: DefectAgentState) -> dict[str, Any]:
        try:
            video_state = self.services.video_graph.invoke(
                {
                    "video_path": state["media_path"],
                    "output_dir": state.get("output_dir"),
                    "render_result_video": bool(
                        state.get("render_result_video", False)
                    ),
                    "output_video_name": state.get(
                        "output_video_name", "defect_mask_tracking.mp4"
                    ),
                }
            )
            result = video_state["video_result"]
            detection = video_state["detection_result"]
        except Exception as exc:
            result = _failure("video_detection", exc)
            detection = result
        return {
            "video_result": result,
            "detection_result": detection,
            "result_video_path": result.get("result_video_path"),
            "error": result.get("error", ""),
        }

    def route_after_detection(
        self, state: DefectAgentState
    ) -> Literal["rag", "error"]:
        detection = state.get("detection_result") or {}
        return "rag" if detection.get("success") is True else "error"

    def retrieve_knowledge(self, state: DefectAgentState) -> dict[str, Any]:
        try:
            queries = build_rag_queries(state)
            # Retrieve a few records for each defect/parameter aspect, then
            # deduplicate before passing a bounded context to the 3B model.
            per_query_limit = min(3, self.rag_limit)
            target_material = str(state.get("print_context", {}).get("material", ""))
            defect_names = list(
                (state.get("detection_result") or {}).get("class_counts", {})
            )
            documents: list[dict[str, Any]] = []
            seen: set[str] = set()
            seen_source_documents: set[str] = set()
            for query_index, query in enumerate(queries):
                target_defect = (
                    str(defect_names[query_index]).lower()
                    if query_index < len(defect_names)
                    else None
                )
                is_profile_query = query.startswith(
                    "Official 3D printer material process profile"
                )
                # Fetch extra candidates because the highest scoring parameter
                # record may describe a different material family.
                search_limit = max(per_query_limit, 12 if target_material else 3)
                accepted_for_query = 0
                for row in self.services.retriever.search(
                    query, limit=search_limit
                ):
                    metadata = row.get("metadata", {})
                    if target_defect and not _document_matches_defect(
                        row, target_defect
                    ):
                        continue
                    if target_material and not _document_matches_material(
                        row, target_material, require_match=is_profile_query
                    ):
                        continue
                    identity = str(row.get("chunk_id") or row.get("id"))
                    if identity in seen:
                        continue
                    source_document_id = str(
                        metadata.get("source_document_id") or ""
                    )
                    if (
                        source_document_id
                        and source_document_id in seen_source_documents
                    ):
                        continue
                    seen.add(identity)
                    if source_document_id:
                        seen_source_documents.add(source_document_id)
                    documents.append(row)
                    accepted_for_query += 1
                    if len(documents) >= self.rag_limit:
                        break
                    if accepted_for_query >= per_query_limit:
                        break
                if len(documents) >= self.rag_limit:
                    break

            if not documents:
                raise RuntimeError("The local knowledge base returned no documents")

            sources = []
            for index, row in enumerate(documents, start=1):
                metadata = row.get("metadata", {})
                sources.append(
                    {
                        "source_id": f"S{index}",
                        "chunk_id": row.get("chunk_id"),
                        "score": row.get("score"),
                        "title": metadata.get("entity_title")
                        or metadata.get("document_title")
                        or metadata.get("source_path"),
                        "source_path": metadata.get("source_path"),
                        "section": metadata.get("section_title"),
                        "authority_rank": metadata.get("authority_rank"),
                        "material_family": metadata.get("material_family"),
                    }
                )
            return {
                "rag_queries": queries,
                "rag_documents": documents,
                "sources": sources,
                "error": "",
            }
        except Exception as exc:
            failure = _failure("rag_retrieval", exc)
            return {"error": failure["error"], "rag_documents": []}

    def route_after_rag(
        self, state: DefectAgentState
    ) -> Literal["generate", "error"]:
        return "error" if state.get("error") else "generate"

    def generate_answer(self, state: DefectAgentState) -> dict[str, Any]:
        evidence_blocks: list[str] = []
        target_material = str(state.get("print_context", {}).get("material", ""))
        target_family = _material_family(target_material)
        for index, document in enumerate(state.get("rag_documents", []), start=1):
            content = str(document.get("page_content", ""))[:3500]
            metadata = document.get("metadata", {})
            source_family = _material_family(metadata.get("material_family"))
            usage = (
                f"may_use_for_{target_family}_parameters"
                if target_family and source_family == target_family
                else "general_defect_mechanism_only_no_material_parameter_ranges"
            )
            evidence_blocks.append(
                f"[S{index}] source={metadata.get('source_path')}; "
                f"material_family={metadata.get('material_family') or 'unspecified'}; "
                f"usage={usage}\n"
                f"{content}"
            )
        evidence = "\n\n".join(evidence_blocks)
        print_context = state.get("print_context", {})
        known_context_fields = ", ".join(sorted(print_context)) or "none"
        detection_facts = _detection_fact_text(
            state.get("detection_result") or {}
        )

        system_prompt = """你是离线3D打印缺陷诊断助手。请严格遵守：
1. 检测结果是观测事实；知识库中的原因只是可能原因，不得把相关性写成已经证实的因果关系。
2. 只能依据给出的检测结果、打印上下文和知识库证据回答，不得编造检测、参数或来源。
3. 参数建议必须说明适用条件。缺少材料、打印机或喷嘴信息时，先指出信息缺口，不给出虚假的唯一精确值。
4. 区分“知识库基准参数”和“针对当前缺陷的调整方向”，不要混为一谈。
5. 材料参数只能引用 material_family 与用户材料一致的证据；material_family=unspecified 的资料只能用于通用缺陷机理，不能作为材料参数范围。
6. 做数值比较时必须先核对范围：当前值位于推荐范围内时，不得称为“过高”“过低”或“超出范围”，只能说明仍可在范围内逐步试调。
7. 打印上下文 JSON 中已经出现的字段都属于已知信息，禁止再次列入“需要补充的信息”。
8. 每个数值范围必须紧跟真正包含该数值的来源编号；不得把某来源没有记载的参数归因给它。
9. “检测事实摘要”已经完成事件计数和持续时间计算，必须原样采用，不得用消失时间替代持续时间，也不得重新计算。
10. 不要把数据表中的上限直接写成调整目标；当前参数已在资料范围内时，只能建议单变量小步试验并观察结果。
11. 引用知识库时使用 [S1]、[S2] 形式；结论使用中文。
12. 输出顺序：检测结论、可能原因、参数分析与建议、需要补充的信息、证据来源。"""

        user_prompt = (
            f"用户问题：{state['user_query']}\n\n"
            f"检测事实摘要（权威，不得重新计算）：\n{detection_facts}\n\n"
            "检测结果：\n"
            f"{json.dumps(state.get('detection_result'), ensure_ascii=False, indent=2)}\n\n"
            f"打印上下文（已知字段：{known_context_fields}）：\n"
            f"{json.dumps(print_context, ensure_ascii=False, indent=2)}\n\n"
            f"本地知识库证据：\n{evidence}"
        )
        try:
            answer = self.services.llm.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            if not answer:
                raise RuntimeError("The local Qwen model returned an empty answer")
            guardrail_issues = _numeric_guardrail_issues(
                answer,
                print_context,
                state.get("detection_result") or {},
            )
            if guardrail_issues:
                correction_prompt = (
                    "下面的回答存在可验证的一致性错误。请完整重写回答，修正列出的错误；"
                    "保留原有章节和有依据的来源引用，不增加新数值或新事实。\n\n"
                    "必须修正：\n- "
                    + "\n- ".join(guardrail_issues)
                    + f"\n\n待修正回答：\n{answer}"
                )
                answer = self.services.llm.generate(
                    [
                        {
                            "role": "system",
                            "content": (
                                "你是严格的3D打印技术回答校对器。"
                                "列出的算术事实优先级最高，必须逐项修正。"
                            ),
                        },
                        {"role": "user", "content": correction_prompt},
                    ]
                )
                if not answer:
                    raise RuntimeError("The local Qwen correction pass returned an empty answer")
            return {
                "final_answer": answer,
                "guardrail_issues": guardrail_issues,
                "error": "",
            }
        except Exception as exc:
            return {
                "error": str(exc),
                "final_answer": f"本地大模型生成失败：{exc}",
            }

    def error_response(self, state: DefectAgentState) -> dict[str, Any]:
        detection = state.get("detection_result") or {}
        detail = state.get("error") or detection.get("error") or "未知错误"
        return {"final_answer": f"任务未完成：{detail}"}
