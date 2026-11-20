"""Coordinate the intuitive and analytic perception pathways."""
from src.utils.console import tr
from typing import Dict, Any, Optional
from src.utils.api_client import LLMClient
from src.agents.phase1_perception import Phase1TrackAAgent
from src.agents.phase1_trackB import Phase1TrackBAgent


class Phase1Manager:
    """Combine intuitive candidates with analytic observations."""

    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):

        self.llm_client = llm_client
        self.model_name = model_name

        self.track_a = Phase1TrackAAgent(llm_client, model_name=model_name)
        self.track_b = Phase1TrackBAgent(llm_client, model_name=model_name)

    def process(
        self,
        raw_narrative: str,
        age: int = None,
        sex: str = None,
        global_hint: Optional[str] = None
    ) -> Dict[str, Any]:

        print("=" * 60)
        print(tr(
            "[Phase1 Manager] Starting Dual-Track Processing...",
            '[阶段1] 开始双轨处理...'
        ))
        if global_hint:
            print(tr(
                f"[Phase1 Manager] Teacher Hint injected (length: {len(global_hint)})",
                f'[阶段1] 已注入教师提示（长度：{len(global_hint)}）'
            ))
        print("=" * 60)

        # ==================== Track A: Intuition ====================
        print(tr(
            "\n[Phase1 Manager] Executing Track A (Intuition)...",
            '\n[阶段1] 正在执行轨道 A（直觉诊断）...'
        ))
        track_a_result = self.track_a.process(raw_narrative, global_hint=global_hint)

        if track_a_result.get("error"):
            print(tr(
                f"[Phase1 Manager] Track A failed: {track_a_result['error']}",
                f"[阶段1] 轨道 A 失败：{track_a_result['error']}"
            ))
            return {
                "top_candidates": [],
                "final_diagnosis_id": None,
                "final_diagnosis_name": None,
                "differential_reasoning": "",
                "calculated_confidence": 0.0,
                "track_b_output": None,
                "raw_narrative": raw_narrative,
                "error": f"Track A failed: {track_a_result['error']}",
                "raw_response": track_a_result.get("raw_response", "")
            }

        print(tr(
            f"[Phase1 Manager] Track A complete. Top candidates: {track_a_result.get('top_candidates', [])}",
            f"[阶段1] 轨道 A 已完成。候选诊断：{track_a_result.get('top_candidates', [])}"
        ))

        # ==================== Track B: Perception ====================
        print(tr(
            "\n[Phase1 Manager] Executing Track B (Perception)...",
            '\n[阶段1] 正在执行轨道 B（病情感知）...'
        ))
        track_b_result = self.track_b.process(raw_narrative, age=age, sex=sex, global_hint=global_hint)

        if track_b_result.get("error"):
            print(tr(
                f"[Phase1 Manager] Track B failed: {track_b_result['error']}",
                f"[阶段1] 轨道 B 失败：{track_b_result['error']}"
            ))

            print(tr(
                "[Phase1 Manager] WARNING: Proceeding without P-Nodes from Track B",
                '[阶段1] 警告：轨道 B 未提供患者节点（P），将继续处理'
            ))
            track_b_result = {
                "problem_representation_one_liner": "",
                "p_nodes": [],
                "error": track_b_result.get("error")
            }
        else:
            p_nodes_count = len(track_b_result.get("p_nodes", []))
            print(tr(
                f"[Phase1 Manager] Track B complete. Extracted {p_nodes_count} P-Nodes",
                f'[阶段1] 轨道 B 已完成。提取了 {p_nodes_count} 个患者节点（P）'
            ))

        # ==================== Merge Results ====================
        print(tr(
            "\n[Phase1 Manager] Merging Track A and Track B results...",
            '\n[阶段1] 正在合并轨道 A 和轨道 B 的结果...'
        ))

        merged_result = {
            # From Track A
            "top_candidates": track_a_result.get("top_candidates", []),
            "final_diagnosis_id": track_a_result.get("final_diagnosis_id"),
            "final_diagnosis_name": self._get_diagnosis_name(track_a_result.get("final_diagnosis_id")),
            "differential_reasoning": track_a_result.get("differential_reasoning", ""),
            "calculated_confidence": track_a_result.get("calculated_confidence", 0.0),

            "track_b_output": {
                "problem_representation_one_liner": track_b_result.get("problem_representation_one_liner", ""),
                "p_nodes": track_b_result.get("p_nodes", [])
            },

            "raw_narrative": raw_narrative,

            "error": None,
            "track_b_warning": track_b_result.get("error")
        }

        print(tr(
            f"[Phase1 Manager] Phase 1 complete.",
            '[阶段1] 第一阶段已完成。'
        ))
        print(tr(
            f"  - Top-1 Diagnosis: {merged_result['final_diagnosis_name']} (ID: {merged_result['final_diagnosis_id']})",
            f"  - 首选诊断：{merged_result['final_diagnosis_name']}（ID：{merged_result['final_diagnosis_id']}）"
        ))
        print(tr(
            f"  - P-Nodes extracted: {len(merged_result['track_b_output']['p_nodes'])}",
            f"  - 已提取患者节点（P）：{len(merged_result['track_b_output']['p_nodes'])}"
        ))
        print("=" * 60)

        return merged_result

    def _get_diagnosis_name(self, diagnosis_id: Optional[str]) -> Optional[str]:

        if not diagnosis_id:
            return None

        from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
        return DIAGNOSIS_ID_MAP.get(diagnosis_id)

