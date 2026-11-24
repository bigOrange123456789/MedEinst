"""Intuitive pathway for candidate differential diagnoses."""
from src.utils.console import tr
import json
import math
import re
from typing import Dict, Any, Optional, List
from src.utils.api_client import LLMClient
from src.utils.json_utils import parse_json_from_text, extract_json_field
from config.prompt_phase1_trackA_templates import PHASE1_TRACKA_SYSTEM_PROMPT, PHASE1_TRACKA_FEW_SHOT_MESSAGES, build_phase1_tracka_user_prompt
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP, normalize_diagnosis_id, normalize_diagnosis_id_list


def normalize_diagnosis_id(raw_id: Any) -> Optional[str]:

    if raw_id is None:
        return None

    id_str = str(raw_id).strip().lstrip('0') or '0'

    if id_str.isdigit():
        num = int(id_str)
        if 1 <= num <= 49:
            return f"{num:02d}"

    raw_str = str(raw_id).strip()
    if len(raw_str) == 3 and raw_str.isdigit():

        option1 = raw_str[:2]  # "04" + "5"
        option2 = raw_str[1:]  # "0" + "45"

        valid_options = []
        for opt in [option1, option2]:
            normalized = opt.lstrip('0') or '0'
            if normalized.isdigit():
                num = int(normalized)
                if 1 <= num <= 49:
                    valid_options.append(f"{num:02d}")

        if len(valid_options) == 1:
            print(tr(
                f"[ID Normalize] Corrected '{raw_str}' -> '{valid_options[0]}'",
                f"[ID 规范化] 已修正 '{raw_str}' -> '{valid_options[0]}'"
            ))
            return valid_options[0]
        elif len(valid_options) > 1:

            print(tr(
                f"[ID Normalize] Ambiguous '{raw_str}', choosing '{valid_options[1]}' over '{valid_options[0]}'",
                f"[ID 规范化] '{raw_str}' 有歧义，选择 '{valid_options[1]}'，舍弃 '{valid_options[0]}'"
            ))
            return valid_options[1]

    print(tr(
        f"[ID Normalize] Invalid ID '{raw_id}', returning None",
        f"[ID 规范化] ID '{raw_id}' 无效，返回 None"
    ))
    return None

def normalize_candidate_list(candidates: List[Any]) -> List[str]:

    normalized = []
    seen = set()

    for raw_id in candidates:
        norm_id = normalize_diagnosis_id(raw_id)
        if norm_id and norm_id not in seen:
            normalized.append(norm_id)
            seen.add(norm_id)

    return normalized

class Phase1TrackAAgent: # 直觉诊断
    """Generate initial differential diagnosis candidates."""

    def __init__(self, llm_client: LLMClient, model_name: str = "gpt-4o"):
        self.llm_client = llm_client # 基模型
        self.model_name = model_name # 基模型名称

    def extract_json_from_response(self, response_text: str) -> Optional[Dict[str, Any]]:

        result = parse_json_from_text(response_text, verbose=True)

        if isinstance(result, dict):
            return result

        if isinstance(result, list):

            valid_ids = []
            for item in result:
                norm_id = normalize_diagnosis_id(item)
                if norm_id:
                    valid_ids.append(norm_id)

            if valid_ids:
                print(tr(
                    f"[Phase1 Track A] Parsed array as top_candidates: {valid_ids}",
                    f'[阶段1 轨道A] 已将数组解析为候选诊断列表：{valid_ids}'
                ))

                final_diagnosis_id = extract_json_field(response_text, 'final_diagnosis_id', verbose=False)
                if final_diagnosis_id:
                    final_id = normalize_diagnosis_id(final_diagnosis_id)
                else:
                    final_id = valid_ids[0]

                differential_reasoning = extract_json_field(response_text, 'differential_reasoning', verbose=False)

                return {
                    "top_candidates": valid_ids,
                    "final_diagnosis_id": final_id,
                    "differential_reasoning": differential_reasoning or ""
                }

        if result is None:

            top_candidates = extract_json_field(response_text, 'top_candidates', verbose=False)
            final_diagnosis_id = extract_json_field(response_text, 'final_diagnosis_id', verbose=False)
            differential_reasoning = extract_json_field(response_text, 'differential_reasoning', verbose=False)

            if top_candidates or final_diagnosis_id:
                print(tr(
                    "[Phase1 Track A] Using field extraction fallback",
                    '[阶段1 轨道A] 使用字段提取作为后备方案'
                ))
                return {
                    "top_candidates": top_candidates if isinstance(top_candidates, list) else [],
                    "final_diagnosis_id": str(final_diagnosis_id) if final_diagnosis_id else None,
                    "differential_reasoning": differential_reasoning or ""
                }

        return None

    def calculate_confidence_from_logprobs(
        self,
        response_obj: Any,
        final_diagnosis_id: str,
        response_text: str
    ) -> float:

        if response_obj is None:
            return 0.0

        try:
            logprobs_content = response_obj.choices[0].logprobs.content

            if not logprobs_content:
                return 0.0

            diagnosis_id_str = str(final_diagnosis_id)

            reconstructed_text = ""
            token_boundaries = []
            for token_info in logprobs_content:
                start_pos = len(reconstructed_text)
                reconstructed_text += token_info.token
                end_pos = len(reconstructed_text)
                token_boundaries.append((start_pos, end_pos, token_info))

            field_marker = '"final_diagnosis_id"'
            field_start_idx = reconstructed_text.rfind(field_marker)

            if field_start_idx == -1:
                return 0.0

            suffix_text = reconstructed_text[field_start_idx + len(field_marker):]

            pattern_quoted = rf'\s*:\s*"({re.escape(diagnosis_id_str)})"'
            match_quoted = re.search(pattern_quoted, suffix_text)

            if match_quoted:

                val_start_abs = field_start_idx + len(field_marker) + match_quoted.start(1)
                val_end_abs = field_start_idx + len(field_marker) + match_quoted.end(1)
            else:

                pattern_unquoted = rf'\s*:\s*({re.escape(diagnosis_id_str)})'
                match_unquoted = re.search(pattern_unquoted, suffix_text)

                if match_unquoted:
                    val_start_abs = field_start_idx + len(field_marker) + match_unquoted.start(1)
                    val_end_abs = field_start_idx + len(field_marker) + match_unquoted.end(1)
                else:

                    val_start_relative = suffix_text.find(diagnosis_id_str)
                    if val_start_relative == -1:
                        return 0.0
                    val_start_abs = field_start_idx + len(field_marker) + val_start_relative
                    val_end_abs = val_start_abs + len(diagnosis_id_str)

            target_tokens = []
            token_positions = []
            for start, end, token_info in token_boundaries:

                if end > val_start_abs and start < val_end_abs:
                    target_tokens.append(token_info)
                    token_positions.append((start, end, token_info.token))

            if not target_tokens:
                print(tr(
                    f"[DEBUG] No tokens found for diagnosis_id '{diagnosis_id_str}'",
                    f"[调试] 未找到诊断 ID '{diagnosis_id_str}' 对应的词元"
                ))
                return 0.0

            print(tr(
                f"[DEBUG] Value position: {val_start_abs} to {val_end_abs}",
                f'[调试] 值的位置：{val_start_abs} 至 {val_end_abs}'
            ))
            print(tr(
                f"[DEBUG] All candidate tokens (overlapping with value position):",
                '[调试] 所有候选词元（与值的位置重叠）：'
            ))
            for idx, (start, end, token) in enumerate(token_positions):
                token_info = target_tokens[idx]
                print(tr(
                    f"  Token: {repr(token)}, Logprob: {token_info.logprob}, Position: {start}-{end}",
                    f'  词元：{repr(token)}，对数概率：{token_info.logprob}，位置：{start}-{end}'
                ))

            sorted_indices = sorted(range(len(target_tokens)),
                                  key=lambda i: token_positions[i][0])

            matched_tokens = None
            for i in range(len(sorted_indices)):
                for j in range(i + 1, len(sorted_indices) + 1):

                    candidate_tokens = [target_tokens[sorted_indices[k]] for k in range(i, j)]
                    reconstructed_value = "".join(t.token for t in candidate_tokens)
                    cleaned_value = reconstructed_value.strip('"').strip("'").strip()

                    if cleaned_value == diagnosis_id_str:
                        matched_tokens = candidate_tokens
                        break
                if matched_tokens:
                    break

            if not matched_tokens:
                matched_tokens = target_tokens

            tokens_repr = [repr(t.token) for t in matched_tokens]
            logprobs_list = [t.logprob for t in matched_tokens]
            reconstructed_value = "".join(t.token for t in matched_tokens)
            cleaned_value = reconstructed_value.strip('"').strip("'").strip()

            print(tr(
                f"[DEBUG] Final diagnosis tokens: {tokens_repr} with logprobs: {logprobs_list}",
                f'[调试] 最终诊断词元：{tokens_repr}，对应对数概率：{logprobs_list}'
            ))
            print(tr(
                f"[DEBUG] Reconstructed value: {repr(reconstructed_value)}, Cleaned: {repr(cleaned_value)}, Expected: {repr(diagnosis_id_str)}",
                f'[调试] 重建值：{repr(reconstructed_value)}，清理后：{repr(cleaned_value)}，预期值：{repr(diagnosis_id_str)}'
            ))

            if cleaned_value != diagnosis_id_str:

                print(tr(
                    f"[DEBUG] Validation failed: cleaned_value '{cleaned_value}' != diagnosis_id_str '{diagnosis_id_str}'",
                    f"[调试] 校验失败：清理值 '{cleaned_value}' != 诊断 ID '{diagnosis_id_str}'"
                ))
                return 0.0

            sum_logprob = sum(t.logprob for t in matched_tokens)
            print(tr(
                f"[DEBUG] Sum of logprobs: {sum_logprob}, Exp(sum_logprob): {math.exp(sum_logprob)}",
                f'[调试] 对数概率总和：{sum_logprob}，总和的指数值：{math.exp(sum_logprob)}'
            ))

            confidence = math.exp(sum_logprob) * 100
            print(tr(
                f"[DEBUG] Calculated confidence: {confidence}%",
                f'[调试] 计算得到的置信度：{confidence}%'
            ))
            return max(0.0, min(100.0, confidence))

        except Exception as e:
            print(tr(
                f"Error calculating confidence from logprobs: {e}",
                f'根据对数概率计算置信度时出错：{e}'
            ))
            import traceback
            traceback.print_exc()
            return 0.0

    def process(self, patient_narrative: str, global_hint: Optional[str] = None) -> Dict[str, Any]: # 病例描述, 教授点评引导

        system_prompt = PHASE1_TRACKA_SYSTEM_PROMPT
        if global_hint and global_hint.strip():
            system_prompt = system_prompt + f"\n\n[TEACHER FEEDBACK - IMPORTANT]:\n{global_hint}"

        messages = [
            {"role": "system", "content": system_prompt}
        ] # system_prompt 中的关键点：从49种疾病中找出最可能的5种疾病，
        '''
        输出格式的要求：
        {
            "structured_analysis": { #结构化分析
                "demographics": "...",
                "positive_findings": ["..."], # 阳性症状
                "negative_findings": ["..."], # 阴性症状
                "subjective_to_objective_map": { # 主观到客观的映射
                "patient_quote": "medical_term" # 患者引用
                }
            },
            "differential_reasoning": "...", # 
            "top_candidates": ["ID1", "ID2", "ID3", "ID4", "ID5"], # 最可能的疾病
            "final_diagnosis_id": "XX" # 最终诊断结果
        }
        '''

        messages.extend(PHASE1_TRACKA_FEW_SHOT_MESSAGES) # 往聊天消息列表里注入1对示例。extend把输入展开后依次接入。

        user_prompt = build_phase1_tracka_user_prompt(patient_narrative) 
        # 头部添加一行“ **Patient Profile:**”，尾部添加两行“**Instruction:** Analyze the case above following the JSON format.”
        messages.append({"role": "user", "content": user_prompt})

        result = self.llm_client.generate_json(
            messages=messages, # 上限文长度为4，四个的role分别为：system【提示】、user【问题示例】、assistant【回复示例】、user【问题】
            model=self.model_name, # DeepSeek-R1-Distill-Qwen-1.5B
            logprobs=False,

            temperature=0.0, # 降低随机
            max_tokens=4096  # token上限为4K
        )

        if result["error"]:
            return {
                "structured_analysis": None,
                "top_candidates": [],
                "final_diagnosis_id": None,
                "calculated_confidence": 0.0,
                "error": result["error"]
            }

        response_content = result["content"]
        parsed_json = self.extract_json_from_response(response_content)

        if parsed_json is None:
            return {
                "structured_analysis": None,
                "top_candidates": [],
                "final_diagnosis_id": None,
                "calculated_confidence": 0.0,
                "error": "Failed to parse JSON from response",
                "raw_response": response_content[:2000] if response_content else ""
            }

        raw_final_id = parsed_json.get("final_diagnosis_id")
        final_diagnosis_id = normalize_diagnosis_id(raw_final_id)

        raw_candidates = parsed_json.get("top_candidates", [])
        top_candidates = normalize_candidate_list(raw_candidates)

        if not final_diagnosis_id:

            if top_candidates:
                final_diagnosis_id = top_candidates[0]
                print(tr(
                    f"[Phase1 Track A] Using first candidate '{final_diagnosis_id}' as fallback for missing final_diagnosis_id",
                    f"[阶段1 轨道A] 最终诊断 ID 缺失，使用首个候选 '{final_diagnosis_id}' 作为后备诊断"
                ))
            else:
                return {
                    "structured_analysis": parsed_json.get("structured_analysis"),
                    "top_candidates": top_candidates,
                    "final_diagnosis_id": None,
                    "calculated_confidence": 0.0,
                    "error": "final_diagnosis_id not found in response",
                    "raw_response": response_content[:2000] if response_content else ""
                }

        # if final_diagnosis_id not in top_candidates and top_candidates:

        #     top_candidates.insert(0, final_diagnosis_id)
        #     print(f"[Phase1 Track A] Added final_diagnosis_id '{final_diagnosis_id}' to top_candidates")

        # confidence = self.calculate_confidence_from_logprobs(
        #     result["response"],
        #     final_diagnosis_id,
        #     response_content
        # )
        confidence = 0.0

        return {
            "structured_analysis": parsed_json.get("structured_analysis"),
            "differential_reasoning": parsed_json.get("differential_reasoning"),
            "top_candidates": top_candidates,
            "final_diagnosis_id": final_diagnosis_id,
            "calculated_confidence": confidence,
            "error": None,
            "raw_response": None
        }

