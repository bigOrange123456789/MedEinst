#!/usr/bin/env python3
"""Run full ECR-Agent inference on MedEinst case pairs."""

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.console import (
    ConsoleArgumentParser, configure_console_encoding, set_console_language, tr,
)
from src.utils.prompt_utils import DIAGNOSIS_ID_MAP
from src.graph.state import AgentState
from src.data.medeinst_adapter import CasePair, load_pairs
from src.utils.runtime_config import configured_base_url, config_relative_path


@dataclass
class OnlineResult:
    """One case-level inference result."""
    case_id: str
    case_type: str
    status: str  # "success" | "error"
    ground_truth_id: Optional[str] = None
    ground_truth_name: Optional[str] = None
    phase1_diagnosis_id: Optional[str] = None
    phase1_diagnosis_name: Optional[str] = None
    final_diagnosis_id: Optional[str] = None
    final_diagnosis_name: Optional[str] = None
    verdict_status: Optional[str] = None  # "Confirm" | "Overturn" | "Fallback"
    is_correct: bool = False
    processing_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    hybrid_stats: Optional[Dict[str, int]] = None
    memory_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class OnlineOutputManager:
    """Persist Online results, summaries, and case graphs."""

    def __init__(self, output_dir: str = "output_online"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.graphs_dir = self.output_dir / "graphs"
        self.graphs_dir.mkdir(exist_ok=True)

        self.results_file = self.output_dir / "results_detail.jsonl"
        self.summary_file = self.output_dir / "inference_summary.json"
        self.error_log_file = self.output_dir / "error_log.jsonl"

        print(tr(f"[OutputManager] Output directory: {self.output_dir}",
                 f"[输出管理] 输出目录：{self.output_dir}"))

    def save_result(self, result: OnlineResult) -> None:

        with open(self.results_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def save_graph(self, case_id: str, graph_json: Dict[str, Any]) -> None:

        filepath = self.graphs_dir / f"{case_id}_online_graph.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(graph_json, f, indent=2, ensure_ascii=False)

    def save_error(self, case_id: str, case_type: str, error_message: str) -> None:

        error_record = {
            "case_id": case_id,
            "case_type": case_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    def save_summary(self, summary: Dict[str, Any]) -> None:

        with open(self.summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def load_processed_cases(self) -> Dict[str, str]:

        processed = {}
        if self.results_file.exists():
            with open(self.results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            key = f"{record['case_id']}_{record['case_type']}"
                            processed[key] = record.get("status", "unknown")
                        except:
                            pass
        return processed

    def get_error_cases(self) -> List[Tuple[str, str]]:

        error_cases = []
        processed = self.load_processed_cases()

        for key, status in processed.items():
            if status == "error":
                for variant in ("control_case", "trap_case"):
                    suffix = f"_{variant}"
                    if key.endswith(suffix):
                        error_cases.append((key[:-len(suffix)], variant))
                        break

        return error_cases

class OnlineInferenceRunner:
    """Coordinate the full Online DCI pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        output_manager: OnlineOutputManager,
        golden_graph_dir: str = "golden_graphs",
        refined_graph_dir: str = "golden_graphs_refined",
        memory_bank_dir: str = "memory_bank",
        parallel: bool = False,
        max_workers: int = 4,
        config_path: Optional[str] = None,
        verbose_timing: bool = False
    ):
        # Import runtime modules after CLI language selection, including any
        # optional-dependency warnings emitted during import.
        from src.utils.api_client import LLMClient
        from src.agents.phase1_manager import Phase1Manager
        from src.agents.phase2_hybrid_engine import Phase2HybridEngine
        from src.agents.phase3_memory_judge import create_judge_agent
        from src.memory.memory_bank import MemoryBankManager
        from src.memory.embedding_client import EmbeddingClient

        self.config = config
        self.output_manager = output_manager
        self.parallel = parallel
        self.max_workers = max_workers
        self.verbose_timing = verbose_timing

        api_config = config.get("api", {})
        self.llm_client = LLMClient(
            base_url=configured_base_url(api_config),
            api_key_env=api_config.get("api_key_env", "OPENAI_API_KEY"),
            timeout=api_config.get("timeout", 120)
        )
        self.model_name = api_config.get("model_name", "qwen3-32b")
        embedding_config = config.get("embedding", {})
        embedding_client = EmbeddingClient(
            base_url=configured_base_url(embedding_config),
            api_key_env=embedding_config.get("api_key_env", "OPENAI_API_KEY"),
            model=embedding_config.get("model_name", "text-embedding-3-small"),
            dimension=embedding_config.get("dimension", 1536)
        )

        self.phase1 = Phase1Manager(self.llm_client, self.model_name)

        # Online DCI starts from Illness Graphs, then extends them with live evidence.
        self.phase2 = Phase2HybridEngine(
            llm_client=self.llm_client,
            model_name=self.model_name,
            golden_graph_dir=golden_graph_dir,
            refined_graph_dir=refined_graph_dir,
            embedding_client=embedding_client,
            config_path=config_path
        )

        self.memory_bank = MemoryBankManager(
            output_dir=memory_bank_dir, embedding_client=embedding_client
        )
        try:
            self.memory_bank.load()
            print(tr(f"[OnlineRunner] Memory Bank loaded: {self.memory_bank.get_statistics()}",
                     f"[在线推理] 范例库已加载，统计信息：{self.memory_bank.get_statistics()}"))
        except Exception as e:
            print(tr(f"[OnlineRunner] Warning: Failed to load Memory Bank: {e}",
                     f"[在线推理] 警告：范例库加载失败：{e}"))
            self.memory_bank = None

        self.phase3 = create_judge_agent(
            llm_client=self.llm_client,
            model_name=self.model_name,
            memory_bank=self.memory_bank,
            memory_bank_dir=memory_bank_dir
        )

        print(tr("[OnlineRunner] Initialized with:", "[在线推理] 初始化完成："))
        print(tr(f"  - Model: {self.model_name}", f"  - 模型：{self.model_name}"))
        print(tr(f"  - Raw Illness Graphs: {golden_graph_dir}", f"  - 原始疾病图：{golden_graph_dir}"))
        print(tr(f"  - Memory Bank: {'Enabled' if self.memory_bank else 'Disabled'}",
                 f"  - 范例库：{'已启用' if self.memory_bank else '已禁用'}"))
        print(tr(f"  - Parallel: {parallel} (workers: {max_workers})",
                 f"  - 并行模式：{'是' if parallel else '否'}（工作线程：{max_workers}）"))

    def process_single_case(
        self,
        case_dir: CasePair,
        case_type: str
    ) -> OnlineResult:
        from src.utils.graph_tools import summarize_graph_for_judge, calculate_naive_scores

        case_id = case_dir.name
        start_time = time.time()
        # Per-phase processing times, measured on the calling thread so the
        # breakdown is correct even when cases run concurrently.
        phase_times: Dict[str, float] = {}

        try:
            case_info = case_dir.get(case_type)
            narrative = case_info.narrative
            ground_truth = case_info.ground_truth
            gt_id = case_info.ground_truth_id

            # Phase 1
            phase_start = time.time()
            phase1_result = self.phase1.process(narrative)
            phase_times["phase1"] = time.time() - phase_start

            if phase1_result.get("error"):
                raise Exception(f"Phase 1 error: {phase1_result['error']}")

            state: AgentState = {
                "case_id": case_id,
                "input_case": {
                    "narrative": narrative,
                    "ground_truth": ground_truth,
                    "ground_truth_id": gt_id
                },
                "phase1_result": phase1_result,
                "graph_json": None,
                "graph_summary": None,
                "naive_scores": None,
                "final_output": None,
                "status": "processing",
                "error_log": None,
                "global_hint": None,
                "retry_count": 0,
                "memory_context": None,
                "memory_records": None
            }

            # Phase 2 (Hybrid Engine)
            phase_start = time.time()
            state = self.phase2.process(state)

            if state.get("status") == "failed":
                phase_times["phase2"] = time.time() - phase_start
                raise Exception(f"Phase 2 error: {state.get('error_log')}")

            graph_json = state.get("graph_json", {})

            graph_summary = summarize_graph_for_judge(graph_json)
            state["graph_summary"] = graph_summary

            naive_scores = calculate_naive_scores(graph_json)
            state["naive_scores"] = naive_scores
            phase_times["phase2"] = time.time() - phase_start

            # Phase 3 (Memory-Augmented Judge)
            phase_start = time.time()
            state = self.phase3.process(state)
            phase_times["phase3"] = time.time() - phase_start

            if state.get("status") == "failed":
                raise Exception(f"Phase 3 error: {state.get('error_log')}")

            final_output = state.get("final_output", {})
            final_id = final_output.get("final_diagnosis_id")
            final_name = final_output.get("final_diagnosis_name")
            verdict_status = final_output.get("status")

            phase1_id = phase1_result.get("final_diagnosis_id")
            phase1_name = DIAGNOSIS_ID_MAP.get(phase1_id, "Unknown")

            is_correct = (final_id == gt_id) if gt_id else False

            self.output_manager.save_graph(f"{case_id}_{case_type}", graph_json)

            processing_time = time.time() - start_time
            phase_times["other"] = max(0.0, processing_time - sum(phase_times.values()))

            hybrid_stats = graph_json.get("hybrid_engine_stats", {})
            memory_used = final_output.get("memory_retrieval_used", False)

            return OnlineResult(
                case_id=case_id,
                case_type=case_type,
                status="success",
                ground_truth_id=gt_id,
                ground_truth_name=ground_truth,
                phase1_diagnosis_id=phase1_id,
                phase1_diagnosis_name=phase1_name,
                final_diagnosis_id=final_id,
                final_diagnosis_name=final_name,
                verdict_status=verdict_status,
                is_correct=is_correct,
                processing_time=processing_time,
                phase_times=phase_times,
                error_message=None,
                hybrid_stats=hybrid_stats,
                memory_used=memory_used
            )

        except Exception as e:
            processing_time = time.time() - start_time
            phase_times["other"] = max(0.0, processing_time - sum(phase_times.values()))
            error_msg = f"{str(e)}\n{traceback.format_exc()}"

            self.output_manager.save_error(case_id, case_type, error_msg)

            return OnlineResult(
                case_id=case_id,
                case_type=case_type,
                status="error",
                processing_time=processing_time,
                phase_times=phase_times,
                error_message=str(e)
            )

    def _report_case_timing(self, result: OnlineResult) -> None:
        """Print one case's total and per-phase times when detailed timing is on.

        Called from the main thread only (sequentially, or once per completed
        future in parallel mode), so output lines never interleave even when
        cases are processed by multiple workers.
        """
        if not self.verbose_timing or result.processing_time <= 0:
            return
        parts = [
            f"{key}={result.phase_times[key]:.2f}s"
            for key in ("phase1", "phase2", "phase3", "other")
            if key in result.phase_times
        ]
        tqdm.write(tr(
            f"⏱️  {result.case_id} ({result.case_type}): total={result.processing_time:.2f}s | {' '.join(parts)}",
            f"⏱️  {result.case_id}（{result.case_type}）：总耗时={result.processing_time:.2f} 秒 | {' '.join(parts)}"
        ))

    def run(
        self,
        case_dirs: List[CasePair],
        case_types: List[str],
        retry_failed: bool = False
    ) -> List[OnlineResult]:

        tasks = []

        if retry_failed:

            error_cases = self.output_manager.get_error_cases()
            for case_id, case_type in error_cases:
                if case_type not in case_types:
                    continue
                for case_dir in case_dirs:
                    if case_dir.name == case_id:
                        tasks.append((case_dir, case_type))
                        break
            print(tr(f"[OnlineRunner] Retrying {len(tasks)} failed cases", f"[在线推理] 重试 {len(tasks)} 个失败病例"))
        else:

            processed = self.output_manager.load_processed_cases()

            for case_dir in case_dirs:
                for case_type in case_types:
                    key = f"{case_dir.name}_{case_type}"
                    if key not in processed or processed[key] == "error":
                        tasks.append((case_dir, case_type))

            print(tr(f"[OnlineRunner] Processing {len(tasks)} cases", f"[在线推理] 处理 {len(tasks)} 个病例"))

        if not tasks:
            print(tr("[OnlineRunner] No cases to process", "[在线推理] 没有需要处理的病例"))
            return []

        if self.parallel:
            results = self._run_parallel(tasks)
        else:
            results = self._run_sequential(tasks)

        return results

    def _run_sequential(self, tasks: List[Tuple[Path, str]]) -> List[OnlineResult]:

        results = []

        with tqdm(total=len(tasks), desc=tr("Online Inference", "在线推理")) as pbar:
            for case_dir, case_type in tasks:
                result = self.process_single_case(case_dir, case_type)
                results.append(result)

                self.output_manager.save_result(result)
                self._report_case_timing(result)

                status_icon = "✓" if result.status == "success" else "✗"
                correct_icon = "🎯" if result.is_correct else ""
                pbar.set_postfix_str(f"{status_icon} {result.case_id} {correct_icon}")
                pbar.update(1)

        return results

    def _run_parallel(self, tasks: List[Tuple[Path, str]]) -> List[OnlineResult]:

        results = []

        with tqdm(total=len(tasks), desc=tr("Online Inference (Parallel)", "在线推理（并行）")) as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(self.process_single_case, case_dir, case_type): (case_dir, case_type)
                    for case_dir, case_type in tasks
                }

                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                    except Exception as e:
                        case_dir, case_type = future_to_task[future]
                        result = OnlineResult(
                            case_id=case_dir.name,
                            case_type=case_type,
                            status="error",
                            error_message=str(e)
                        )

                    results.append(result)
                    self.output_manager.save_result(result)
                    self._report_case_timing(result)

                    status_icon = "✓" if result.status == "success" else "✗"
                    pbar.set_postfix_str(f"{status_icon} {result.case_id}")
                    pbar.update(1)

        return results

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:

    config_path = Path(config_path or PROJECT_ROOT / "config" / "settings.yaml").expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PHASE_KEYS = ("phase1", "phase2", "phase3", "other")

def build_timing_report(results: List[OnlineResult], wall_time: float) -> Dict[str, Any]:
    """Aggregate per-case and per-phase processing times (all values in seconds)."""
    case_entries = []
    phase_samples: Dict[str, List[float]] = {key: [] for key in PHASE_KEYS}
    total_samples: List[float] = []

    for r in sorted(results, key=lambda x: (x.case_id, x.case_type)):
        entry = {
            "case_id": r.case_id,
            "case_type": r.case_type,
            "status": r.status,
            "total_seconds": round(r.processing_time, 4),
            "phase_times": {
                key: round(r.phase_times[key], 4)
                for key in PHASE_KEYS if key in r.phase_times
            },
        }
        case_entries.append(entry)
        if r.processing_time > 0:
            total_samples.append(r.processing_time)
        for key in PHASE_KEYS:
            if key in r.phase_times:
                phase_samples[key].append(r.phase_times[key])

    def stats(samples: List[float]) -> Dict[str, Any]:
        if not samples:
            return {"count": 0, "avg_seconds": 0.0, "min_seconds": 0.0, "max_seconds": 0.0}
        return {
            "count": len(samples),
            "avg_seconds": round(sum(samples) / len(samples), 4),
            "min_seconds": round(min(samples), 4),
            "max_seconds": round(max(samples), 4),
        }

    return {
        "wall_time_seconds": round(wall_time, 4),
        "sum_case_time_seconds": round(sum(total_samples), 4),
        "total_per_case": stats(total_samples),
        "phases": {key: stats(phase_samples[key]) for key in PHASE_KEYS},
        "cases": case_entries,
    }

def print_timing_report(report: Dict[str, Any], parallel: bool) -> None:
    """Print per-case and per-phase timing details (all values in seconds)."""
    total_stats = report["total_per_case"]
    phases = report["phases"]

    print(tr("\nDetailed timing report:", "\n详细耗时报告："))
    print(tr(
        f"  Wall-clock total time: {report['wall_time_seconds']:.2f}s",
        f"  总耗时（墙钟时间）：{report['wall_time_seconds']:.2f} 秒"
    ))
    print(tr(
        f"  Sum of per-case times: {report['sum_case_time_seconds']:.2f}s",
        f"  各病例耗时之和：{report['sum_case_time_seconds']:.2f} 秒"
    ))
    if parallel:
        print(tr(
            "  (Parallel mode: the wall-clock total is smaller than the sum of per-case times)",
            "  （并行模式：墙钟总耗时小于各病例耗时之和，因为病例是并发处理的）"
        ))
    print(tr("  Per-case time (seconds):", "  每个病例的处理时间（秒）："))
    print(tr(
        f"    {'case_id':<20} {'type':<13} {'total':>9} {'phase1':>9} {'phase2':>9} {'phase3':>9} {'other':>9}",
        f"    {'病例ID':<18} {'类型':<11} {'总耗时':>9} {'阶段1':>9} {'阶段2':>9} {'阶段3':>9} {'其他':>9}"
    ))
    for entry in report["cases"]:
        pt = entry["phase_times"]

        def fmt(key: str) -> str:
            return f"{pt[key]:.2f}" if key in pt else "-"

        print(
            f"    {entry['case_id']:<20} {entry['case_type']:<13} "
            f"{entry['total_seconds']:>8.2f} {fmt('phase1'):>9} {fmt('phase2'):>9} "
            f"{fmt('phase3'):>9} {fmt('other'):>9}"
        )
    print(tr(
        f"  Per-case total (avg / min / max): "
        f"{total_stats['avg_seconds']:.2f}s / {total_stats['min_seconds']:.2f}s / {total_stats['max_seconds']:.2f}s",
        f"  单个病例总耗时（平均 / 最小 / 最大）："
        f"{total_stats['avg_seconds']:.2f} 秒 / {total_stats['min_seconds']:.2f} 秒 / {total_stats['max_seconds']:.2f} 秒"
    ))
    print(tr("  Per-phase time (avg / min / max, seconds):",
             "  各阶段耗时（平均 / 最小 / 最大，秒）："))
    for key in PHASE_KEYS:
        s = phases[key]
        if not s["count"]:
            continue
        print(tr(
            f"    {key:<8} {s['avg_seconds']:>8.2f} / {s['min_seconds']:>8.2f} / {s['max_seconds']:>8.2f}  (n={s['count']})",
            f"    {key:<8} {s['avg_seconds']:>8.2f} / {s['min_seconds']:>8.2f} / {s['max_seconds']:>8.2f}  （样本数={s['count']}）"
        ))

def parse_args(argv=None):
    # Read language first so --console-language zh --help is also in Chinese.
    language_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    language_parser.add_argument("--console-language", default="en")
    preliminary, _ = language_parser.parse_known_args(argv)
    set_console_language(preliminary.console_language if preliminary.console_language in ("en", "zh") else "en")

    parser = ConsoleArgumentParser(
        description=tr("ECR-Agent Online DCI with Illness Graphs and Exemplar Base",
                       "ECR-Agent 在线诊断推理（疾病图 + 范例库）"),
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=tr("""
Examples:
  # Process all Benchmark Test Set cases
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl
  
  # Limit the number of pairs
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --limit 10
  
  # Process cases concurrently
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --parallel --workers 4
  
  # Retry failed cases
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --retry-failed

  # Detailed per-case and per-phase timing
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --limit 10 --detailed-timing
        """, """
示例:
  # 使用中文控制台输出，处理一组病例
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --limit 1 --console-language zh

  # 使用英文控制台输出（默认）
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --console-language en

  # 并行处理病例
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --parallel --workers 4 --console-language zh

  # 重试失败的病例
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --retry-failed --console-language zh

  # 输出每个病例及各阶段的详细耗时
  python scripts/run_ecr_agent.py --benchmark-set test.jsonl --limit 10 --detailed-timing --console-language zh
        """)
    )

    parser.add_argument(
        "--console-language", choices=("en", "zh"), default="en",
        help=tr("Console language: en=English, zh=Chinese (default: en; 中文请选择 zh)",
                "控制台语言：en=英文，zh=中文（默认：en）")
    )

    parser.add_argument(
        "--benchmark-set",
        type=str,
        required=True,
        help=tr("Public MedEinst test.jsonl (Benchmark Test Set), or a paired-case directory",
                "MedEinst 公共测试集 test.jsonl，或成对病例目录")
    )

    parser.add_argument(
        "--config", type=str, default=None,
        help=tr("YAML configuration file (default: config/settings.yaml)",
                "YAML 配置文件（默认：config/settings.yaml）")
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=tr("Output directory (overrides online_inference.output_dir)",
                "输出目录（覆盖 online_inference.output_dir）")
    )

    parser.add_argument(
        "--raw-graph-dir",
        type=str,
        default=None,
        help=tr("Raw Illness Graph directory (overrides online_inference.golden_graph_dir)",
                "原始疾病图目录（覆盖 online_inference.golden_graph_dir）")
    )

    parser.add_argument(
        "--refined-graph-dir", type=str, default=None,
        help=tr("Refined Illness Graph directory (overrides online_inference.refined_graph_dir)",
                "精化疾病图目录（覆盖 online_inference.refined_graph_dir）")
    )

    parser.add_argument(
        "--memory-bank-dir",
        type=str,
        default=None,
        help=tr("Exemplar Base directory (overrides online_inference.memory_bank_dir)",
                "范例库目录（覆盖 online_inference.memory_bank_dir）")
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=tr("Maximum number of case pairs to process", "最多处理的病例对数")
    )

    parser.add_argument(
        "--case-ids",
        type=str,
        default=None,
        help=tr("Comma-separated case pair IDs to process", "要处理的病例对 ID，以英文逗号分隔")
    )

    parser.add_argument(
        "--skip-control",
        action="store_true",
        help=tr("Skip control cases", "跳过对照病例")
    )

    parser.add_argument(
        "--skip-trap",
        action="store_true",
        help=tr("Skip trap cases", "跳过陷阱病例")
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help=tr("Process cases concurrently", "并行处理病例")
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=tr("Number of concurrent workers (default: 4)", "并行工作线程数（默认：4）")
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=tr("Retry only previously failed cases", "仅重试之前失败的病例")
    )

    parser.add_argument(
        "--detailed-timing",
        action="store_true",
        help=tr("Print detailed timing: per-case processing time and per-phase (Phase 1/2/3) time",
                "输出详细耗时信息：每个病例的处理时间，以及各阶段（阶段1/2/3）的处理时间")
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=tr("Override the base model ID in the configuration", "覆盖配置文件中的基座模型 ID")
    )

    return parser.parse_args(argv)

def main():
    configure_console_encoding()
    load_dotenv()

    args = parse_args()

    config_path = Path(args.config or PROJECT_ROOT / "config" / "settings.yaml").expanduser().resolve()
    config = load_config(config_path)
    online_config = config.get("online_inference", {})

    def runtime_path(cli_value: Optional[str], key: str, default: str) -> str:
        if cli_value:
            return str(Path(cli_value).expanduser().resolve())
        return config_relative_path(online_config.get(key, default), config_path)

    output_dir = runtime_path(args.output_dir, "output_dir", "../output_online")
    raw_graph_dir = runtime_path(args.raw_graph_dir, "golden_graph_dir", "../golden_graphs")
    refined_graph_dir = runtime_path(args.refined_graph_dir, "refined_graph_dir", "../golden_graphs_refined")
    memory_bank_dir = runtime_path(args.memory_bank_dir, "memory_bank_dir", "../memory_bank")

    if args.model:
        config["api"]["model_name"] = args.model

    output_manager = OnlineOutputManager(output_dir)

    # Load the Benchmark Test Set, preserving original pair IDs and narratives.
    input_path = Path(args.benchmark_set).expanduser()
    case_dirs = load_pairs(input_path)

    if args.case_ids:
        specified_ids = {item.strip() for item in args.case_ids.split(",")}
        missing_ids = specified_ids - {pair.case_id for pair in case_dirs}
        if missing_ids:
            raise ValueError(tr(f"Pair IDs absent from Benchmark Test Set: {sorted(missing_ids)}",
                                f"基准测试集中不存在以下病例对 ID：{sorted(missing_ids)}"))
        case_dirs = [pair for pair in case_dirs if pair.case_id in specified_ids]

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError(tr("--limit must be positive", "--limit 必须是正整数"))
        case_dirs = case_dirs[:args.limit]

    case_types = []
    if not args.skip_control:
        case_types.append("control_case")
    if not args.skip_trap:
        case_types.append("trap_case")

    if not case_types:
        print(tr("❌ No case types to process", "❌ 没有可处理的病例类型"))
        sys.exit(1)

    print("=" * 70)
    print(tr("ECR-Agent Online DCI (Illness Graphs + Exemplar Base)", "ECR-Agent 在线诊断推理（疾病图 + 范例库）"))
    print("=" * 70)
    print(tr(f"📁 Input directory: {input_path}", f"📁 输入路径：{input_path}"))
    print(tr(f"📁 Output directory: {output_manager.output_dir}", f"📁 输出目录：{output_manager.output_dir}"))
    print(tr(f"📁 Raw Illness Graphs: {raw_graph_dir}", f"📁 原始疾病图：{raw_graph_dir}"))
    print(tr(f"📁 Refined Illness Graphs: {refined_graph_dir}", f"📁 精化疾病图：{refined_graph_dir}"))
    print(tr(f"📁 Exemplar Base: {memory_bank_dir}", f"📁 范例库：{memory_bank_dir}"))
    print(tr(f"📊 Benchmark pairs selected: {len(case_dirs)}", f"📊 已选病例对数：{len(case_dirs)}"))
    case_labels = {"control_case": "对照病例", "trap_case": "陷阱病例"}
    print(tr(f"📋 Case types: {', '.join(case_types)}", f"📋 病例类型：{'、'.join(case_labels[t] for t in case_types)}"))
    print(tr(f"🔧 Model: {config['api']['model_name']}", f"🔧 模型：{config['api']['model_name']}"))
    print(tr("📊 Illness Graphs: Enabled", "📊 疾病图：已启用"))
    print(tr("🧠 Exemplar Base: Enabled", "🧠 范例库：已启用"))
    print(tr(f"⚡ Parallel mode: {'Yes' if args.parallel else 'No'}", f"⚡ 并行模式：{'是' if args.parallel else '否'}"))
    if args.parallel:
        print(tr(f"👷 Workers: {args.workers}", f"👷 工作线程数：{args.workers}"))
    if args.detailed_timing:
        print(tr("⏱️ Detailed timing: Enabled", "⏱️ 详细耗时输出：已启用"))
    print("=" * 70)

    try:
        runner = OnlineInferenceRunner(
            config=config,
            output_manager=output_manager,
            golden_graph_dir=raw_graph_dir,
            refined_graph_dir=refined_graph_dir,
            memory_bank_dir=memory_bank_dir,
            parallel=args.parallel,
            max_workers=args.workers,
            config_path=str(config_path),
            verbose_timing=args.detailed_timing
        )
    except Exception as e:
        print(tr(f"❌ Failed to initialize runner: {e}", f"❌ 推理程序初始化失败：{e}"))
        traceback.print_exc()
        sys.exit(1)

    start_time = time.time()
    results = runner.run(
        case_dirs=case_dirs,
        case_types=case_types,
        retry_failed=args.retry_failed
    )
    total_time = time.time() - start_time  # wall-clock seconds

    success_count = sum(1 for r in results if r.status == "success")
    error_count = sum(1 for r in results if r.status == "error")
    correct_count = sum(1 for r in results if r.is_correct)

    control_results = [r for r in results if r.case_type == "control_case"]
    trap_results = [r for r in results if r.case_type == "trap_case"]

    control_correct = sum(1 for r in control_results if r.is_correct)
    trap_correct = sum(1 for r in trap_results if r.is_correct)

    overturn_count = sum(1 for r in results if r.verdict_status == "Overturn")

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "total_processed": len(results),
        "success": success_count,
        "error": error_count,
        "correct": correct_count,
        "accuracy": correct_count / success_count if success_count > 0 else 0,
        "control_accuracy": control_correct / len(control_results) if control_results else 0,
        "trap_accuracy": trap_correct / len(trap_results) if trap_results else 0,
        "overturn_count": overturn_count,
        "total_time_seconds": total_time,
        "avg_time_per_case": total_time / len(results) if results else 0,
        "config": {
            "model": config["api"]["model_name"],
            "use_memory": True,
            "golden_graph_dir": raw_graph_dir,
            "refined_graph_dir": refined_graph_dir,
            "memory_bank_dir": memory_bank_dir,
            "parallel": args.parallel,
            "workers": args.workers if args.parallel else 1
        }
    }
    if args.detailed_timing:
        summary["timing_details"] = build_timing_report(results, total_time)
    output_manager.save_summary(summary)

    print("\n" + "=" * 70)
    print(tr("Online Inference Complete!", "在线推理完成！"))
    print("=" * 70)
    print(tr(f"✅ Success: {success_count}", f"✅ 成功：{success_count}"))
    print(tr(f"❌ Error: {error_count}", f"❌ 失败：{error_count}"))
    print(tr(f"🎯 Correct: {correct_count}/{success_count} ({summary['accuracy']:.2%})",
             f"🎯 诊断正确：{correct_count}/{success_count}（{summary['accuracy']:.2%}）"))
    if control_results:
        print(tr(f"  - Control: {control_correct}/{len(control_results)} ({summary['control_accuracy']:.2%})",
                 f"  - 对照病例：{control_correct}/{len(control_results)}（{summary['control_accuracy']:.2%}）"))
    if trap_results:
        print(tr(f"  - Trap: {trap_correct}/{len(trap_results)} ({summary['trap_accuracy']:.2%})",
                 f"  - 陷阱病例：{trap_correct}/{len(trap_results)}（{summary['trap_accuracy']:.2%}）"))
    print(tr(f"🔄 Overturn: {overturn_count}", f"🔄 推翻初始诊断：{overturn_count}"))
    print(tr(f"⏱️  Total time: {total_time:.1f}s", f"⏱️  总耗时：{total_time:.1f} 秒"))
    if args.detailed_timing and results:
        print_timing_report(summary["timing_details"], parallel=args.parallel)
    print(tr(f"📊 Results saved to: {output_manager.output_dir}", f"📊 结果已保存至：{output_manager.output_dir}"))
    print("=" * 70)

if __name__ == "__main__":
    main()

