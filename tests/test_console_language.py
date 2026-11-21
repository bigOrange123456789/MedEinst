"""Offline checks for console localization without changing inference data."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run_ecr_agent
from scripts.run_ecr_agent import OnlineResult, parse_args
from src.agents.phase1_manager import Phase1Manager
from src.utils.console import set_console_language, tr


ROOT = Path(__file__).resolve().parents[1]


class ConsoleLanguageTests(unittest.TestCase):
    def tearDown(self):
        set_console_language("en")

    def test_default_and_explicit_language_selection(self):
        base = ["--benchmark-set", "unused.jsonl"]
        self.assertEqual(parse_args(base + ["--console-language", "zh"]).console_language, "zh")
        self.assertEqual(tr("English", "中文"), "中文")
        self.assertEqual(parse_args(base).console_language, "en")
        self.assertEqual(tr("English", "中文"), "English")
        self.assertEqual(parse_args(base + ["--console-language=en"]).console_language, "en")

    def test_help_works_in_both_languages_and_argument_orders(self):
        for language, expected in (("en", "Maximum number of case pairs"), ("zh", "最多处理的病例对数")):
            for args in (["--help", "--console-language", language],
                         [f"--console-language={language}", "--help"]):
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "scripts" / "run_ecr_agent.py"), *args],
                        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(expected, result.stdout)
                    self.assertIn("--console-language {en,zh}", result.stdout)

    def test_invalid_language_is_rejected(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            parse_args(["--benchmark-set", "unused.jsonl", "--console-language", "fr"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())
        self.assertIn("--console-language", stderr.getvalue())

    def test_language_reaches_worker_threads(self):
        # A thread-local/context-local language would silently revert to English.
        with ThreadPoolExecutor(max_workers=2) as executor:
            for language, expected in (("zh", "处理中"), ("en", "Processing")):
                set_console_language(language)
                outputs = list(executor.map(lambda _: tr("Processing", "处理中"), range(4)))
                self.assertEqual(outputs, [expected] * 4)

    def test_phase1_logs_change_but_inputs_and_results_do_not(self):
        manager = Phase1Manager(Mock(), "test-model")
        manager.track_a = Mock()
        manager.track_a.process.return_value = {
            "top_candidates": ["45"], "final_diagnosis_id": "45",
            "differential_reasoning": "Original model reasoning.", "error": None,
        }
        manager.track_b = Mock()
        manager.track_b.process.return_value = {
            "problem_representation_one_liner": "Original observation.",
            "p_nodes": [{"content": "Original clinical detail."}], "error": None,
        }
        results, logs = {}, {}
        for language in ("en", "zh"):
            set_console_language(language)
            output = io.StringIO()
            with redirect_stdout(output):
                results[language] = manager.process("Original narrative.", global_hint="Original hint.")
            logs[language] = output.getvalue()
            manager.track_a.process.assert_called_with("Original narrative.", global_hint="Original hint.")
            manager.track_b.process.assert_called_with(
                "Original narrative.", age=None, sex=None, global_hint="Original hint.")

        self.assertEqual(results["en"], results["zh"])
        self.assertIn("Starting Dual-Track Processing", logs["en"])
        self.assertNotIn("Starting Dual-Track Processing", logs["zh"])
        self.assertRegex(logs["zh"], r"[\u4e00-\u9fff]")
        self.assertIn("Pneumonia", logs["zh"])

    def test_external_error_details_are_preserved(self):
        manager = Phase1Manager(Mock(), "test-model")
        manager.track_a = Mock()
        manager.track_a.process.return_value = {"error": "Rate limit reached: request-123"}
        set_console_language("zh")
        output = io.StringIO()
        with redirect_stdout(output):
            result = manager.process("Original narrative.")
        self.assertRegex(output.getvalue(), r"[\u4e00-\u9fff]")
        self.assertIn("Rate limit reached: request-123", output.getvalue())
        self.assertEqual(result["error"], "Track A failed: Rate limit reached: request-123")

    def test_main_localizes_serial_and_parallel_runs_without_changing_results(self):
        def fake_init(runner, **kwargs):
            runner.output_manager = kwargs["output_manager"]
            runner.parallel = kwargs["parallel"]
            runner.max_workers = kwargs["max_workers"]
            runner.verbose_timing = kwargs.get("verbose_timing", False)

        def fake_case(runner, pair, case_type):
            return OnlineResult(
                pair.case_id, case_type, "success", final_diagnosis_id="45",
                final_diagnosis_name="Pneumonia", verdict_status="Confirm", is_correct=True,
            )

        saved_results = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            dataset = path / "test.jsonl"
            dataset.write_text("".join(json.dumps({
                "case_id": "case_1", "case_type": case_type,
                "narrative": "Original narrative.", "ground_truth": "Pneumonia",
            }) + "\n" for case_type in ("control", "trap")), encoding="utf-8")
            config = path / "settings.yaml"
            config.write_text(json.dumps({"api": {"model_name": "test-model"}}), encoding="utf-8")

            for parallel in (False, True):
                for language in ("en", "zh"):
                    with self.subTest(language=language, parallel=parallel):
                        output_dir = path / f"{language}_{parallel}"
                        argv = ["run_ecr_agent.py", "--benchmark-set", str(dataset),
                                "--config", str(config), "--output-dir", str(output_dir),
                                "--limit", "1", "--console-language", language]
                        if parallel:
                            argv += ["--parallel", "--workers", "2"]
                        stdout, stderr = io.StringIO(), io.StringIO()
                        with patch.object(sys, "argv", argv), redirect_stdout(stdout), redirect_stderr(stderr), \
                                patch.object(run_ecr_agent.OnlineInferenceRunner, "__init__", fake_init), \
                                patch.object(run_ecr_agent.OnlineInferenceRunner, "process_single_case", fake_case):
                            run_ecr_agent.main()

                        self.assertIn("已选病例对数：1" if language == "zh" else "Benchmark pairs selected: 1", stdout.getvalue())
                        self.assertIn("在线推理完成" if language == "zh" else "Online Inference Complete", stdout.getvalue())
                        self.assertIn("在线推理" if language == "zh" else "Online Inference", stderr.getvalue())
                        records = [json.loads(line) for line in (output_dir / "results_detail.jsonl").read_text(encoding="utf-8").splitlines()]
                        saved_results.append(sorted(records, key=lambda record: record["case_type"]))
                        summary = json.loads((output_dir / "inference_summary.json").read_text(encoding="utf-8"))
                        self.assertEqual(summary["success"], 2)
                        self.assertEqual(summary["accuracy"], 1.0)
                        self.assertEqual(summary["config"]["parallel"], parallel)
        self.assertTrue(all(records == saved_results[0] for records in saved_results))


if __name__ == "__main__":
    unittest.main()
