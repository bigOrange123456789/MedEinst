"""Launcher regression checks for local settings in a custom YAML file."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, call, patch

from scripts import refine_illness_graphs, run_cgme, run_ecr_agent


class LocalLauncherTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.config_path = self.root / "custom.yaml"
        self.config = {
            "api": {"localhost": True, "model_path": "./models/main", "model_name": "local-main"},
            "critical_model": {"localhost": True, "model_path": "./models/critic", "model_name": "local-critic"},
            "embedding": {"enabled": False},
            "offline_training": {"golden_graph_dir": "./graphs", "log_dir": "./logs"},
            "online_inference": {"refined_graph_dir": "./refined"},
        }
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_online_passes_custom_settings_to_both_client_factories(self):
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            llm_factory = stack.enter_context(patch("src.utils.api_client.LLMClient.from_config"))
            embedding_factory = stack.enter_context(patch("src.memory.embedding_client.EmbeddingClient.from_config"))
            stack.enter_context(patch("src.agents.phase1_manager.Phase1Manager"))
            hybrid = stack.enter_context(patch("src.agents.phase2_hybrid_engine.Phase2HybridEngine"))
            memory = stack.enter_context(patch("src.memory.memory_bank.MemoryBankManager"))
            judge = stack.enter_context(patch("src.agents.phase3_memory_judge.create_judge_agent"))

            runner = run_ecr_agent.OnlineInferenceRunner(
                self.config, Mock(), config_path=str(self.config_path),
                memory_bank_dir=str(self.root / "memory"),
            )

        llm_factory.assert_called_once_with(
            self.config["api"], config_path=self.config_path, default_timeout=120)
        embedding_factory.assert_called_once_with(
            self.config["embedding"], config_path=self.config_path)
        self.assertIs(runner.llm_client, llm_factory.return_value)
        self.assertIs(hybrid.call_args.kwargs["embedding_client"], embedding_factory.return_value)
        self.assertIs(memory.call_args.kwargs["embedding_client"], embedding_factory.return_value)
        self.assertIs(judge.call_args.kwargs["llm_client"], llm_factory.return_value)

    def test_cgme_passes_main_critic_and_disabled_embeddings_to_trainers(self):
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent), ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                argv = ["run_cgme.py", "--config", str(self.config_path), "--reference-set", "unused.jsonl"]
                if concurrent:
                    argv.append("--concurrent")
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(run_cgme, "load_pairs", return_value=[]))
                stack.enter_context(patch.object(run_cgme, "resolve_support_manifest", return_value=[]))
                main_client, critic_client, embedding_client = Mock(), Mock(), Mock()
                llm_factory = stack.enter_context(patch.object(
                    run_cgme.LLMClient, "from_config", side_effect=[main_client, critic_client]))
                embedding_factory = stack.enter_context(patch.object(
                    run_cgme.EmbeddingClient, "from_config", return_value=embedding_client))
                trainer_name = "ConcurrentOfflineTrainer" if concurrent else "OfflineTrainer"
                trainer = stack.enter_context(patch.object(run_cgme, trainer_name))

                run_cgme.main()

                self.assertEqual(llm_factory.call_args_list, [
                    call(self.config["api"], config_path=self.config_path, default_timeout=120),
                    call(self.config["critical_model"], config_path=self.config_path, default_timeout=180),
                ])
                embedding_factory.assert_called_once_with(
                    self.config["embedding"], config_path=self.config_path)
                self.assertIs(trainer.call_args.kwargs["llm_client"], main_client)
                self.assertIs(trainer.call_args.kwargs["critical_client"], critic_client)
                self.assertIs(trainer.call_args.kwargs["embedding_client"], embedding_client)
                self.assertEqual(trainer.call_args.kwargs["golden_graph_dir"], str(self.root / "graphs"))

    def test_refinement_uses_critic_section_from_custom_yaml(self):
        graph_dir = self.root / "graphs"
        graph_dir.mkdir()
        (graph_dir / "01_GG.json").write_text("{}", encoding="utf-8")
        argv = ["refine_illness_graphs.py", "--config", str(self.config_path), "--dry-run"]
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                patch.object(refine_illness_graphs.LLMClient, "from_config") as factory, \
                patch.object(refine_illness_graphs, "process_single_graph", return_value={"status": "dry_run"}) as process:
            refine_illness_graphs.main()
        factory.assert_called_once_with(
            self.config["critical_model"], config_path=self.config_path, default_timeout=180)
        self.assertIs(process.call_args.kwargs["client"], factory.return_value)
        self.assertEqual(process.call_args.kwargs["model"], "local-critic")


if __name__ == "__main__":
    unittest.main()
