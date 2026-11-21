"""Embedding configuration must not force local LLM runs to use a cloud API."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np

from src.memory.embedding_client import EmbeddingClient
from src.memory.memory_bank import MemoryBankManager
from src.utils.text_utils import SemanticMatcher


class EmbeddingConfigurationTests(unittest.TestCase):
    def test_loopback_endpoint_needs_no_cloud_key(self):
        for endpoint in ("http://localhost:11434/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"):
            with self.subTest(endpoint=endpoint), patch.dict(os.environ, {}, clear=True), \
                    patch("src.memory.embedding_client.OpenAI") as constructor:
                constructor.return_value.embeddings.create.return_value = SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.25, 0.75])]
                )
                client = EmbeddingClient.from_config({
                    "base_url": endpoint, "model_name": "local-embed", "dimension": 2,
                })
                np.testing.assert_array_equal(client.embed_text("  observation  "), [0.25, 0.75])
                self.assertEqual(constructor.call_args.kwargs["api_key"], "local")
                constructor.return_value.embeddings.create.assert_called_once_with(
                    model="local-embed", input="observation"
                )

    def test_remote_endpoint_still_requires_key(self):
        with patch.dict(os.environ, {}, clear=True), patch("src.memory.embedding_client.OpenAI") as constructor:
            with self.assertRaisesRegex(ValueError, "API key"):
                EmbeddingClient.from_config({"base_url": "https://example.invalid/v1"})
            constructor.assert_not_called()

    def test_config_accepts_explicit_credentials_and_env_url(self):
        with patch.dict(os.environ, {"EMBED_URL": "http://127.0.0.1:8080/v1"}), \
                patch("src.memory.embedding_client.OpenAI") as constructor:
            client = EmbeddingClient.from_config({
                "api_key": "explicit-test-key", "api_key_env": None,
                "base_url": "https://example.invalid/v1", "base_url_env": "EMBED_URL",
                "timeout": 12, "max_retries": 1, "retry_delay": 0,
            })
            self.assertEqual(constructor.call_args.kwargs["base_url"], "http://127.0.0.1:8080/v1")
            self.assertEqual(constructor.call_args.kwargs["api_key"], "explicit-test-key")
            self.assertEqual(constructor.call_args.kwargs["timeout"], 12)
            self.assertEqual(client.max_retries, 1)

    def test_disabled_embeddings_never_create_http_client_or_lazy_matcher(self):
        with patch("src.memory.embedding_client.OpenAI") as constructor, \
                patch.object(SemanticMatcher, "_init_embedding_client") as initialize:
            client = EmbeddingClient.from_config({"enabled": "false", "dimension": 8})
            self.assertFalse(client.enabled)
            self.assertEqual(client.dimension, 8)
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                client.embed_text("observation")
            matcher = SemanticMatcher(embedding_client=client)
            self.assertIsNone(matcher.get_embedding("observation"))
            self.assertEqual(matcher.get_embeddings_batch(["observation"]), {})
            constructor.assert_not_called()
            initialize.assert_not_called()

    def test_matcher_default_initialization_honors_disabled_config(self):
        config = json.dumps({"embedding": {"enabled": False}})
        with patch("builtins.open", mock_open(read_data=config)), \
                patch("src.memory.embedding_client.OpenAI") as constructor:
            matcher = SemanticMatcher()
            self.assertFalse(matcher._embedding_available)
            self.assertIsNone(matcher.get_embedding("observation"))
            constructor.assert_not_called()

    def test_matcher_default_initialization_accepts_local_no_key_config(self):
        config = json.dumps({"embedding": {"base_url": "http://localhost:8000/v1"}})
        with patch("builtins.open", mock_open(read_data=config)), \
                patch.dict(os.environ, {}, clear=True), \
                patch("src.memory.embedding_client.OpenAI"):
            self.assertTrue(SemanticMatcher()._embedding_available)


class DisabledMemoryTests(unittest.TestCase):
    @staticmethod
    def add_case(manager):
        manager.add_case(
            case_id="test", case_type="control", ground_truth_id="45",
            ground_truth_name="test", initial_diagnosis_id="45", initial_diagnosis_name="test",
            final_diagnosis_id="45", final_diagnosis_name="test",
            p_nodes=[{"content": "Fever", "status": "Present"}],
        )

    def test_disabled_memory_persists_zero_vectors_without_retrieving(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("src.memory.embedding_client.OpenAI") as constructor:
            client = EmbeddingClient.from_config({"enabled": False, "dimension": 3})
            manager = MemoryBankManager(directory, embedding_client=client)
            self.add_case(manager)
            np.testing.assert_array_equal(manager.embeddings[0], np.zeros(3))
            self.assertEqual(manager.retrieve_similar([]), {"overturn": [], "confirm": []})
            manager.save()
            reloaded = MemoryBankManager(directory, embedding_client=client)
            reloaded.load()
            self.assertEqual(reloaded.retrieve_similar([]), {"overturn": [], "confirm": []})
            constructor.assert_not_called()

    def test_disabled_rows_are_not_retrieved_after_enabling_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = MemoryBankManager(directory, embedding_client=EmbeddingClient(enabled=False, dimension=3))
            self.add_case(manager)
            manager._embedding_client = Mock(enabled=True, dimension=3)
            manager._embedding_client.embed_text.return_value = np.array([1, 0, 0], dtype=np.float32)
            self.assertEqual(manager.retrieve_similar([]), {"overturn": [], "confirm": []})

    def test_different_local_embedding_dimension_skips_incompatible_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = MemoryBankManager(directory, embedding_client=EmbeddingClient(enabled=False, dimension=3))
            self.add_case(manager)
            manager._embedding_client = Mock(enabled=True, dimension=2)
            manager._embedding_client.embed_text.return_value = np.array([1, 0], dtype=np.float32)
            self.assertEqual(manager.retrieve_similar([]), {"overturn": [], "confirm": []})
            self.add_case(manager)
            self.assertEqual(np.stack(manager.embeddings).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
