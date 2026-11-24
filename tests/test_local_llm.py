"""Exercise local model routing and generation without downloading weights."""

import io
import json
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from threading import Event, Lock
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from src.agents.phase1_manager import Phase1Manager
from src.agents.phase1_perception import Phase1TrackAAgent
from src.utils.api_client import LLMClient
from src.utils.local_llm import LocalTransformersBackend


class TensorBatch(dict):
    """Minimal BatchEncoding-like container; tokens remain observable in tests."""

    def __init__(self, tokens=None):
        if tokens is None:
            super().__init__(input_ids=SimpleNamespace(shape=(1, 3)), attention_mask=[1, 1, 1])
        else:
            super().__init__(input_ids=TokenRows([tokens]), attention_mask=TokenRows([[1] * len(tokens)]))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class TokenRows(list):
    @property
    def shape(self):
        return len(self), len(self[0])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, columns = key
            return TokenRows(row[columns] for row in super().__getitem__(rows))
        return super().__getitem__(key)


class MutablePrefixCache:
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def get_seq_length(self):
        return len(self.tokens)


class LocalLLMTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model_path = self.root / "weights"
        self.model_path.mkdir()
        self.adapter_path = self.root / "adapter"
        self.adapter_path.mkdir()
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.openai_patch = patch("src.utils.api_client.OpenAI")
        self.openai = self.openai_patch.start()
        self.addCleanup(self.openai_patch.stop)

    def fake_runtime(self, model_type="qwen3", text='{"ok": true}', cuda=False, mps=False):
        self.torch = ModuleType("torch")
        self.torch.inference_mode = Mock(side_effect=nullcontext)
        self.torch.cuda = SimpleNamespace(
            is_available=Mock(return_value=cuda),
            get_device_name=Mock(return_value="Test GPU"),
        )
        self.torch.backends = SimpleNamespace(mps=SimpleNamespace(
            is_available=Mock(return_value=mps),
        ))
        for dtype in ("float16", "float32", "float64", "bfloat16"):
            setattr(self.torch, dtype, object())
        self.inputs = TensorBatch()
        self.tokenizer = Mock(pad_token_id=0, eos_token_id=9)
        self.tokenizer.apply_chat_template.return_value = "formatted conversation"
        self.tokenizer.return_value = self.inputs
        self.tokenizer.decode.return_value = text
        self.model = Mock(device="cuda:0" if cuda else "mps" if mps else "cpu")
        self.model.generate.return_value = [[11, 12, 13, 90, 91]]
        self.transformers = ModuleType("transformers")
        self.transformers.AutoConfig = SimpleNamespace(from_pretrained=Mock(
            return_value=SimpleNamespace(model_type=model_type, architectures=[])))
        self.transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer))
        self.transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(return_value=self.model))
        self.transformers.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=Mock(return_value=self.model))
        modules = patch.dict(sys.modules, {"torch": self.torch, "transformers": self.transformers})
        modules.start()
        self.addCleanup(modules.stop)

    def client(self, **kwargs):
        return LLMClient(localhost=True, model_path=str(self.model_path), **kwargs)

    def fake_prefix_runtime(self):
        self.fake_runtime(model_type="qwen2")
        self.torch.equal = lambda left, right: left == right
        self.model.config = SimpleNamespace(model_type="qwen2", is_encoder_decoder=False, use_cache=True)
        self.model.generation_config = SimpleNamespace(use_cache=None, cache_implementation=None)
        self.tokenizer.apply_chat_template.side_effect = (
            lambda messages, *, add_generation_prompt, **_: (
                "".join(message["content"] for message in messages) + ("!" if add_generation_prompt else "")
            )
        )

        def tokenize(prompt, **_):
            # Reproduce a token boundary that changes when a particular suffix is appended.
            return TensorBatch(list(map(ord, prompt.replace("Az", "qz"))))

        def prefill(*, input_ids, **_):
            cache = MutablePrefixCache(input_ids[0])
            self.built_caches.append(cache)
            return SimpleNamespace(past_key_values=cache)

        def generate(*, input_ids, past_key_values=None, **_):
            if past_key_values is not None:
                self.cache_inputs.append(past_key_values.tokens.copy())
                past_key_values.tokens.extend(input_ids[0][past_key_values.get_seq_length():])
            return [input_ids[0] + [90, 91]]

        self.tokenizer.side_effect = tokenize
        self.built_caches = []
        self.cache_inputs = []
        self.model.side_effect = prefill
        self.model.generate.side_effect = generate

    def prefix_messages(self, case, system="S"):
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": case},
        ]

    def test_local_mode_needs_no_api_key_and_loads_dependencies_lazily(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None, "peft": None}):
            client = self.client()
        self.assertTrue(client.is_local)
        self.assertIsNone(client.client)
        self.assertIsNone(client.local_backend.model)
        self.openai.assert_not_called()

    def test_string_false_keeps_remote_mode(self):
        client = LLMClient(localhost="false", api_key="test-only")
        self.assertFalse(client.is_local)
        self.assertIsNone(client.local_backend)
        self.openai.assert_called_once()

    def test_yaml_paths_resolve_relative_to_config_not_working_directory(self):
        config_dir = self.root / "config"
        config_dir.mkdir()
        config_path = config_dir / "settings.yaml"
        config_path.write_text("""api:
  localhost: 'true'
  model_path: '../weights'
  lora_adapter: '../adapter'
  torch_dtype: float32
  max_new_tokens: 73
  chat_template_kwargs:
    enable_thinking: false
""", encoding="utf-8")
        section = yaml.safe_load(config_path.read_text(encoding="utf-8"))["api"]
        client = LLMClient.from_config(section, config_path=config_path)
        self.assertEqual(client.local_backend.model_path, str(self.model_path.resolve()))
        self.assertEqual(client.local_backend.lora_adapter, str(self.adapter_path.resolve()))
        self.assertEqual(client.generation_config["max_new_tokens"], 73)
        self.assertEqual(client.chat_template_kwargs, {"enable_thinking": False})

    def test_bad_model_and_adapter_paths_fail_before_inference(self):
        with self.assertRaisesRegex(ValueError, "model_path"):
            LLMClient(localhost=True)
        with self.assertRaisesRegex(FileNotFoundError, "model_path"):
            LLMClient(localhost=True, model_path=str(self.root / "missing"))
        with self.assertRaisesRegex(FileNotFoundError, "lora_adapter"):
            self.client(lora_adapter=str(self.root / "missing"))
        self.openai.assert_not_called()

    def test_cuda_has_priority_over_mps_and_reports_selected_gpu(self):
        self.fake_runtime(cuda=True, mps=True)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            client = self.client()
            result = client.generate_json([], "local")
        self.assertIsNone(result["error"])
        self.assertEqual(client.local_backend.device, "cuda:0")
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["device_map"],
                         "cuda:0")
        self.assertEqual(self.inputs.device, "cuda:0")
        self.torch.cuda.is_available.assert_called_once_with()
        self.torch.cuda.get_device_name.assert_called_once_with(0)
        self.torch.backends.mps.is_available.assert_not_called()
        self.assertIn("GPU (CUDA, cuda:0, Test GPU)", stdout.getvalue())

    def test_mps_is_used_when_cuda_is_unavailable(self):
        self.fake_runtime(mps=True)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            client = self.client()
            result = client.generate_json([], "local")
        self.assertIsNone(result["error"])
        self.assertEqual(client.local_backend.device, "mps")
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["device_map"],
                         "mps")
        self.assertEqual(self.inputs.device, "mps")
        self.torch.backends.mps.is_available.assert_called_once_with()
        self.torch.cuda.get_device_name.assert_not_called()
        self.assertIn("GPU (Apple MPS, mps)", stdout.getvalue())

    def test_cpu_is_used_and_reported_when_no_gpu_is_available(self):
        self.fake_runtime()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            client = self.client()
            result = client.generate_json([], "local")
        self.assertIsNone(result["error"])
        self.assertEqual(client.local_backend.device, "cpu")
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["device_map"],
                         "cpu")
        self.assertEqual(self.inputs.device, "cpu")
        self.assertIn("CPU (cpu)", stdout.getvalue())
        self.assertNotIn("GPU", stdout.getvalue())

    def test_legacy_yaml_device_map_cannot_override_automatic_selection(self):
        self.fake_runtime(cuda=True)
        client = LLMClient.from_config({
            "localhost": True, "model_path": str(self.model_path), "device_map": "cpu",
        })
        result = client.generate_json([], "local")
        self.assertIsNone(result["error"])
        self.assertEqual(client.local_backend.device, "cuda:0")
        self.assertEqual(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["device_map"],
                         "cuda:0")

    def test_http_mode_never_probes_or_reports_local_devices(self):
        self.fake_runtime(cuda=True)
        self.openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            client = LLMClient(base_url="http://localhost:8000/v1")
            result = client.generate_json([], "served-model", prefix_message_count=3)
        self.assertIsNone(result["error"])
        self.torch.cuda.is_available.assert_not_called()
        self.torch.backends.mps.is_available.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")
        request = self.openai.return_value.chat.completions.create.call_args.kwargs
        self.assertNotIn("prefix_message_count", request)

    def test_causal_generation_decodes_only_new_tokens_and_preserves_pad_zero(self):
        self.fake_runtime(text='private reasoning</think> {"ok": true}')
        client = self.client(chat_template_kwargs={"enable_thinking": False})
        messages = [{"role": "user", "content": "Return JSON."}]
        result = client.generate_json(messages, "local-label", max_tokens=19, temperature=0)
        self.assertIsNone(result["error"])
        self.assertEqual(json.loads(result["content"]), {"ok": True})
        self.assertEqual(result["response"].choices[0].message.content, result["content"])
        self.assertIsNone(result["response"].choices[0].logprobs)
        self.tokenizer.decode.assert_called_once_with([90, 91], skip_special_tokens=True)
        self.tokenizer.assert_called_once_with(
            "formatted conversation", return_tensors="pt", add_special_tokens=False)
        self.tokenizer.apply_chat_template.assert_called_once_with(
            messages, enable_thinking=False, tokenize=False, add_generation_prompt=True)
        request = self.model.generate.call_args.kwargs
        self.assertEqual(request["max_new_tokens"], 19)
        self.assertEqual(request["pad_token_id"], 0)
        self.assertFalse(request["do_sample"])
        self.assertNotIn("temperature", request)
        self.assertNotIn("top_p", request)
        self.assertNotIn("eos_token_id", request)
        self.assertNotIn("past_key_values", request)
        self.model.assert_not_called()
        self.assertEqual(self.inputs.device, "cpu")
        self.model.eval.assert_called_once()
        self.torch.inference_mode.assert_called_once()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        self.transformers.AutoModelForImageTextToText.from_pretrained.assert_not_called()
        for loader in (self.transformers.AutoConfig, self.transformers.AutoTokenizer,
                       self.transformers.AutoModelForCausalLM):
            self.assertTrue(loader.from_pretrained.call_args.kwargs["local_files_only"])
        self.openai.assert_not_called()

    def test_prefix_cache_reuses_prefill_without_mixing_cases(self):
        self.fake_prefix_runtime()
        client = self.client()

        for case in ("x", "y"):
            result = client.generate_json(self.prefix_messages(case), "local", prefix_message_count=3)
            self.assertIsNone(result["error"])

        prefix = list(map(ord, "SUA"))
        self.assertEqual(len(self.built_caches), 1)
        self.assertEqual(self.built_caches[0].tokens, prefix)
        self.assertEqual(self.cache_inputs, [prefix, prefix])
        requests = [call.kwargs for call in self.model.generate.call_args_list]
        self.assertIsNot(requests[0]["past_key_values"], requests[1]["past_key_values"])
        self.assertEqual([request["input_ids"][0] for request in requests],
                         [list(map(ord, text)) for text in ("SUAx!", "SUAy!")])
        self.assertEqual([call.args[0] for call in self.tokenizer.decode.call_args_list], [[90, 91], [90, 91]])

    def test_prefix_cache_does_not_apply_to_changed_or_mismatched_prefix(self):
        self.fake_prefix_runtime()
        client = self.client()

        for messages in (self.prefix_messages("x"), self.prefix_messages("z"),
                         self.prefix_messages("y", system="T")):
            result = client.generate_json(messages, "local", prefix_message_count=3)
            self.assertIsNone(result["error"])

        mismatched = self.model.generate.call_args_list[1].kwargs
        self.assertNotIn("past_key_values", mismatched)
        self.assertEqual(self.cache_inputs, [list(map(ord, prefix)) for prefix in ("SUA", "TUA")])
        self.assertEqual(len(self.built_caches), 2)

    def test_prefix_cache_respects_model_cache_disable_setting(self):
        self.fake_prefix_runtime()
        self.model.generation_config.use_cache = False

        result = self.client().generate_json(self.prefix_messages("x"), "local", prefix_message_count=3)

        self.assertIsNone(result["error"])
        self.model.assert_not_called()
        self.assertNotIn("past_key_values", self.model.generate.call_args.kwargs)

    def test_prefix_cache_generation_failure_falls_back_to_normal_generation(self):
        self.fake_prefix_runtime()
        client = self.client()
        normal_generate = self.model.generate.side_effect

        def reject_prefix_cache(*, past_key_values=None, **kwargs):
            if past_key_values is not None:
                raise TypeError("This model version cannot continue from a cache")
            return normal_generate(**kwargs)

        self.model.generate.side_effect = reject_prefix_cache
        with redirect_stdout(io.StringIO()):
            for case in ("x", "y"):
                result = client.generate_json(self.prefix_messages(case), "local", prefix_message_count=3)
                self.assertIsNone(result["error"])

        requests = [call.kwargs for call in self.model.generate.call_args_list]
        self.assertEqual(["past_key_values" in request for request in requests], [True, False, False])

    def test_generation_settings_override_agent_defaults_and_enable_sampling(self):
        self.fake_runtime()
        client = self.client(generation_config={
            "max_new_tokens": 71, "temperature": 0.3, "top_p": 0.8,
            "repetition_penalty": 1.15,
        })
        result = client.generate_json([], "local", max_tokens=999, temperature=0)
        self.assertIsNone(result["error"])
        request = self.model.generate.call_args.kwargs
        self.assertEqual(request["max_new_tokens"], 71)
        self.assertEqual(request["temperature"], 0.3)
        self.assertEqual(request["top_p"], 0.8)
        self.assertEqual(request["repetition_penalty"], 1.15)
        self.assertTrue(request["do_sample"])

    def test_vl_weights_use_vision_class_for_text_generation(self):
        self.fake_runtime(model_type="qwen3_vl")
        result = self.client().generate_json([], "qwen3-vl")
        self.assertIsNone(result["error"])
        self.transformers.AutoModelForImageTextToText.from_pretrained.assert_called_once()
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_lora_wraps_base_weights_and_dtype_is_resolved(self):
        self.fake_runtime()
        adapted = Mock(device="cpu")
        adapted.generate.return_value = SimpleNamespace(sequences=[[11, 12, 13, 90, 91]])
        peft = ModuleType("peft")
        peft.PeftModel = SimpleNamespace(from_pretrained=Mock(return_value=adapted))
        with patch.dict(sys.modules, {"peft": peft}):
            client = self.client(lora_adapter=str(self.adapter_path), torch_dtype="float32")
            result = client.generate_json([], "adapter")
        self.assertIsNone(result["error"])
        peft.PeftModel.from_pretrained.assert_called_once_with(
            self.model, str(self.adapter_path), local_files_only=True)
        self.assertIs(self.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"],
                      self.torch.float32)
        adapted.eval.assert_called_once()
        adapted.generate.assert_called_once()
        self.model.generate.assert_not_called()

    def test_same_weights_are_loaded_once_across_clients(self):
        self.fake_runtime()
        main, critic = self.client(), self.client()
        self.assertIs(main.local_backend, critic.local_backend)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            for client in (main, critic):
                self.assertIsNone(client.generate_json([], "same-weights")["error"])
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        self.assertEqual(self.model.generate.call_count, 2)
        self.torch.cuda.is_available.assert_called_once_with()
        self.torch.backends.mps.is_available.assert_called_once_with()
        mode_lines = [line for line in stdout.getvalue().splitlines()
                      if "Current inference mode" in line or "当前运行模式" in line]
        self.assertEqual(len(mode_lines), 1)
        adapted = self.client(lora_adapter=str(self.adapter_path))
        self.assertIsNot(main.local_backend, adapted.local_backend)

    def test_parallel_clients_serialize_shared_model_generation(self):
        self.fake_runtime()
        main, critic = self.client(), self.client()
        first_entered, second_started, second_entered, release = Event(), Event(), Event(), Event()
        count_lock = Lock()
        calls = 0

        def blocking_generate(**kwargs):
            nonlocal calls
            with count_lock:
                calls += 1
                call = calls
            if call == 1:
                first_entered.set()
                if not release.wait(3):
                    raise RuntimeError("test failed to release generation")
            else:
                second_entered.set()
            return [[11, 12, 13, 90, 91]]

        def second_request():
            second_started.set()
            return critic.generate_json([], "critic")

        self.model.generate.side_effect = blocking_generate
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(main.generate_json, [], "main")
            try:
                self.assertTrue(first_entered.wait(2))
                second = pool.submit(second_request)
                self.assertTrue(second_started.wait(2))
                self.assertFalse(second_entered.wait(0.1))
            finally:
                release.set()
            self.assertIsNone(first.result(timeout=3)["error"])
            self.assertIsNone(second.result(timeout=3)["error"])
        self.assertTrue(second_entered.is_set())
        self.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_model_errors_follow_agent_error_contract_without_retries(self):
        self.fake_runtime()
        self.model.generate.side_effect = RuntimeError("not enough memory")
        with patch("src.utils.api_client.time.sleep") as sleep:
            result = self.client().generate_json([], "local", max_retries=5)
        self.assertIsNone(result["response"])
        self.assertIsNone(result["content"])
        self.assertIn("not enough memory", result["error"])
        self.model.generate.assert_called_once()
        sleep.assert_not_called()

    def test_missing_runtime_returns_actionable_dependency_error(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            result = self.client().generate_json([], "local")
        self.assertIn("requirements-local.txt", result["error"])
        self.assertIsNone(result["response"])
        self.openai.assert_not_called()

    def test_phase1_accepts_local_structured_response(self):
        self.fake_runtime(text=json.dumps({
            "top_candidates": ["45", "23"], "final_diagnosis_id": "45",
            "differential_reasoning": "Test response.",
        }))
        result = Phase1TrackAAgent(self.client(), "local-model").process("Test narrative.")
        self.assertIsNone(result["error"])
        self.assertEqual(result["final_diagnosis_id"], "45")
        self.assertEqual(result["top_candidates"], ["45", "23"])

    def test_phase1_track_a_requests_cache_without_changing_messages(self):
        llm_client = Mock()
        llm_client.generate_json.return_value = {
            "error": None,
            "content": '{"top_candidates":["45"],"final_diagnosis_id":"45"}',
        }

        Phase1Manager(llm_client, "model").track_a.process("Example test narrative.")
        Phase1Manager(llm_client, "model", kv_prefix_cache=True).track_a.process("Example test narrative.")

        default, enabled = [call.kwargs for call in llm_client.generate_json.call_args_list]
        self.assertEqual(enabled.pop("prefix_message_count"), 3)
        self.assertEqual(default, enabled)
        self.assertEqual([message["role"] for message in default["messages"]],
                         ["system", "user", "assistant", "user"])

    def test_loopback_api_accepts_no_key_and_forwards_extra_body(self):
        self.openai.return_value.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
        for url in ("http://localhost:8000/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"):
            with self.subTest(url=url):
                client = LLMClient.from_config({
                    "localhost": False, "base_url": url, "api_key_env": "UNSET_TEST_KEY",
                    "supports_logprobs": False, "extra_body": {"enable_thinking": False},
                })
                self.assertFalse(client.is_local)
                self.assertTrue(self.openai.call_args.kwargs["api_key"])
                result = client.generate_json([], "local-http")
                self.assertIsNone(result["error"])
                request = self.openai.return_value.chat.completions.create.call_args.kwargs
                self.assertEqual(request["extra_body"], {"enable_thinking": False})
                self.assertNotIn("logprobs", request)
                self.assertNotIn("top_logprobs", request)

    def test_remote_endpoint_still_requires_credentials(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            LLMClient(base_url="https://example.invalid/v1")
        self.openai.assert_not_called()

    def test_switch_endpoint_preserves_explicit_key_without_environment_key(self):
        client = LLMClient(base_url="https://first.invalid/v1", api_key="explicit-test-key")
        client.switch_base_url("https://second.invalid/v1")
        self.assertEqual(self.openai.call_args.kwargs["base_url"], "https://second.invalid/v1")
        self.assertEqual(self.openai.call_args.kwargs["api_key"], "explicit-test-key")
        self.assertFalse(client.is_local)

    def test_remote_config_controls_retries_without_sdk_double_retry(self):
        completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
        self.openai.return_value.chat.completions.create.side_effect = [RuntimeError("temporary"), completion]
        client = LLMClient.from_config({
            "base_url": "http://localhost:8000/v1", "max_retries": 2, "retry_delay": 0.25,
        })
        with patch("src.utils.api_client.time.sleep") as sleep:
            result = client.generate_json([], "test-model", logprobs=False)
        self.assertIsNone(result["error"])
        self.assertEqual(self.openai.return_value.chat.completions.create.call_count, 2)
        self.assertEqual(self.openai.call_args.kwargs["max_retries"], 0)
        sleep.assert_called_once_with(0.25)


if __name__ == "__main__":
    unittest.main()
