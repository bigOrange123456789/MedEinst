"""Exercise worker-to-GPU routing without CUDA hardware or model downloads."""

import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Barrier, Event, Lock
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from scripts.run_ecr_agent import OnlineInferenceRunner, OnlineResult
from src.utils.api_client import LLMClient
from src.utils.local_llm import LocalInferenceWorkers


class TokenBatch(dict):
    def __init__(self):
        super().__init__(input_ids=SimpleNamespace(shape=(1, 3), device=None))

    def to(self, device):
        self["input_ids"].device = device
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        return "conversation"

    def __call__(self, prompt, **kwargs):
        return TokenBatch()

    def decode(self, tokens, **kwargs):
        return json.dumps({"device": tokens[0]})


class FakeModel:
    def __init__(self, device, runtime):
        self.device = device
        self.runtime = runtime

    def eval(self):
        pass

    def generate(self, input_ids, **kwargs):
        if input_ids.device != self.device:
            raise RuntimeError(f"Input on {input_ids.device}, model on {self.device}")
        if self.runtime.on_generate is not None:
            self.runtime.on_generate(self)
        return [[11, 12, 13, self.device]]


class FakeRuntime:
    def __init__(self, cuda_count=2, mps=False):
        self.loads = []
        self.on_generate = None
        self.torch = ModuleType("torch")
        self.torch.inference_mode = nullcontext
        self.torch.cuda = SimpleNamespace(
            is_available=Mock(return_value=cuda_count > 0),
            device_count=Mock(return_value=cuda_count),
            get_device_name=Mock(side_effect=lambda index: f"Test GPU {index}"),
            device=Mock(side_effect=lambda device: nullcontext()),
        )
        self.torch.backends = SimpleNamespace(mps=SimpleNamespace(
            is_available=Mock(return_value=mps),
        ))
        self.transformers = ModuleType("transformers")
        self.transformers.AutoConfig = SimpleNamespace(from_pretrained=Mock(
            return_value=SimpleNamespace(model_type="qwen3", architectures=[])))
        self.transformers.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(
            side_effect=lambda *args, **kwargs: FakeTokenizer()))
        self.transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(
            side_effect=self.load_model))

    def load_model(self, model_path, **kwargs):
        device = kwargs["device_map"]
        self.loads.append((model_path, device))
        return FakeModel(device, self)


class MultiGPUWorkerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.model_path = self.root / "model"
        self.model_path.mkdir()
        self.output = io.StringIO()
        capture = redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def runtime(self, cuda_count=2, mps=False):
        runtime = FakeRuntime(cuda_count, mps)
        modules = patch.dict(sys.modules, {
            "torch": runtime.torch, "transformers": runtime.transformers,
        })
        modules.start()
        self.addCleanup(modules.stop)
        return runtime

    def client(self, model_path=None):
        return LLMClient(localhost=True, model_path=str(model_path or self.model_path))

    def generate_device(self, client):
        result = client.generate_json([], "test-model")
        self.assertIsNone(result["error"])
        return json.loads(result["content"])["device"]

    def run_workers(self, count, request):
        # Keep all workers occupied so a fast task cannot hide missing workers.
        ready = Barrier(count)

        def work():
            ready.wait(timeout=5)
            return request()

        with ThreadPoolExecutor(
            max_workers=count, initializer=LocalInferenceWorkers().initialize,
        ) as pool:
            futures = [pool.submit(work) for _ in range(count)]
            return [future.result(timeout=10) for future in futures]

    def test_four_workers_share_two_replicas_and_keep_their_gpu(self):
        runtime = self.runtime(cuda_count=2)
        client = self.client()
        devices = self.run_workers(4, lambda: [self.generate_device(client) for _ in range(3)])

        self.assertEqual(Counter(row[0] for row in devices), {"cuda:0": 2, "cuda:1": 2})
        for row in devices:
            self.assertEqual(len(set(row)), 1)
        self.assertCountEqual([device for _, device in runtime.loads], ["cuda:0", "cuda:1"])
        self.assertIn("-> cuda:0", self.output.getvalue())
        self.assertIn("-> cuda:1", self.output.getvalue())

    def test_unused_gpus_do_not_load_replicas(self):
        runtime = self.runtime(cuda_count=4)
        client = self.client()
        devices = self.run_workers(2, lambda: self.generate_device(client))

        self.assertCountEqual(devices, ["cuda:0", "cuda:1"])
        self.assertCountEqual([device for _, device in runtime.loads], devices)

    def test_different_gpus_generate_concurrently(self):
        runtime = self.runtime(cuda_count=2)
        client = self.client()
        generating = Barrier(2)
        runtime.on_generate = lambda model: generating.wait(timeout=5)

        devices = self.run_workers(2, lambda: self.generate_device(client))

        self.assertCountEqual(devices, ["cuda:0", "cuda:1"])

    def test_main_and_critic_share_each_gpu_replica(self):
        runtime = self.runtime(cuda_count=2)
        main, critic = self.client(), self.client()
        devices = self.run_workers(2, lambda: (
            self.generate_device(main), self.generate_device(critic),
        ))

        self.assertCountEqual(devices, [("cuda:0", "cuda:0"), ("cuda:1", "cuda:1")])
        self.assertEqual(len(runtime.loads), 2)

    def test_different_models_follow_the_same_worker_device(self):
        runtime = self.runtime(cuda_count=2)
        critic_path = self.root / "critic"
        critic_path.mkdir()
        main, critic = self.client(), self.client(critic_path)
        devices = self.run_workers(2, lambda: (
            self.generate_device(main), self.generate_device(critic),
        ))

        self.assertCountEqual(devices, [("cuda:0", "cuda:0"), ("cuda:1", "cuda:1")])
        self.assertCountEqual(runtime.loads, [
            (str(path), device)
            for path in (self.model_path, critic_path)
            for device in ("cuda:0", "cuda:1")
        ])

    def test_single_gpu_reuses_one_model_and_serializes_generation(self):
        runtime = self.runtime(cuda_count=1)
        main, critic = self.client(), self.client()
        first_entered, second_started, second_entered, release = Event(), Event(), Event(), Event()
        counter_lock = Lock()
        calls = 0

        def blocking_generate(model):
            nonlocal calls
            with counter_lock:
                calls += 1
                call = calls
            if call == 1:
                first_entered.set()
                if not release.wait(5):
                    raise RuntimeError("test did not release first generation")
            else:
                second_entered.set()

        def second_request():
            second_started.set()
            return self.generate_device(critic)

        runtime.on_generate = blocking_generate
        with ThreadPoolExecutor(
            max_workers=2, initializer=LocalInferenceWorkers().initialize,
        ) as pool:
            first = pool.submit(self.generate_device, main)
            try:
                self.assertTrue(first_entered.wait(5))
                second = pool.submit(second_request)
                self.assertTrue(second_started.wait(5))
                self.assertFalse(second_entered.wait(0.1))
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5), "cuda:0")
            self.assertEqual(second.result(timeout=5), "cuda:0")
        self.assertTrue(second_entered.is_set())
        self.assertEqual(len(runtime.loads), 1)

    def test_cpu_workers_share_one_model(self):
        runtime = self.runtime(cuda_count=0)
        client = self.client()

        self.assertEqual(self.run_workers(3, lambda: self.generate_device(client)), ["cpu"] * 3)
        self.assertEqual(runtime.loads, [(str(self.model_path), "cpu")])
        runtime.torch.cuda.get_device_name.assert_not_called()

    def test_mps_workers_share_one_model(self):
        runtime = self.runtime(cuda_count=0, mps=True)
        client = self.client()

        self.assertEqual(self.run_workers(3, lambda: self.generate_device(client)), ["mps"] * 3)
        self.assertEqual(runtime.loads, [(str(self.model_path), "mps")])
        runtime.torch.cuda.get_device_name.assert_not_called()

    def test_new_executors_restart_at_gpu_zero_and_serial_calls_keep_gpu_zero(self):
        runtime = self.runtime(cuda_count=4)
        client = self.client()

        self.assertEqual(self.generate_device(client), "cuda:0")
        for _ in range(3):
            self.assertEqual(self.run_workers(1, lambda: self.generate_device(client)), ["cuda:0"])
        self.assertEqual(self.generate_device(client), "cuda:0")
        self.assertEqual(runtime.loads, [(str(self.model_path), "cuda:0")])

    def test_http_workers_do_not_require_local_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}), \
                patch("src.utils.api_client.OpenAI") as openai:
            openai.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"device":"server"}'))])
            client = LLMClient(base_url="http://localhost:8000/v1", api_key="test")

            self.assertEqual(self.run_workers(2, lambda: self.generate_device(client)), ["server"] * 2)
        self.assertEqual(self.output.getvalue(), "")

    def test_online_parallel_runner_assigns_real_worker_threads(self):
        runtime = self.runtime(cuda_count=2)
        client = self.client()
        ready = Barrier(2)
        runner = OnlineInferenceRunner.__new__(OnlineInferenceRunner)
        runner.max_workers = 2
        runner.verbose_timing = False
        runner.output_manager = Mock()

        def process_case(case, case_type):
            ready.wait(timeout=5)
            device = self.generate_device(client)
            return OnlineResult(case.name, case_type, "success", final_diagnosis_id=device)

        runner.process_single_case = process_case
        tasks = [(Path("case_a"), "control_case"), (Path("case_b"), "trap_case")]
        with redirect_stderr(io.StringIO()):
            results = runner._run_parallel(tasks)

        self.assertEqual([result.status for result in results], ["success", "success"])
        self.assertCountEqual([result.final_diagnosis_id for result in results], ["cuda:0", "cuda:1"])
        self.assertEqual(runner.output_manager.save_result.call_count, 2)
        self.assertEqual(len(runtime.loads), 2)


if __name__ == "__main__":
    unittest.main()
