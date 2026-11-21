"""Lazy, shared Transformers inference for the project's text-only agent calls."""

from pathlib import Path
from threading import RLock
from weakref import WeakValueDictionary
import json

from src.utils.console import tr


def is_vision_language_model(config) -> bool:
    description = " ".join([
        str(getattr(config, "model_type", "")),
        *list(getattr(config, "architectures", None) or []),
    ]).lower()
    return any(marker in description for marker in ("vl", "vision", "visual", "image"))


class LocalTransformersBackend:
    # Sharing both the weights and the lock avoids duplicate VRAM allocations and
    # concurrent generate calls when the main model and critic use the same model.
    _instances = WeakValueDictionary()
    _instances_lock = RLock()

    @classmethod
    def shared(cls, **options):
        key = json.dumps(options, sort_keys=True)
        with cls._instances_lock:
            backend = cls._instances.get(key)
            if backend is None:
                backend = cls(**options)
                cls._instances[key] = backend
            return backend

    def __init__(self, model_path, lora_adapter=None, torch_dtype="auto",
                 trust_remote_code=True):
        self.model_path = self._directory(model_path, "model_path")
        self.lora_adapter = self._directory(lora_adapter, "lora_adapter") if lora_adapter else None
        self.device = None
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self._lock = RLock()

    @staticmethod
    def _directory(value, name):
        if not value:
            raise ValueError(f"Local inference requires {name} in settings.yaml")
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local {name} directory does not exist: {path}")
        return str(path)

    @staticmethod
    def _select_device(torch):
        """Use an accelerator exposed by this PyTorch runtime, otherwise CPU."""
        if torch.cuda.is_available():
            return "cuda:0", f"GPU (CUDA, cuda:0, {torch.cuda.get_device_name(0)})"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "GPU (Apple MPS, mps)"
        return "cpu", "CPU (cpu)"

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Local inference requires torch, transformers and accelerate. "
                "Install requirements-local.txt in the environment used to run MedEinst."
            ) from exc

        self.device, mode = self._select_device(torch)
        print(tr(
            f"[LocalLLM] Auto-selected {mode}; loading model: {self.model_path}",
            f"[本地模型] 自动选择 {mode}，正在加载模型：{self.model_path}",
        ), flush=True)
        common = dict(trust_remote_code=self.trust_remote_code, local_files_only=True)
        config = AutoConfig.from_pretrained(self.model_path, **common)
        model_class = AutoModelForCausalLM
        if is_vision_language_model(config):
            try:
                from transformers import AutoModelForImageTextToText
            except ImportError as exc:
                raise ImportError(
                    "This vision-language model needs a newer transformers version "
                    "with AutoModelForImageTextToText (Qwen3-VL: >=4.57)."
                ) from exc
            model_class = AutoModelForImageTextToText

        dtype = self.torch_dtype
        if dtype != "auto":
            if dtype not in ("float16", "bfloat16", "float32", "float64"):
                raise ValueError(f"Unsupported torch_dtype: {dtype}")
            dtype = getattr(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, **common)
        model = model_class.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map=self.device, **common
        )
        if self.lora_adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("Install peft to use lora_adapter in settings.yaml") from exc
            model = PeftModel.from_pretrained(model, self.lora_adapter, local_files_only=True)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        print(tr(
            f"[LocalLLM] Current inference mode: {mode}",
            f"[本地模型] 当前运行模式：{mode}",
        ), flush=True)

    def generate(self, messages, max_new_tokens, temperature=0.0, top_p=0.95,
                 repetition_penalty=1.0, chat_template_kwargs=None):
        with self._lock:
            self._load()
            import torch

            # Existing agents only send text. VL loading stays text-only, as in
            # lab2.py, without requiring image/video processor dependencies.
            template_options = dict(chat_template_kwargs or {})
            template_options.update(tokenize=False, add_generation_prompt=True)
            prompt = self.tokenizer.apply_chat_template(messages, **template_options)
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = inputs.to(self.model.device)
            generation = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                repetition_penalty=repetition_penalty,
            )
            if temperature > 0:
                generation.update(temperature=temperature, top_p=top_p)
            # Preserve model generation_config EOS lists (e.g. Qwen turn-end).
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            if pad_id is not None:
                generation["pad_token_id"] = pad_id
            with torch.inference_mode():
                output = self.model.generate(**inputs, **generation)
            sequences = getattr(output, "sequences", output)
            generated = sequences[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
