"""Lazy, shared Transformers inference for the project's text-only agent calls."""

from pathlib import Path
from threading import RLock, current_thread, local
from weakref import WeakValueDictionary
import json

from src.utils.console import tr


_worker_context = local()


class LocalInferenceWorkers:
    """Assign stable worker slots without importing optional local dependencies."""

    def __init__(self):
        self._next_slot = 0
        self._lock = RLock()

    def initialize(self):
        # A fresh allocator per executor makes repeated runs start at GPU 0.
        with self._lock:
            _worker_context.slot = self._next_slot
            self._next_slot += 1
        _worker_context.device_reported = False


def is_vision_language_model(config) -> bool:
    description = " ".join([
        str(getattr(config, "model_type", "")),
        *list(getattr(config, "architectures", None) or []),
    ]).lower()
    return any(marker in description for marker in ("vl", "vision", "visual", "image"))


class LocalTransformersBackend:
    # Equal model configurations share one replica and generation lock per GPU.
    _instances = WeakValueDictionary()
    _instances_lock = RLock()
    # Transformers/Accelerate loading can temporarily modify process-wide state.
    # Serialize loading only; already loaded replicas can generate independently.
    _load_lock = RLock()

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
                 trust_remote_code=True, *, cuda_index=None):
        self.model_path = self._directory(model_path, "model_path")
        self.lora_adapter = self._directory(lora_adapter, "lora_adapter") if lora_adapter else None
        self.device = None
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None
        self._lock = RLock()
        self._cuda_index = cuda_index
        self._cuda_count = None
        self._replicas = {}
        self._replicas_lock = RLock()

    def _worker_backend(self):
        slot = getattr(_worker_context, "slot", None)
        if slot is None or self._cuda_index is not None:
            return self
        with self._replicas_lock:
            if self._cuda_count is None:
                try:
                    import torch
                except ImportError:
                    # Keep the actionable dependency error raised by _load().
                    return self
                self._cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if not self._cuda_count:
                return self
            index = slot % self._cuda_count
            if not _worker_context.device_reported:
                worker = current_thread().name
                print(tr(
                    f"[LocalLLM] Worker {worker} -> cuda:{index} "
                    f"({self._cuda_count} visible CUDA GPUs)",
                    f"[本地模型] 工作线程 {worker} -> cuda:{index} "
                    f"（检测到 {self._cuda_count} 张可用 CUDA 显卡）",
                ), flush=True)
                _worker_context.device_reported = True
            if index == 0:
                return self
            if index not in self._replicas:
                self._replicas[index] = type(self)(
                    model_path=self.model_path, lora_adapter=self.lora_adapter,
                    torch_dtype=self.torch_dtype, trust_remote_code=self.trust_remote_code,
                    cuda_index=index,
                )
            return self._replicas[index]

    @staticmethod
    def _directory(value, name):
        if not value:
            raise ValueError(f"Local inference requires {name} in settings.yaml")
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Local {name} directory does not exist: {path}")
        return str(path)

    @staticmethod
    def _select_device(torch, cuda_index=0):
        """Use an accelerator exposed by this PyTorch runtime, otherwise CPU."""
        if torch.cuda.is_available():
            device = f"cuda:{cuda_index}"
            return device, f"GPU (CUDA, {device}, {torch.cuda.get_device_name(cuda_index)})"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "GPU (Apple MPS, mps)"
        return "cpu", "CPU (cpu)"

    def _load(self):
        if self.model is not None:
            return
        with self._load_lock:
            self._load_model()

    def _load_model(self):
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

        self.device, mode = self._select_device(torch, self._cuda_index or 0)
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
        backend = self._worker_backend() # <src.utils.local_llm.LocalTransformersBackend object>
        if backend is not self: # False
            return backend.generate(
                messages, max_new_tokens, temperature, top_p,
                repetition_penalty, chat_template_kwargs,
            )
        with self._lock:
            self._load() # 处理每个病例的时候重新读入模型
            import torch

            # Existing agents only send text. VL loading stays text-only, as in lab2.py, without requiring image/video processor dependencies.
            # 现有代理仅发送文本。VL加载仅保留文本，如lab2.py中所示，不需要图像/视频处理器依赖关系。
            template_options = dict(chat_template_kwargs or {}) # {}
            template_options.update(tokenize=False, add_generation_prompt=True) # {'tokenize': False, 'add_generation_prompt': True}
            # tokenize=F：别帮我分词，只给我文本；add_generation_prompt=T：轮到模型说了
            prompt = self.tokenizer.apply_chat_template(messages, **template_options) # 通过huggingface的apply_chat_template：将上下文messages数组转换为文本字符串
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False) # 编码：对话文本字符串 => 张量
            # return_tensors="pt"：让结果变成张量；add_special_tokens=F ： 不加开始结束等控制符( 开始BOS、结束EOS)
            inputs = inputs.to(self.model.device)
            generation = dict(
                max_new_tokens=max_new_tokens,        # 4096
                do_sample=temperature > 0,            # False   # temperature=0,无随机
                repetition_penalty=repetition_penalty,# 1       # 重复惩罚系数
            )
            if temperature > 0: # False
                generation.update(temperature=temperature, top_p=top_p)
            # Preserve model generation_config EOS lists (e.g. Qwen turn-end).
            # 保留模型生成_config EOS列表（例如Qwen转弯结束）。
            pad_id = self.tokenizer.pad_token_id # pad_id=151643 # PAD是“填充符”，151643是当前模型的空白填充符编码
            if pad_id is None: # False
                pad_id = self.tokenizer.eos_token_id
            if pad_id is not None: # True
                generation["pad_token_id"] = pad_id
            with torch.inference_mode(): # 推理模式，关闭训练所需的求导
                output = self.model.generate(**inputs, **generation) # 运行自回归解码
                # generation包含四项内容：最大长度、随机性、重复性、空白填空符。 {'max_new_tokens': 4096, 'do_sample': False, 'repetition_penalty': 1.0, 'pad_token_id': 151643}
            sequences = getattr(output, "sequences", output)
            generated = sequences[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
