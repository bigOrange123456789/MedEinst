"""Unified local Transformers and OpenAI-compatible structured-response client."""
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

from src.utils.runtime_config import (
    config_bool, config_relative_path, configured_base_url, is_loopback_url,
)

load_dotenv()


class LLMClient:
    """Keep one response contract for all agents, regardless of inference backend."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: int = 120, api_key_env: str = "OPENAI_API_KEY", *,
                 localhost=False, backend=None, model_path=None, lora_adapter=None,
                 torch_dtype="auto", trust_remote_code=True,
                 generation_config=None, chat_template_kwargs=None, extra_body=None,
                 supports_logprobs=True, max_retries=3, retry_delay=2.0):
        backend = str(backend or "").strip().lower()
        if backend not in ("", "openai", "local", "transformers"):
            raise ValueError(f"Unknown LLM backend: {backend}")
        self.is_local = config_bool(localhost) or backend in ("local", "transformers")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.timeout = timeout
        self.api_key_env = api_key_env
        self.generation_config = dict(generation_config or {})
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self.supports_logprobs = config_bool(supports_logprobs, True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        self.api_key = None
        self.local_backend = None
        if self.is_local:
            from src.utils.local_llm import LocalTransformersBackend
            self.local_backend = LocalTransformersBackend.shared(
                model_path=str(Path(model_path).expanduser().resolve()) if model_path else None,
                lora_adapter=str(Path(lora_adapter).expanduser().resolve()) if lora_adapter else None,
                torch_dtype=torch_dtype,
                trust_remote_code=config_bool(trust_remote_code, True),
            )
        else:
            self.api_key = self._resolve_api_key(api_key)
            self._create_api_client()

    @classmethod
    def from_config(cls, section: dict, config_path=None, default_timeout=120):
        """Build a client from either api or critical_model in settings.yaml."""
        config_path = Path(config_path or Path(__file__).resolve().parents[2] / "config" / "settings.yaml").resolve()
        options = {name: section[name] for name in (
            "api_key", "api_key_env", "localhost", "backend", "torch_dtype",
            "trust_remote_code", "chat_template_kwargs", "extra_body", "supports_logprobs",
            "max_retries", "retry_delay",
        ) if name in section}
        for name in ("model_path", "lora_adapter"):
            if section.get(name):
                options[name] = config_relative_path(section[name], config_path)
        generation = dict(section.get("generation_config") or {})
        for name in ("max_new_tokens", "temperature", "top_p", "repetition_penalty"):
            if name in section:
                generation[name] = section[name]
        return cls(base_url=configured_base_url(section),
                   timeout=section.get("timeout", default_timeout),
                   generation_config=generation, **options)

    def _resolve_api_key(self, api_key):
        if api_key is None:
            api_key = os.getenv(self.api_key_env) if self.api_key_env else None
        if not api_key and is_loopback_url(self.base_url):
            api_key = "local"
        if not api_key:
            raise ValueError(f"API key not provided and {self.api_key_env} environment variable not set")
        return api_key

    def _create_api_client(self):
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                             timeout=self.timeout, max_retries=0)

    @staticmethod
    def _answer_content(content):
        # Some thinking templates prefill <think>; only </think> is decoded.
        return re.sub(r"^.*?</think>\s*", "", content or "", flags=re.DOTALL).strip()

    def generate_json(
        self, 
        messages: list, # 上下文：四条上下文，system【提示】、user【问题示例】、assistant【回复示例】、user【问题】
        model: str, # 模型名称：DeepSeek-R1-Distill-Qwen-1.5B
        logprobs: bool = True, # 是否返回token级概率
        top_logprobs: int = 5, # 每个位置额外返回几个"备选 token"
        temperature: float = 0.0, 
        max_tokens: int = 4096,
        max_retries: Optional[int] = None,   # 最多尝试几次
        retry_delay: Optional[float] = None, # 固定等待多少秒
        *,
        prefix_message_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        temperature = self.generation_config.get("temperature", temperature)    # 0
        max_tokens = self.generation_config.get("max_new_tokens", max_tokens)   # 4096
        if self.is_local:
            cache_options = {} if prefix_message_count is None else {
                "prefix_message_count": prefix_message_count,
            }
            try:
                content = self.local_backend.generate(
                    messages, max_new_tokens=max_tokens, temperature=temperature,
                    top_p=self.generation_config.get("top_p", 0.95), # 0.95
                    repetition_penalty=self.generation_config.get("repetition_penalty", 1.0), # 1.0 # 重复惩罚:>1 抑制重复、鼓励用词多样；<1 反而鼓励重复（几乎没人用）
                    chat_template_kwargs=self.chat_template_kwargs, # {} # 每个模型的模板不一样，能用的 key 就不一样
                    **cache_options,
                )
                content = self._answer_content(content)
                completion = SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(role="assistant", content=content), logprobs=None,
                )])
                return {"response": completion, "content": content, "error": None}
            except Exception as exc:
                return {"response": None, "content": None,
                        "error": f"{type(exc).__name__}: {exc}"}

        request = {"model": model, "messages": messages}
        model_id = model.lower().rsplit("/", 1)[-1]
        if re.match(r"^(?:gpt-5|o1|o3|o4)(?:-|$)", model_id):
            request["max_completion_tokens"] = max_tokens
        else:
            request.update(temperature=temperature, max_tokens=max_tokens)
        if logprobs and self.supports_logprobs:
            request.update(logprobs=True, top_logprobs=top_logprobs)
        if self.extra_body:
            request["extra_body"] = self.extra_body
        attempts = self.max_retries if max_retries is None else max_retries
        delay = self.retry_delay if retry_delay is None else retry_delay
        for attempt in range(attempts):
            try:
                completion = self.client.chat.completions.create(**request)
                content = self._answer_content(completion.choices[0].message.content)
                return {"response": completion, "content": content, "error": None}
            except Exception as exc:
                if attempt == attempts - 1:
                    return {"response": None, "content": None,
                            "error": f"{type(exc).__name__}: {exc}"}
                time.sleep(delay)
        return {"response": None, "content": None, "error": "Max retries exceeded"}

    def switch_base_url(self, new_base_url: str, new_api_key: Optional[str] = None):
        self.base_url = new_base_url
        if new_api_key is None:
            new_api_key = (os.getenv(self.api_key_env) if self.api_key_env else None) or self.api_key
        self.api_key = self._resolve_api_key(new_api_key)
        self._create_api_client()
        self.is_local = False
        self.local_backend = None
