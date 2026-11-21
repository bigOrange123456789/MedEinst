"""OpenAI-compatible embedding client for matching and retrieval."""

from src.utils.console import tr

import os
import time
import numpy as np
from typing import List, Optional, Union
from openai import OpenAI
from src.utils.runtime_config import config_bool, configured_base_url, is_loopback_url


class EmbeddingClient:
    """Create vectors for exemplar retrieval and semantic matching."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enabled: bool = True,
        timeout: float = 120
    ):

        self.enabled = config_bool(enabled, default=True)
        self.model = model
        self.dimension = int(dimension)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.client = None
        if not self.enabled:
            return

        if api_key is None:
            api_key = os.getenv(api_key_env) if api_key_env else None

        # Local OpenAI-compatible servers often run without authentication.
        if not api_key and is_loopback_url(base_url):
            api_key = "local"

        if not api_key:
            raise ValueError(
                f"API key not provided. Set {api_key_env} environment variable."
            )

        self.client = OpenAI(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

    @classmethod
    def from_config(cls, section: dict, config_path=None):
        """Build a remote/local HTTP embedding client, or explicitly disable it."""
        section = section or {}
        return cls(
            base_url=configured_base_url(section),
            api_key=section.get("api_key"),
            api_key_env=section.get("api_key_env", "OPENAI_API_KEY"),
            model=section.get("model_name", section.get("model", "text-embedding-3-small")),
            dimension=section.get("dimension", 1536),
            max_retries=section.get("max_retries", 3),
            retry_delay=section.get("retry_delay", 1.0),
            enabled=config_bool(section.get("enabled"), default=True),
            timeout=section.get("timeout", 120),
        )

    def embed_text(self, text: str) -> np.ndarray:

        if not self.enabled:
            raise RuntimeError("Embeddings are disabled (embedding.enabled: false).")

        if not text or not text.strip():

            return np.zeros(self.dimension, dtype=np.float32)

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text.strip()
                )

                embedding = response.data[0].embedding
                return np.array(embedding, dtype=np.float32)

            except Exception as e:
                print(tr(
                    f"[EmbeddingClient] Attempt {attempt + 1} failed: {e}",
                    f'[嵌入客户端] 第 {attempt + 1} 次尝试失败：{e}',
                ))
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        return np.zeros(self.dimension, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = []

        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return np.stack(embeddings)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> np.ndarray:

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.append(batch_embeddings)

            if i + batch_size < len(texts):
                time.sleep(0.5)

        return np.vstack(all_embeddings)

if __name__ == "__main__":

    try:
        client = EmbeddingClient()

        test_text = "Patient presents with fever and productive cough"
        embedding = client.embed_text(test_text)

        print(f"Text: {test_text}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"First 5 values: {embedding[:5]}")

    except Exception as e:
        print(f"Test failed: {e}")

