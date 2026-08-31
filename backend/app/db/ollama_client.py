"""
Thin async client around Ollama's HTTP API. We use two endpoints:

- POST /api/embeddings  -> single text embedding (used for chunk + query embedding)
- POST /api/generate    -> text generation, with stream=True for token-by-token
  streaming to the frontend, and stream=False for the structured entity/relation
  extraction step during ingestion (we want one clean JSON blob back, not a stream).
"""
import json
from typing import AsyncGenerator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings


class OllamaClient:
    def __init__(self, base_url: str = settings.ollama_base_url):
        self.base_url = base_url.rstrip("/")

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(8))
    async def embed(self, text: str, model: Optional[str] = None) -> list[float]:
        model = model or settings.embedding_model
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embedding = data.get("embedding")
            if not embedding:
                raise ValueError(f"Ollama returned no embedding for model={model}")
            return embedding

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Non-streaming generation. Returns the full completion text."""
        model = model or settings.llm_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> AsyncGenerator[str, None]:
        """Streaming generation. Yields text tokens/fragments as they arrive."""
        model = model or settings.llm_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

    async def ping(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


ollama_client = OllamaClient()
