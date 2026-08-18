"""Thin wrapper tying the prompt builder to the streaming Ollama call."""
from typing import AsyncGenerator

from app.db.ollama_client import ollama_client
from app.generation.prompt import SYSTEM_PROMPT


async def stream_answer(prompt: str) -> AsyncGenerator[str, None]:
    async for token in ollama_client.generate_stream(prompt=prompt, system=SYSTEM_PROMPT):
        yield token
