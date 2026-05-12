"""Async Ollama /api/chat client with streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from docstack.config import get_settings


class ModelNotFoundError(RuntimeError):
    """Raised when Ollama returns 404 — model is not pulled yet."""


async def chat_stream(
    model: str,
    messages: list[dict[str, Any]],
    *,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
    temperature: float | None = None,
) -> AsyncIterator[str]:
    settings = get_settings()
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {},
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx
    if num_gpu is not None:
        payload["options"]["num_gpu"] = num_gpu
    if temperature is not None:
        payload["options"]["temperature"] = temperature

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code == 404:
                raise ModelNotFoundError(
                    f"Model '{model}' not found in Ollama. "
                    f"Run: ollama pull {model}"
                )
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("done"):
                    break
                msg = data.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    yield piece


async def chat_once(
    model: str,
    messages: list[dict[str, Any]],
    *,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
    temperature: float | None = None,
) -> str:
    parts: list[str] = []
    async for p in chat_stream(model, messages, num_ctx=num_ctx, num_gpu=num_gpu, temperature=temperature):
        parts.append(p)
    return "".join(parts)
