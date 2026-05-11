"""Async Ollama /api/chat client with streaming."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from docstack.config import get_settings

logger = logging.getLogger(__name__)


async def chat_stream(
    model: str,
    messages: list[dict[str, str]],
    *,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
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

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        async with client.stream("POST", url, json=payload) as resp:
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
    messages: list[dict[str, str]],
    *,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
) -> str:
    parts: list[str] = []
    async for p in chat_stream(model, messages, num_ctx=num_ctx, num_gpu=num_gpu):
        parts.append(p)
    return "".join(parts)
