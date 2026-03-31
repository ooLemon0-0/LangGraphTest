"""Client for a local OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.common.schemas import ChatMessage
from app.common.settings import LLMSettings


class OpenAICompatibleClient:
    """Minimal async client for OpenAI-compatible chat completions."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat completion request and return text content."""
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        return body["choices"][0]["message"]["content"]

    async def chat_json(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Request JSON output and parse it with a safe fallback."""
        content = await self.chat(
            messages=messages,
            response_format={"type": "json_object"},
        )
        return json.loads(content)
