"""Client for a local OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import ast
import json
import logging
import re
from time import perf_counter
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.common.schemas import ChatMessage
from app.common.settings import LLMSettings
from app.graph.planner_models import PlannerTrace


logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """Minimal async client for a local OpenAI-compatible chat API."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    async def chat(
        self,
        messages: list[ChatMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        purpose: str = "chat",
    ) -> str:
        """Send one chat completion request and return the assistant text."""
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_tokens,
            "metadata": {"purpose": purpose},
        }
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

        started = perf_counter()
        async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        latency_seconds = perf_counter() - started
        logger.info(
            "LLM call purpose=%s max_tokens=%s temperature=%s latency=%.3fs",
            purpose,
            payload["max_tokens"],
            payload["temperature"],
            latency_seconds,
        )
        return body["choices"][0]["message"]["content"]

    async def chat_json(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Request JSON output and parse it."""
        content = await self.chat(
            messages=messages,
            response_format={"type": "json_object"},
            purpose="chat_json",
        )
        return json.loads(content)

    async def chat_structured(
        self,
        messages: list[ChatMessage],
        *,
        schema: type[BaseModel],
        purpose: str,
        max_tokens: int,
        temperature: float,
    ) -> tuple[BaseModel, PlannerTrace]:
        """Request structured output with extraction, validation, and one repair pass.

        This layer is intentionally explicit and local. It is a temporary
        stabilization step until we have enough planner traces to support
        provider-native structured outputs or SFT on the planner format.
        """
        raw_output = await self.chat(
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
            purpose=purpose,
        )
        started = perf_counter()
        parsed = self._parse_structured_output(raw_output, schema)
        parse_latency = perf_counter() - started
        if parsed is not None:
            return parsed, PlannerTrace(
                purpose=purpose,
                max_tokens=max_tokens,
                temperature=temperature,
                latency_seconds=parse_latency,
                parsed_ok=True,
                raw_output=raw_output,
            )

        repair_prompt = (
            "请把下面内容修复为一个合法 JSON 对象，且必须满足既定 schema。"
            "只能输出 JSON，不要输出解释。\n\n"
            + raw_output
        )
        repair_output = await self.chat(
            [ChatMessage(role="system", content=repair_prompt)],
            response_format={"type": "json_object"},
            max_tokens=min(96, max_tokens),
            temperature=0.0,
            purpose=f"{purpose}_repair",
        )
        repaired = self._parse_structured_output(repair_output, schema)
        if repaired is None:
            raise ValueError(f"Unable to parse structured output for {purpose}. raw={raw_output!r} repair={repair_output!r}")
        return repaired, PlannerTrace(
            purpose=purpose,
            max_tokens=max_tokens,
            temperature=temperature,
            latency_seconds=parse_latency,
            parsed_ok=True,
            repaired=True,
            raw_output=raw_output,
            repair_raw_output=repair_output,
        )

    def _parse_structured_output(self, raw_output: str, schema: type[BaseModel]) -> BaseModel | None:
        """Parse model output into the requested schema."""
        for candidate in candidate_json_strings(raw_output):
            try:
                return schema.model_validate_json(candidate)
            except (ValidationError, json.JSONDecodeError):
                pass

            python_dict = try_parse_python_dict(candidate)
            if python_dict is not None:
                try:
                    return schema.model_validate(python_dict)
                except ValidationError:
                    continue
        return None


def candidate_json_strings(text: str) -> list[str]:
    """Generate progressively cleaned JSON candidates from raw model output."""
    cleaned = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    extracted = extract_json_object(fenced)
    variants = [cleaned, fenced, extracted]
    normalized_variants: list[str] = []
    for item in variants:
        if not item:
            continue
        normalized_variants.append(item)
        normalized_variants.append(
            item.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in normalized_variants:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def extract_json_object(text: str) -> str:
    """Extract the first JSON object-looking substring."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def try_parse_python_dict(text: str) -> dict[str, Any] | None:
    """Parse Python-like dict syntax as a last local fallback."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, dict) else None
