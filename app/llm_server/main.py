"""Cross-platform local OpenAI-compatible LLM server."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.common.logging import configure_logging
from app.common.schemas import HealthResponse
from app.common.settings import get_settings, resolve_from_root


settings = get_settings()
configure_logging(settings.logging.level, settings.logging.json)
logger = logging.getLogger(__name__)
app = FastAPI(title="Local OpenAI-Compatible LLM", version="0.1.0")


class ChatCompletionMessage(BaseModel):
    """One OpenAI-style chat message."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Minimal subset of the OpenAI chat completion payload."""

    model: str
    messages: list[ChatCompletionMessage]
    temperature: float = 0.1
    max_tokens: int = 512
    response_format: dict[str, Any] | None = None


class ModelRuntime:
    """Lazily loads the configured Hugging Face model."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def load(self) -> None:
        """Load tokenizer and model once."""
        if self.model is not None and self.tokenizer is not None:
            return

        llm_service = settings.llm.service
        model_path = resolve_from_root(llm_service.model_cache_dir)
        self.device = self._resolve_device(llm_service.device)
        torch_dtype = self._resolve_dtype(llm_service.dtype, self.device)
        logger.info(
            "Loading local model from %s (device=%s, dtype=%s)",
            llm_service.model_source,
            self.device,
            torch_dtype,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_service.model_source,
            cache_dir=str(model_path),
            trust_remote_code=llm_service.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_service.model_source,
            cache_dir=str(model_path),
            torch_dtype=torch_dtype,
            trust_remote_code=llm_service.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, messages: list[ChatCompletionMessage], temperature: float, max_tokens: int) -> str:
        """Generate one assistant message from chat input."""
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = self._build_prompt(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        do_sample = temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output = self.model.generate(**encoded, **generation_kwargs)

        prompt_tokens = encoded["input_ids"].shape[-1]
        new_tokens = output[0][prompt_tokens:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _build_prompt(self, messages: list[ChatCompletionMessage]) -> str:
        """Build a model-ready prompt from chat messages."""
        assert self.tokenizer is not None
        prompt_messages = [{"role": item.role, "content": item.content} for item in messages]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines = [f"{item.role}: {item.content}" for item in messages]
        lines.append("assistant:")
        return "\n".join(lines)

    @staticmethod
    def _resolve_device(configured_device: str) -> str:
        """Map config to a concrete PyTorch device."""
        if configured_device not in {"auto", "cpu", "cuda"}:
            return "cpu"
        if configured_device == "cpu":
            return "cpu"
        if configured_device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_dtype(configured_dtype: str, device: str) -> torch.dtype:
        """Map config dtype to a PyTorch dtype."""
        if device == "cpu":
            return torch.float32
        if configured_dtype in {"float16", "fp16"}:
            return torch.float16
        if configured_dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if configured_dtype in {"float32", "fp32"}:
            return torch.float32
        return torch.float16


runtime = ModelRuntime()
started_at = time.time()


@app.on_event("startup")
async def startup_event() -> None:
    """Warm up the model during service startup."""
    runtime.load()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server health."""
    service = "llm_server"
    if runtime.model is None:
        service = "llm_server_loading"
    return HealthResponse(status="ok", service=service)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    """Serve a minimal OpenAI-compatible chat completion response."""
    content = runtime.generate(
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    created = int(started_at)
    return {
        "id": f"chatcmpl-local-{created}",
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.llm_server.main:app",
        host=settings.llm.service.host,
        port=settings.llm.service.port,
    )
