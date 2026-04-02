"""Local embedding-based tool retriever.

The retriever lives in the gateway process so the three-process architecture
stays intact while retrieval becomes much stronger for Chinese tool-use queries.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from app.common.settings import RetrievalSettings, resolve_from_root
from app.graph.tool_retrieval import normalize_tool_metadata


logger = logging.getLogger(__name__)


class LocalToolEmbeddingRetriever:
    """Embeds tools locally and serves similarity-based retrieval."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self.settings = settings
        self._tokenizer = None
        self._model = None
        self._device = self._resolve_device(settings.device)
        self._tool_cache: dict[str, dict[str, Any]] = {}

    def preload(self, tools: list[dict[str, Any]]) -> None:
        """Warm the model and embed the initial tool set."""
        if not self.settings.enabled:
            logger.info("Embedding retriever disabled; skipping preload.")
            return
        self._ensure_model()
        self.ensure_tool_embeddings(tools)
        logger.info("Preloaded %s tool embeddings", len(self._tool_cache))

    def ensure_tool_embeddings(self, tools: list[dict[str, Any]]) -> None:
        """Embed any tools missing from the in-memory cache."""
        if not self.settings.enabled:
            return
        self._ensure_model()
        for tool in tools:
            normalized = normalize_tool_metadata(tool)
            cache_key = self._tool_cache_key(normalized)
            if cache_key in self._tool_cache:
                continue
            text = self._tool_text(normalized)
            embedding = self._embed_text(text)
            self._tool_cache[cache_key] = {
                "metadata": normalized,
                "embedding": embedding,
                "text": text,
            }

    def retrieve(self, query: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return tools above threshold, then trimmed to top-k."""
        if not self.settings.enabled:
            return []

        self.ensure_tool_embeddings(tools)
        query_embedding = self._embed_text(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        seen_names: set[str] = set()

        for tool in tools:
            normalized = normalize_tool_metadata(tool)
            cache_key = self._tool_cache_key(normalized)
            cached = self._tool_cache.get(cache_key)
            if cached is None or normalized["name"] in seen_names:
                continue
            similarity = cosine_similarity(query_embedding, cached["embedding"])
            enriched = dict(normalized)
            enriched["similarity"] = round(similarity, 4)
            scored.append((similarity, enriched))
            seen_names.add(normalized["name"])

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [
            tool
            for score, tool in scored
            if score >= self.settings.similarity_threshold
        ][: self.settings.top_k]
        if not selected:
            selected = [tool for _, tool in scored[: self.settings.top_k]]
        return selected

    def _ensure_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None and self._tokenizer is not None:
            return
        logger.info(
            "Loading local embedding retriever model %s from local cache on %s",
            self.settings.model_name,
            self._device,
        )
        local_path = resolve_from_root(self.settings.model_cache_dir) / self.settings.model_name.split("/")[-1]
        self._tokenizer = AutoTokenizer.from_pretrained(str(local_path), local_files_only=True)
        self._model = AutoModel.from_pretrained(str(local_path), local_files_only=True)
        self._model.to(self._device)
        self._model.eval()

    def _embed_text(self, text: str) -> torch.Tensor:
        """Encode one text into a normalized embedding vector."""
        assert self._tokenizer is not None
        assert self._model is not None
        encoded = self._tokenizer(
            text,
            max_length=self.settings.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state
            attention = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden * attention
            summed = masked_hidden.sum(dim=1)
            counts = attention.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized[0].cpu()

    def _tool_text(self, tool: dict[str, Any]) -> str:
        """Build the retrieval text used for tool embeddings."""
        return "\n".join(
            [
                f"name: {tool['name']}",
                f"display_name_zh: {tool.get('display_name_zh', '')}",
                f"description: {tool['description']}",
                f"description_zh: {tool.get('description_zh', '')}",
                f"aliases_zh: {' '.join(tool.get('aliases_zh', []))}",
                f"mode: {tool['mode']}",
                f"risk_level: {tool['risk_level']}",
                f"business_domain: {tool['business_domain']}",
                f"permission: {tool['permission']}",
                f"tags: {' '.join(tool['tags'])}",
                f"input_fields: {' '.join(tool['input_fields'])}",
            ]
        )

    def _tool_cache_key(self, tool: dict[str, Any]) -> str:
        """Build a stable cache key for one tool definition."""
        payload = self._tool_text(tool).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _resolve_device(configured_device: str) -> str:
        """Pick a concrete runtime device."""
        if configured_device == "cpu":
            return "cpu"
        if configured_device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return cosine similarity for two normalized vectors."""
    return float(torch.dot(left, right).item())
