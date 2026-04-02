"""Deterministic candidate tool retrieval utilities.

This module is intentionally simple today so we can later swap the scorer
for embedding retrieval or a reranker without touching the graph nodes.
"""

from __future__ import annotations

import re
from typing import Any


def normalize_tool_metadata(tool: dict[str, Any]) -> dict[str, Any]:
    """Standardize raw MCP tool metadata into a retrieval-friendly shape."""
    name = str(tool.get("name", "")).strip()
    description = str(tool.get("description", "")).strip()
    description_zh = str(tool.get("description_zh", "")).strip()
    display_name_zh = str(tool.get("display_name_zh", "")).strip()
    aliases_zh = [str(item).strip() for item in tool.get("aliases_zh", [])]
    input_fields = [str(item).strip() for item in tool.get("input_fields", [])]
    tags = [str(item).strip().lower() for item in tool.get("tags", [])]
    business_domain = str(tool.get("business_domain", "")).strip().lower()
    permission = str(tool.get("permission", "")).strip().lower()
    mode = str(tool.get("mode", "read")).strip().lower()
    risk_level = str(tool.get("risk_level", "low")).strip().lower()

    inferred_tags = sorted(
        {
            token
            for token in re.split(
                r"[^a-zA-Z0-9_\u4e00-\u9fff]+",
                f"{name} {description} {description_zh} {display_name_zh} {' '.join(aliases_zh)} {' '.join(input_fields)}",
            )
            if token and len(token) > 2
        }
    )
    return {
        "name": name,
        "description": description,
        "description_zh": description_zh,
        "display_name_zh": display_name_zh,
        "aliases_zh": aliases_zh,
        "input_fields": input_fields,
        "tags": sorted(set(tags + inferred_tags)),
        "business_domain": business_domain or infer_business_domain(name, description),
        "permission": permission or infer_permission(name, mode),
        "mode": mode if mode in {"read", "write"} else "read",
        "risk_level": risk_level if risk_level in {"low", "medium", "high"} else "low",
    }


def infer_business_domain(name: str, description: str) -> str:
    """Infer a coarse business domain for filtering and future retrieval."""
    text = f"{name} {description}".lower()
    if "house" in text:
        return "housing"
    if "agent" in text:
        return "agent"
    if "tool" in text:
        return "meta"
    return "general"


def infer_permission(name: str, mode: str) -> str:
    """Infer a coarse permission label for future role-based policies."""
    if name.startswith("update_") or mode == "write":
        return "write"
    return "read"


def retrieve_candidate_tools(
    user_input: str,
    all_tools: list[dict[str, Any]],
    *,
    top_k: int = 5,
    allowed_modes: set[str] | None = None,
    allowed_permissions: set[str] | None = None,
    allowed_domains: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Rank the current tool manifest down to a small candidate set.

    This is the seam that future embedding retrieval / reranking can replace.
    """
    normalized_query = user_input.lower()
    query_terms = {term for term in re.split(r"[^a-zA-Z0-9_]+", normalized_query) if len(term) > 1}
    scored: list[tuple[int, dict[str, Any]]] = []

    for raw_tool in all_tools:
        tool = normalize_tool_metadata(raw_tool)
        if allowed_modes and tool["mode"] not in allowed_modes:
            continue
        if allowed_permissions and tool["permission"] not in allowed_permissions:
            continue
        if allowed_domains and tool["business_domain"] not in allowed_domains:
            continue

        score = 0
        if tool["name"].lower() in normalized_query:
            score += 10
        if tool["mode"] == "write" and any(word in normalized_query for word in ["update", "rename", "change", "set"]):
            score += 5
        if tool["mode"] == "read" and any(word in normalized_query for word in ["get", "show", "list", "detail", "find"]):
            score += 4

        haystacks = [
            tool["name"].lower(),
            tool["description"].lower(),
            tool["description_zh"].lower(),
            tool["display_name_zh"].lower(),
            " ".join(tool["aliases_zh"]).lower(),
            " ".join(tool["input_fields"]).lower(),
            " ".join(tool["tags"]).lower(),
            tool["business_domain"],
            tool["permission"],
        ]
        for term in query_terms:
            for haystack in haystacks:
                if term in haystack:
                    score += 2
                    break

        if score == 0 and tool["name"] in {"list_tools", "get_tool_detail"}:
            score = 1

        if any(term in normalized_query for term in ["房源", "房子", "楼盘", "详情", "查询", "查看"]):
            if tool["business_domain"] == "housing" and tool["mode"] == "read":
                score += 6
            if tool["name"] == "get_house_detail":
                score += 10
        if any(term in normalized_query for term in ["价格", "改价", "房价", "修改价格", "更新价格"]):
            if tool["name"] == "update_house_price":
                score += 12
        if any(term in normalized_query for term in ["改名", "重命名", "名称"]):
            if tool["name"] == "update_house_name":
                score += 12
        if any(term in normalized_query for term in ["经纪人", "中介", "agent"]):
            if tool["business_domain"] == "agent":
                score += 8

        scored.append((score, tool))

    scored.sort(key=lambda item: (item[0], item[1]["risk_level"] == "low"), reverse=True)
    selected = [tool for score, tool in scored[:top_k] if score > 0]
    if not selected:
        selected = [tool for _, tool in scored[: min(top_k, len(scored))]]
    return selected
