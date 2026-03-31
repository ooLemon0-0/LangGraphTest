"""FastAPI application for the external gateway."""

from __future__ import annotations

from fastapi import FastAPI

from app.common.logging import configure_logging
from app.common.settings import get_settings
from app.gateway.api import create_router


settings = get_settings()
configure_logging(settings.logging.level, settings.logging.json)

app = FastAPI(title="Local LangGraph Gateway", version="0.1.0")
app.include_router(create_router(settings))
