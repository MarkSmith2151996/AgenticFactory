"""Structured output schema for keepa-downloader."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KeepaDownloadResult(BaseModel):
    total_urls: int
    successful: int
    failed: int
    total_rows: int
    total_size_mb: float
    files: list[dict[str, Any]]
    errors: list[str | dict[str, Any]] = Field(default_factory=list)
