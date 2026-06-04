"""keepa-downloader agent handler."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import urllib.error
import urllib.request
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentic_factory.schemas.keepa_downloader import KeepaDownloadResult


DEFAULT_SERVICE_URL = "http://anthonys-macbook-pro:8095"
SERVICE_URL_ENV = "KEEPA_DOWNLOADER_URL"
DOWNLOAD_TIMEOUT_S = 900


class KeepaDownloadInput(BaseModel):
    urls: list[str] = Field(min_length=1)
    output_dir: str = "~/keepa-downloads"

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, urls: list[str]) -> list[str]:
        cleaned = [url.strip() for url in urls if url.strip()]
        if not cleaned:
            raise ValueError("keepa-downloader requires at least one URL")
        invalid = [url for url in cleaned if not url.startswith(("http://", "https://"))]
        if invalid:
            raise ValueError(f"keepa-downloader URLs must be HTTP(S): {invalid[0]}")
        return cleaned


def _service_url() -> str:
    return os.environ.get(SERVICE_URL_ENV, DEFAULT_SERVICE_URL).rstrip("/")


def _post_download(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{_service_url()}/download",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mac keepa-downloader returned HTTP {exc.code}: {error_body}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"Mac keepa-downloader request failed: {exc}") from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Mac keepa-downloader returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("Mac keepa-downloader returned a non-object JSON response")
    return decoded


def _summarize_response(total_urls: int, response: dict[str, Any]) -> KeepaDownloadResult:
    results = response.get("results", [])
    errors = response.get("errors", [])
    if not isinstance(results, list):
        results = []
    if not isinstance(errors, list):
        errors = []

    total_rows = 0
    total_size_mb = 0.0
    for result in results:
        if not isinstance(result, dict):
            continue
        total_rows += int(result.get("rows") or 0)
        total_size_mb += float(result.get("size_mb") or 0.0)

    return KeepaDownloadResult(
        total_urls=total_urls,
        successful=len(results),
        failed=len(errors),
        total_rows=total_rows,
        total_size_mb=round(total_size_mb, 1),
        files=[result for result in results if isinstance(result, dict)],
        errors=[error for error in errors if isinstance(error, dict)],
    )


async def invoke_keepa_downloader(input_payload: dict[str, Any]) -> Any:
    """Runtime entrypoint for keepa-downloader."""
    request = KeepaDownloadInput.model_validate(input_payload)
    response = await asyncio.to_thread(
        _post_download,
        {"urls": request.urls, "output_dir": request.output_dir},
    )
    result = _summarize_response(len(request.urls), response)

    from agentic_factory.agents import AgentInvocationResult

    return AgentInvocationResult(output=result.model_dump())
