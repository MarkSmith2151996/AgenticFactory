"""Reusable tools for AgenticFactory agents."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import urllib.error
import urllib.request
from typing import Any

from agentic_factory.db import db_connection, utc_now
from agentic_factory.tools.cc_wrappers import keepa_analyze_handler, scoring_engine_handler
from agentic_factory.tools.registry import ToolDef, register_tool


async def browser_gather(task: str, max_steps: int = 15) -> str:
    """Lazy wrapper around browser-use so registry imports do not require browser dependencies."""
    from agentic_factory.tools.browser_search import browser_gather as run_browser_gather

    return await run_browser_gather(task, max_steps=max_steps)


async def _browser_gather_tool(query: str) -> str:
    return await browser_gather(query)


async def keepa_download(urls: list[str], output_dir: str = "~/keepa-downloads") -> dict[str, Any]:
    """Download Keepa CSV exports through the configured Mac service."""
    payload = {"urls": urls, "output_dir": output_dir}
    return await asyncio.to_thread(_post_keepa_download, payload)


def _post_keepa_download(payload: dict[str, Any]) -> dict[str, Any]:
    base_url = os.environ.get("KEEPA_DOWNLOADER_URL", "http://anthonys-macbook-pro:8095").rstrip("/")
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/download",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"keepa_download returned HTTP {exc.code}: {error_body}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"keepa_download request failed: {exc}") from exc

    decoded = json.loads(response_body)
    if not isinstance(decoded, dict):
        raise RuntimeError("keepa_download returned a non-object JSON response")
    return decoded


async def db_write(table: str, data: dict[str, Any]) -> dict[str, Any]:
    """Write a logical tool record to the AOODB."""
    with db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              table_name TEXT NOT NULL,
              data_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        cursor = conn.execute(
            "INSERT INTO tool_records (table_name, data_json, created_at) VALUES (?, ?, ?)",
            (table, json.dumps(data, sort_keys=True), utc_now()),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "table": table, "status": "written"}


async def db_read(db_path: str, query: str, params: list[Any] | None = None) -> dict[str, Any]:
    """Run a read-only SQLite query and return rows as dictionaries."""
    return await asyncio.to_thread(_db_read_sync, db_path, query, params or [])


def _db_read_sync(db_path: str, query: str, params: list[Any]) -> dict[str, Any]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
    return {"rows": rows, "row_count": len(rows)}


async def error_log(error: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Log an agent execution error to the AOODB."""
    with db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS error_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              error TEXT NOT NULL,
              context_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        cursor = conn.execute(
            "INSERT INTO error_log (error, context_json, created_at) VALUES (?, ?, ?)",
            (error, json.dumps(context or {}, sort_keys=True), utc_now()),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "status": "logged"}


def _register_builtin_tools() -> None:
    builtins = [
        ToolDef(
            name="browser_gather",
            description="Navigate to a URL and gather information using a headless browser",
            project="fba",
            handler=_browser_gather_tool,
            input_schema={
                "query": {
                    "type": "string",
                    "description": "Browser navigation instruction",
                }
            },
        ),
        ToolDef(
            name="keepa_download",
            description="Download Keepa CSV exports via the Mac service",
            project="fba",
            handler=keepa_download,
            input_schema={
                "urls": {"type": "array", "items": {"type": "string"}},
                "output_dir": {"type": "string"},
            },
        ),
        ToolDef(
            name="db_write",
            description="Write a record to the project's AOODB slice",
            project="fba",
            handler=db_write,
            input_schema={"table": {"type": "string"}, "data": {"type": "object"}},
        ),
        ToolDef(
            name="db_read",
            description="Run a read-only SQLite query through the shared ToolBox",
            project="toolbox",
            handler=db_read,
            input_schema={
                "db_path": {"type": "string"},
                "query": {"type": "string"},
                "params": {"type": "array", "items": {"type": "string"}},
            },
        ),
        ToolDef(
            name="keepa_analyze",
            description="Analyze Keepa CSV export data for FBA product metrics",
            project="fba",
            handler=keepa_analyze_handler,
            input_schema={
                "csv_path": {"type": "string"},
                "csv_paths": {"type": "array", "items": {"type": "string"}},
            },
        ),
        ToolDef(
            name="scoring_engine",
            description="Score analyzed product data for FBA wholesale viability",
            project="fba",
            handler=scoring_engine_handler,
            input_schema={"analysis": {"type": "object"}},
        ),
        ToolDef(
            name="error_log",
            description="Log an error encountered during agent execution",
            project=None,
            handler=error_log,
            input_schema={"error": {"type": "string"}, "context": {"type": "object"}},
        ),
    ]
    for tool in builtins:
        register_tool(tool)


_register_builtin_tools()


__all__ = [
    "browser_gather",
    "keepa_download",
    "keepa_analyze_handler",
    "scoring_engine_handler",
    "db_read",
    "db_write",
    "error_log",
]
