"""Tool handlers for the FBA MCP server."""

from __future__ import annotations

import json
import sys
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


RUNTIME_INVOKE_URL = "http://127.0.0.1:8080/invoke_agent"


def log_query(tool_name: str, params: dict[str, Any]) -> None:
    """Lightweight MCP query log for the experimental FBA server."""
    print(
        f"[fba-mcp] tool={tool_name} params={json.dumps(params, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


async def invoke_agent(args: dict[str, Any]) -> str:
    """Forward an agent invocation to the AgenticFactory runtime."""
    agent_name = str(args.get("name") or "").strip()
    input_payload = args.get("input")
    log_query("invoke_agent", {"name": agent_name, "input": input_payload})

    if not agent_name:
        return "Error: 'name' is required."
    if not isinstance(input_payload, dict):
        return "Error: 'input' must be an object."

    payload = json.dumps({"name": agent_name, "input": input_payload}).encode("utf-8")
    req = urllib_request.Request(
        RUNTIME_INVOKE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=120) as response:
            return response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"Agent runtime HTTP {exc.code}: {body}"
    except (TimeoutError, urllib_error.URLError) as exc:
        return f"Agent runtime unavailable: {exc}"
