"""Generic runtime executors for declarative agent specs."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentic_factory.compiler import compile_prompt
from agentic_factory.schemas import get_schema
from agentic_factory.spec import LlmAgentSpec, ServiceAgentSpec


MAX_TOOL_ITERATIONS = int(os.environ.get("AGENTIC_FACTORY_MAX_TOOL_ITERATIONS", "10"))
DEFAULT_TOOL_SERVER_PORT = 9100
DEFAULT_CHAT_COMPLETIONS_URL = "http://127.0.0.1:4096/v1/chat/completions"
PROJECTS_ROOT = Path(__file__).resolve().parents[3]
CUSTODIAN_DB_PATH = PROJECTS_ROOT / "nai-workbench" / "custodian" / "custodian.db"


@dataclass(frozen=True)
class AgentInvocationResult:
    """Agent output plus optional model usage metadata for AOODB persistence."""

    output: dict[str, Any]
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: float | None = None


async def execute_llm_agent(
    spec: LlmAgentSpec,
    input_payload: dict[str, Any],
) -> AgentInvocationResult:
    """Run an LLM agent through a structured JSON tool-call loop."""
    toolbox_tools = spec.toolbox or []
    tool_defs = await _fetch_all_tools(spec, spec.tools, toolbox_tools)
    compiled = compile_prompt(spec, input_payload, tool_defs=tool_defs)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": compiled["system"]},
        {"role": "user", "content": compiled["user"]},
    ]

    for _ in range(MAX_TOOL_ITERATIONS):
        response_text = await _call_llm(compiled["model"], messages, spec)
        parsed = _parse_response(response_text)

        if parsed["type"] == "tool_call":
            tool_result = await _execute_tool(spec.project, parsed["name"], parsed["params"], tool_defs)
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"[Tool Result: {parsed['name']}]\n{json.dumps(tool_result, sort_keys=True)}",
                }
            )
            continue

        if parsed["type"] == "final_answer":
            output_schema = get_schema(spec.output)
            normalized = parsed["data"]
            normalize = getattr(output_schema, "normalize", None)
            if callable(normalize):
                normalized = normalize(normalized)
            validated = output_schema.model_validate(normalized)
            return AgentInvocationResult(output=validated.model_dump())

        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your response was not valid JSON in the required format. "
                    "Respond with ONLY a tool_call JSON object or a final_answer JSON object. "
                    f"Parse error: {parsed['error']}"
                ),
            }
        )

    raise RuntimeError(f"Agent {spec.name} hit max iterations ({MAX_TOOL_ITERATIONS})")


async def _fetch_box_tools(project: str, allowlist: list[str]) -> list[dict[str, Any]]:
    """Fetch tool definitions from a project box and enforce the agent allowlist.

    Tool server ports are configured with BOX_TOOL_PORT_<PROJECT>, where PROJECT is
    uppercased and hyphens become underscores. The default is 9100.
    """
    response = await asyncio.to_thread(_request_box_tool_server, project, "GET", "/tools", None)
    tools = _extract_tool_list(response)
    allowlisted = [_normalize_tool_def(tool) for tool in tools if _tool_name(tool) in set(allowlist)]
    missing = [name for name in allowlist if name not in {tool["name"] for tool in allowlisted}]
    if missing:
        raise RuntimeError(f"Project {project} tool server did not expose allowlisted tools: {missing}")
    return allowlisted


async def _fetch_all_tools(
    spec: LlmAgentSpec,
    project_tools_allowlist: list[str],
    toolbox_tools_allowlist: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch tool definitions from the agent project box and the shared ToolBox."""
    tools: list[dict[str, Any]] = []

    if project_tools_allowlist:
        project_defs = await _fetch_box_tools(spec.project, project_tools_allowlist)
        for tool in project_defs:
            tool["source"] = spec.project
        tools.extend(project_defs)

    if toolbox_tools_allowlist:
        toolbox_defs = await _fetch_box_tools("toolbox", toolbox_tools_allowlist)
        for tool in toolbox_defs:
            tool["source"] = "toolbox"
        tools.extend(toolbox_defs)

    return tools


async def _call_llm(model: str, messages: list[dict[str, str]], spec: LlmAgentSpec) -> str:
    """Call the OpenCode chat-completions proxy without native tool parameters."""
    url = _chat_completions_url(spec)
    body = json.dumps({"model": _chat_model_name(model), "messages": messages}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_body = response.read().decode("utf-8")
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"LLM request failed for {spec.name}: {exc}") from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM proxy returned invalid JSON") from exc
    try:
        content = decoded["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM proxy response did not include choices[0].message.content") from exc
    if not isinstance(content, str):
        raise RuntimeError("LLM proxy returned non-string message content")
    return content


def _parse_response(text: str) -> dict[str, Any]:
    """Parse a structured JSON tool call or final answer from model text."""
    try:
        decoded = _extract_json_object(text)
    except ValueError as exc:
        return {"type": "invalid", "error": str(exc)}

    if not isinstance(decoded, dict):
        return {"type": "invalid", "error": "JSON response must be an object"}

    tool_call = decoded.get("tool_call")
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        params = tool_call.get("params", {})
        if isinstance(name, str) and isinstance(params, dict):
            return {"type": "tool_call", "name": name, "params": params}
        return {"type": "invalid", "error": "tool_call must include string name and object params"}

    if isinstance(decoded.get("name"), str) and isinstance(decoded.get("params", {}), dict):
        return {"type": "tool_call", "name": decoded["name"], "params": decoded.get("params", {})}

    final_answer = decoded.get("final_answer")
    if isinstance(final_answer, dict):
        return {"type": "final_answer", "data": final_answer}

    return {"type": "final_answer", "data": decoded}


async def _execute_tool(
    project: str,
    tool_name: str,
    params: dict[str, Any],
    tool_defs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute a project box tool and return any tool-server error to the model."""
    source = next((tool.get("source") for tool in tool_defs if tool.get("name") == tool_name), project)
    try:
        response = await asyncio.to_thread(
            _request_box_tool_server,
            str(source or project),
            "POST",
            f"/tools/{tool_name}",
            params,
        )
    except Exception as exc:
        return {"error": str(exc)}
    if not isinstance(response, dict):
        return {"result": response}
    return response


def _chat_model_name(model: str) -> str:
    return model.split(":", 1)[1] if ":" in model else model


def _chat_completions_url(spec: LlmAgentSpec) -> str:
    if spec.runtime is None:
        return DEFAULT_CHAT_COMPLETIONS_URL
    base_url = spec.runtime.base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError("no JSON object found")


def _request_box_tool_server(project: str, method: str, path: str, payload: dict[str, Any] | None) -> Any:
    port = _box_tool_port(project)
    url = f"http://127.0.0.1:{port}{path}"
    try:
        return _http_json_request(url, method, payload)
    except Exception as http_exc:
        try:
            return _docker_tool_request(project, port, method, path, payload)
        except Exception as docker_exc:
            raise RuntimeError(
                f"box tool server request failed via localhost ({http_exc}) and docker exec ({docker_exc})"
            ) from docker_exc


def _http_json_request(url: str, method: str, payload: dict[str, Any] | None) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def _docker_tool_request(
    project: str,
    port: int,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> Any:
    command = [
        "docker",
        "exec",
        f"alpha-{project}",
        "curl",
        "-sS",
        "-X",
        method,
        "-H",
        "Content-Type: application/json",
    ]
    if payload is not None:
        command.extend(["-d", json.dumps(payload)])
    command.append(f"http://127.0.0.1:{port}{path}")
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    return json.loads(completed.stdout)


def _box_tool_port(project: str) -> int:
    env_name = f"BOX_TOOL_PORT_{project.upper().replace('-', '_')}"
    env_value = os.environ.get(env_name)
    if env_value:
        return int(env_value)

    db_port = _lookup_box_tool_port(project)
    if db_port is not None:
        return db_port

    return DEFAULT_TOOL_SERVER_PORT


def _lookup_box_tool_port(project: str) -> int | None:
    if not CUSTODIAN_DB_PATH.exists():
        return None

    conn = sqlite3.connect(CUSTODIAN_DB_PATH)
    try:
        row = conn.execute(
            """
            SELECT pb.tool_server_port
            FROM project_boxes pb
            JOIN projects p ON p.id = pb.project_id
            WHERE p.name = ?
            """,
            (project,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()

    if row is None or row[0] is None:
        return None
    return int(row[0])


def _extract_tool_list(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        tools = response
    elif isinstance(response, dict) and isinstance(response.get("tools"), list):
        tools = response["tools"]
    else:
        raise RuntimeError("Tool server /tools response must be a list or object with tools list")
    return [tool for tool in tools if isinstance(tool, dict)]


def _normalize_tool_def(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _tool_name(tool),
        "description": str(tool.get("description") or tool.get("tool_description") or ""),
        "params": tool.get("params") or tool.get("input_schema") or {},
    }


def _tool_name(tool: dict[str, Any]) -> str | None:
    name = tool.get("name") or tool.get("tool_name")
    return name if isinstance(name, str) else None


async def execute_service_agent(
    spec: ServiceAgentSpec,
    input_payload: dict[str, Any],
) -> AgentInvocationResult:
    """Validate input, call a configured HTTP service, and return structured output."""
    missing = [field for field in spec.input.required if field not in input_payload]
    if missing:
        raise ValueError(f"Missing required input for {spec.name}: {missing[0]}")

    response = await asyncio.to_thread(_post_service, spec, input_payload)
    schema = get_schema(spec.output)
    output = _coerce_service_output(schema, input_payload, response)
    return AgentInvocationResult(output=output.model_dump())


def _post_service(spec: ServiceAgentSpec, input_payload: dict[str, Any]) -> dict[str, Any]:
    base_url = os.environ.get(spec.service.url_env or "", spec.service.url_default).rstrip("/")
    endpoint = spec.service.endpoint if spec.service.endpoint.startswith("/") else f"/{spec.service.endpoint}"
    body = json.dumps(input_payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=spec.service.timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{spec.name} service returned HTTP {exc.code}: {error_body}") from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise RuntimeError(f"{spec.name} service request failed: {exc}") from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{spec.name} service returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{spec.name} service returned a non-object JSON response")
    return decoded


def _coerce_service_output(
    schema: type[BaseModel],
    input_payload: dict[str, Any],
    response: dict[str, Any],
) -> BaseModel:
    try:
        return schema.model_validate(response)
    except Exception:
        pass

    if {"total_urls", "successful", "failed", "total_rows", "total_size_mb", "files", "errors"}.issubset(
        schema.model_fields
    ):
        results = response.get("results", [])
        errors = response.get("errors", [])
        if not isinstance(results, list):
            results = []
        if not isinstance(errors, list):
            errors = []
        files = [result for result in results if isinstance(result, dict)]
        error_items = [error for error in errors if isinstance(error, dict)]
        return schema.model_validate(
            {
                "total_urls": len(input_payload.get("urls", [])),
                "successful": len(files),
                "failed": len(error_items),
                "total_rows": sum(int(item.get("rows") or 0) for item in files),
                "total_size_mb": round(sum(float(item.get("size_mb") or 0.0) for item in files), 1),
                "files": files,
                "errors": error_items,
            }
        )

    return schema.model_validate(response)
