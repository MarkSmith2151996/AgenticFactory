"""Prompt compiler for declarative agent specs."""

from __future__ import annotations

import json
import string
from typing import Any

from pydantic import BaseModel

from agentic_factory.loader import AGENTS_DIR
from agentic_factory.schemas import get_schema
from agentic_factory.spec import LlmAgentSpec
from agentic_factory.tools.registry import ToolDef, get_tools_for_agent


SYSTEM_RULES_PATH = AGENTS_DIR / "system_rules.md"

def _load_system_rules() -> str:
    return SYSTEM_RULES_PATH.read_text(encoding="utf-8").strip()


def _format_task(task: str, input_payload: dict[str, Any]) -> str:
    formatter = string.Formatter()
    placeholders = {
        field_name
        for _, field_name, _, _ in formatter.parse(task)
        if field_name is not None and field_name != ""
    }
    missing = sorted(field for field in placeholders if field not in input_payload)
    if missing:
        raise ValueError(f"Missing required input for prompt placeholder: {missing[0]}")
    return task.format(**input_payload)


ToolDescription = ToolDef | dict[str, Any]


def _describe_tools(tools: list[ToolDescription]) -> str:
    lines = []
    for tool in tools:
        lines.append(
            f"- {_tool_name(tool)}: {_tool_description(tool)}. Input: {_format_input_schema(_tool_params(tool))}"
        )
    return "\n".join(lines)


def _tool_name(tool: ToolDescription) -> str:
    return tool.name if isinstance(tool, ToolDef) else str(tool.get("name", ""))


def _tool_description(tool: ToolDescription) -> str:
    return tool.description if isinstance(tool, ToolDef) else str(tool.get("description", ""))


def _tool_params(tool: ToolDescription) -> dict[str, Any]:
    return tool.input_schema if isinstance(tool, ToolDef) else dict(tool.get("params", {}))


def _format_input_schema(input_schema: dict[str, Any]) -> str:
    compact = {}
    for name, schema in input_schema.items():
        if not isinstance(schema, dict):
            compact[name] = "unknown"
            continue
        field_type = schema.get("type", "unknown")
        if field_type == "array" and isinstance(schema.get("items"), dict):
            field_type = f"array[{schema['items'].get('type', 'unknown')}]"
        compact[name] = field_type
    return json.dumps(compact, sort_keys=True)


def _describe_schema(schema: type[BaseModel]) -> str:
    lines = [schema.__name__]
    for name, field in schema.model_fields.items():
        annotation = getattr(field.annotation, "__name__", repr(field.annotation))
        description = field.description or "No description."
        lines.append(f"- {name}: {annotation} - {description}")
    return "\n".join(lines)


def compile_prompt(
    spec: LlmAgentSpec,
    input_payload: dict[str, Any],
    tool_defs: list[ToolDescription] | None = None,
) -> dict[str, Any]:
    """Compile a layered prompt for an LLM-backed agent spec."""
    output_schema = get_schema(spec.output)
    tools = tool_defs if tool_defs is not None else get_tools_for_agent(spec)
    formatted_task = _format_task(spec.task, input_payload).strip()
    task_section = f"[Agent Task]\n{formatted_task}"
    tool_section = f"[Available Tools]\n{_describe_tools(tools)}"
    tool_calling_section = """[Tool Calling Format]
When you need to use a tool, respond with ONLY this JSON (no other text):
{"tool_call": {"name": "<tool_name>", "params": {<params>}}}

You will receive the tool's result in the next message. Then continue or call another tool.

When you have the final answer, respond with ONLY this JSON (no other text):
{"final_answer": {<output matching the Output Schema>}}

Rules:
- One tool call per message
- Always wait for the result before calling the next tool
- If a tool returns an error, you may retry with different params or report the error in your final answer"""
    output_schema_section = f"[Output Schema]\n{_describe_schema(output_schema)}"

    system_sections = [
        task_section,
        tool_section,
        tool_calling_section,
        output_schema_section,
    ]
    if spec.mode == "development":
        system_sections.insert(0, _load_system_rules())

    if spec.mode == "development" and spec.guidance:
        system_sections.append(f"[Guidance]\n{spec.guidance.strip()}")

    user = (
        f"[Task]\n{formatted_task}\n\n"
        f"[Input Payload]\n{json.dumps(input_payload, indent=2, sort_keys=True)}"
    )

    return {
        "system": "\n\n".join(system_sections),
        "user": user,
        "model": spec.model,
        "output_schema": output_schema,
    }
