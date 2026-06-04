"""Central tool registry for declarative agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentic_factory.spec import AgentSpec, LlmAgentSpec


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    project: str | None
    handler: Callable[..., Any]
    input_schema: dict[str, Any]


TOOL_REGISTRY: dict[str, ToolDef] = {}


def register_tool(tool: ToolDef) -> None:
    """Register a tool. Raises on duplicate names."""
    if tool.name in TOOL_REGISTRY:
        raise ValueError(f"Duplicate tool name: {tool.name}")
    TOOL_REGISTRY[tool.name] = tool


def get_tool(name: str) -> ToolDef:
    """Lookup by name. Raises ValueError if not found."""
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:
        available = sorted(TOOL_REGISTRY)
        raise ValueError(f"Unknown tool: {name}. Available: {available}") from exc


def get_tools_for_agent(spec: AgentSpec) -> list[ToolDef]:
    """Return only the project and toolbox tools allowlisted by the agent."""
    if not isinstance(spec, LlmAgentSpec):
        return []
    tools: list[ToolDef] = []
    for name in [*spec.tools, *spec.toolbox]:
        tools.append(get_tool(name))
    return tools


def list_tools(project: str | None = None) -> list[ToolDef]:
    """List all tools, optionally filtered by project tag."""
    tools = list(TOOL_REGISTRY.values())
    if project is None:
        return tools
    return [tool for tool in tools if tool.project == project]
