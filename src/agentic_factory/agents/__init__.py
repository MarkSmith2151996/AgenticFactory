"""Agent dispatch for the AgenticFactory runtime."""

from __future__ import annotations

from typing import Any

from agentic_factory.executor import (
    AgentInvocationResult,
    execute_llm_agent,
    execute_service_agent,
)
from agentic_factory.loader import get_agent
from agentic_factory.spec import LlmAgentSpec, ServiceAgentSpec


async def invoke_agent(name: str, input_payload: dict[str, Any]) -> AgentInvocationResult:
    """Dispatch an agent invocation by name."""
    spec = get_agent(name)
    if isinstance(spec, LlmAgentSpec):
        return await execute_llm_agent(spec, input_payload)
    if isinstance(spec, ServiceAgentSpec):
        return await execute_service_agent(spec, input_payload)
    raise ValueError(f"Unknown agent type: {spec.type}")
