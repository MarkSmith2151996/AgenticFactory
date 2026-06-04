from __future__ import annotations

import pytest

from agentic_factory.spec import LlmAgentSpec
from agentic_factory.tools.registry import (
    ToolDef,
    get_tool,
    get_tools_for_agent,
    list_tools,
    register_tool,
)


async def sample_handler() -> dict[str, str]:
    return {"status": "ok"}


def test_register_tool_and_retrieve_by_name() -> None:
    tool = ToolDef(
        name="unit-test-tool",
        description="Unit test tool",
        project="test",
        handler=sample_handler,
        input_schema={},
    )

    register_tool(tool)

    assert get_tool("unit-test-tool") is tool


def test_duplicate_name_raises_error() -> None:
    tool = ToolDef(
        name="duplicate-test-tool",
        description="Duplicate test tool",
        project="test",
        handler=sample_handler,
        input_schema={},
    )

    register_tool(tool)

    with pytest.raises(ValueError, match="Duplicate tool name: duplicate-test-tool"):
        register_tool(tool)


def test_get_tools_for_agent_returns_allowlisted_subset() -> None:
    spec = LlmAgentSpec(
        name="test-agent",
        type="llm_agent",
        project="fba",
        task="Use a browser.",
        tools=["browser_gather"],
        output="BrandScreenerResult",
        model="openai:gpt-5.4",
    )

    tools = get_tools_for_agent(spec)

    assert [tool.name for tool in tools] == ["browser_gather"]


def test_get_tools_for_agent_includes_toolbox_tools() -> None:
    spec = LlmAgentSpec(
        name="test-agent",
        type="llm_agent",
        project="fba",
        task="Use project and toolbox tools.",
        tools=["keepa_analyze"],
        toolbox=["db_read"],
        output="DataRunnerResult",
        model="openai:gpt-5.4",
    )

    tools = get_tools_for_agent(spec)

    assert [tool.name for tool in tools] == ["keepa_analyze", "db_read"]


def test_get_tools_for_agent_unknown_tool_raises_clear_error() -> None:
    spec = LlmAgentSpec(
        name="test-agent",
        type="llm_agent",
        project="fba",
        task="Use a missing tool.",
        tools=["missing_tool"],
        output="BrandScreenerResult",
        model="openai:gpt-5.4",
    )

    with pytest.raises(ValueError, match="Unknown tool: missing_tool"):
        get_tools_for_agent(spec)


def test_list_tools_filters_by_project() -> None:
    tools = list_tools(project="fba")

    assert tools
    assert all(tool.project == "fba" for tool in tools)
    assert "browser_gather" in {tool.name for tool in tools}
    assert "error_log" not in {tool.name for tool in tools}


def test_list_tools_returns_all_tools() -> None:
    names = {tool.name for tool in list_tools()}

    assert {"browser_gather", "keepa_download", "db_read", "db_write", "error_log"}.issubset(names)
