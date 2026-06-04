from __future__ import annotations

import pytest

from agentic_factory import executor
from agentic_factory.compiler import compile_prompt
from agentic_factory.loader import get_agent
from agentic_factory.schemas import DataRunnerResult, KeepaDownloadResult
from agentic_factory.spec import LlmAgentSpec
from agentic_factory.tools.registry import get_tools_for_agent


def test_data_runner_yaml_loads() -> None:
    spec = get_agent("data-runner")

    assert isinstance(spec, LlmAgentSpec)
    assert spec.output == "DataRunnerResult"


def test_data_runner_tools_resolve() -> None:
    spec = get_agent("data-runner")

    tools = get_tools_for_agent(spec)

    assert [tool.name for tool in tools] == [
        "keepa_download",
        "keepa_analyze",
        "scoring_engine",
        "db_write",
    ]


def test_data_runner_result_accepts_string_errors() -> None:
    result = DataRunnerResult(
        brand="Test",
        total_products_analyzed=0,
        scored_products=[],
        errors=["some error string", {"tool": "keepa_analyze", "message": "failed"}],
    )

    assert result.errors == ["some error string", {"tool": "keepa_analyze", "message": "failed"}]


def test_keepa_download_result_defaults_errors_to_empty_list() -> None:
    result = KeepaDownloadResult(
        total_urls=1,
        successful=1,
        failed=0,
        total_rows=10,
        total_size_mb=0.5,
        files=[],
    )

    assert result.errors == []


def test_normalize_raw_scoring_output() -> None:
    raw = {
        "brand": "Corelle",
        "total_products_analyzed": 2,
        "scored_products": [
            {
                "ASIN": "B0B43YTJQ2",
                "Brand": "Corelle",
                "final_score": 69.0,
                "grade": "A",
                "profit_p50": 2829.68,
                "roi_p50": 0.2672,
                "market_openness_score": 50,
                "v3_excluded": False,
            },
            {
                "ASIN": "B07MFC35ZP",
                "Brand": "Corelle",
                "final_score": 42.0,
                "grade": "C",
                "profit_p50": 150.0,
                "roi_p50": 0.15,
                "market_openness_score": 30,
                "v3_excluded": False,
            },
        ],
        "errors": [],
    }

    normalized = DataRunnerResult.normalize(raw)
    result = DataRunnerResult(**normalized)

    assert result.scored_products[0].asin == "B0B43YTJQ2"
    assert result.scored_products[0].brand == "Corelle"
    assert result.scored_products[0].score == 69.0
    assert result.scored_products[0].recommendation == "A"
    assert "profit_p50" in result.scored_products[0].metrics
    assert result.scored_products[1].asin == "B07MFC35ZP"


def test_data_runner_prompt_compiles() -> None:
    spec = get_agent("data-runner")

    compiled = compile_prompt(spec, {"brand": "Corelle", "product_urls": ["https://keepa.com/#!finder/test"]})

    assert 'Process the brand "Corelle"' in compiled["user"]
    assert "keepa_download" in compiled["system"]
    assert "keepa_analyze" in compiled["system"]
    assert "scoring_engine" in compiled["system"]
    assert "DataRunnerResult" in compiled["system"]


@pytest.mark.asyncio
async def test_data_runner_tool_enforcement(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_fetch_box_tools(project, allowlist):
        captured["project"] = project
        captured["allowlist"] = allowlist
        return [
            {"name": "keepa_download", "description": "Fake keepa_download", "params": {"urls": {"type": "array"}}},
            {"name": "keepa_analyze", "description": "Fake keepa_analyze", "params": {"csv_path": {"type": "string"}}},
            {"name": "scoring_engine", "description": "Fake scoring_engine", "params": {"analysis": {"type": "object"}}},
            {"name": "db_write", "description": "Fake db_write", "params": {"table": {"type": "string"}}},
        ]

    async def fake_call_llm(model, messages, spec):
        captured["system"] = messages[0]["content"]
        return '{"final_answer": {"brand": "Corelle", "total_products_analyzed": 0, "scored_products": [], "errors": []}}'

    monkeypatch.setattr(executor, "_fetch_box_tools", fake_fetch_box_tools)
    monkeypatch.setattr(executor, "_call_llm", fake_call_llm)

    await executor.execute_llm_agent(get_agent("data-runner"), {"brand": "Corelle"})

    assert captured["project"] == "fba-command-center"
    assert captured["allowlist"] == ["keepa_download", "keepa_analyze", "scoring_engine", "db_write"]
    assert "keepa_download" in captured["system"]
    assert "keepa_analyze" in captured["system"]
    assert "scoring_engine" in captured["system"]
    assert "db_write" in captured["system"]
    assert "browser_gather" not in captured["system"]
    assert "error_log" not in captured["system"]


@pytest.mark.asyncio
async def test_data_runner_execute_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}
    fake_tool_names = ["keepa_download", "keepa_analyze", "scoring_engine", "db_write"]

    async def fake_fetch_box_tools(project, allowlist):
        captured["tool_names"] = allowlist
        return [{"name": name, "description": f"Fake {name}", "params": {}} for name in fake_tool_names]

    async def fake_execute_tool(project, tool_name, params, tool_defs):
        captured["tool_call"] = {"project": project, "tool_name": tool_name, "params": params}
        return {"csv_path": "/tmp/corelle.csv", "rows": 500}

    responses = [
        '{"tool_call": {"name": "keepa_download", "params": {"urls": ["https://keepa.com/#!finder/test"]}}}',
        '{"final_answer": {"brand": "Corelle", "total_products_analyzed": 1, "scored_products": [{"asin": "B000TEST", "brand": "Corelle", "score": 87.5, "recommendation": "A", "metrics": {"roi": 0.42}}], "errors": []}}',
    ]

    async def fake_call_llm_sequence(model, messages, spec):
        captured["model"] = model
        captured["user_message"] = messages[1]["content"]
        captured["message_count"] = len(messages)
        return responses.pop(0)

    monkeypatch.setattr(executor, "_fetch_box_tools", fake_fetch_box_tools)
    monkeypatch.setattr(executor, "_call_llm", fake_call_llm_sequence)
    monkeypatch.setattr(executor, "_execute_tool", fake_execute_tool)

    result = await executor.execute_llm_agent(
        get_agent("data-runner"),
        {"brand": "Corelle", "product_urls": ["https://keepa.com/#!finder/test"]},
    )

    assert captured["model"] == "openai:gpt-5.4"
    assert captured["tool_names"] == fake_tool_names
    assert 'Process the brand "Corelle"' in captured["user_message"]
    assert captured["tool_call"] == {
        "project": "fba-command-center",
        "tool_name": "keepa_download",
        "params": {"urls": ["https://keepa.com/#!finder/test"]},
    }
    assert captured["message_count"] == 4
    assert result.output["brand"] == "Corelle"
    assert result.output["total_products_analyzed"] == 1
    assert result.tokens_input is None
    assert result.tokens_output is None
