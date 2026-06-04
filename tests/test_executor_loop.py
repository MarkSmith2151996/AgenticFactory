from __future__ import annotations

import pytest

from agentic_factory import executor
from agentic_factory.loader import get_agent


def test_parse_response_detects_tool_call_json_block() -> None:
    parsed = executor._parse_response(
        '```json\n{"tool_call": {"name": "keepa_download", "params": {"brand": "Corelle"}}}\n```'
    )

    assert parsed == {
        "type": "tool_call",
        "name": "keepa_download",
        "params": {"brand": "Corelle"},
    }


def test_parse_response_detects_bare_tool_call() -> None:
    parsed = executor._parse_response('{"name": "keepa_download", "params": {"brand": "Corelle"}}')

    assert parsed == {
        "type": "tool_call",
        "name": "keepa_download",
        "params": {"brand": "Corelle"},
    }


def test_parse_response_detects_final_answer() -> None:
    parsed = executor._parse_response('{"final_answer": {"brand": "Corelle", "errors": []}}')

    assert parsed == {"type": "final_answer", "data": {"brand": "Corelle", "errors": []}}


def test_parse_response_returns_invalid_for_plain_text() -> None:
    parsed = executor._parse_response("I cannot do that yet.")

    assert parsed["type"] == "invalid"
    assert "no JSON object found" in parsed["error"]


@pytest.mark.asyncio
async def test_execute_llm_agent_nudges_after_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def fake_fetch_box_tools(project, allowlist):
        return [{"name": "browser_gather", "description": "Browser", "params": {}}]

    responses = [
        "plain text",
        '{"final_answer": {"brand": "Corelle", "amazon_catalog_size_estimate": "500-2000", "typical_price_band": "$10-$30", "hazmat_heavy": false, "market_type": "consumer_retail", "distribution_model": "unknown", "verdict": "PROCEED", "verdict_reason": "Enough Amazon consumer retail signal.", "confidence": "medium"}}',
    ]

    async def fake_call_llm(model, messages, spec):
        calls.append(list(messages))
        return responses.pop(0)

    monkeypatch.setattr(executor, "_fetch_box_tools", fake_fetch_box_tools)
    monkeypatch.setattr(executor, "_call_llm", fake_call_llm)

    result = await executor.execute_llm_agent(get_agent("brand-amazon-screener"), {"brand": "Corelle"})

    assert result.output["brand"] == "Corelle"
    assert len(calls) == 2
    assert "Your response was not valid JSON" in calls[1][-1]["content"]


@pytest.mark.asyncio
async def test_execute_llm_agent_raises_after_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch_box_tools(project, allowlist):
        return [{"name": "browser_gather", "description": "Browser", "params": {}}]

    async def fake_call_llm(model, messages, spec):
        return '{"tool_call": {"name": "browser_gather", "params": {"query": "Corelle"}}}'

    async def fake_execute_tool(project, tool_name, params, tool_defs):
        return {"ok": True}

    monkeypatch.setattr(executor, "MAX_TOOL_ITERATIONS", 2)
    monkeypatch.setattr(executor, "_fetch_box_tools", fake_fetch_box_tools)
    monkeypatch.setattr(executor, "_call_llm", fake_call_llm)
    monkeypatch.setattr(executor, "_execute_tool", fake_execute_tool)

    with pytest.raises(RuntimeError, match="hit max iterations"):
        await executor.execute_llm_agent(get_agent("brand-amazon-screener"), {"brand": "Corelle"})
