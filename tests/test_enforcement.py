from __future__ import annotations

import pytest

from agentic_factory import executor
from agentic_factory.loader import get_agent


@pytest.mark.asyncio
async def test_execute_llm_agent_passes_only_allowlisted_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_fetch_box_tools(project, allowlist):
        captured["allowlist"] = allowlist
        return [
            {
                "name": "browser_gather",
                "description": "Gather browser data",
                "params": {"query": {"type": "string"}},
            }
        ]

    async def fake_call_llm(model, messages, spec):
        captured["system"] = messages[0]["content"]
        return '{"final_answer": {"brand": "Corelle", "amazon_catalog_size_estimate": "500-2000", "typical_price_band": "$10-$30", "hazmat_heavy": false, "market_type": "consumer_retail", "distribution_model": "unknown", "verdict": "PROCEED", "verdict_reason": "Enough Amazon consumer retail signal.", "confidence": "medium"}}'

    monkeypatch.setattr(executor, "_fetch_box_tools", fake_fetch_box_tools)
    monkeypatch.setattr(executor, "_call_llm", fake_call_llm)
    spec = get_agent("brand-amazon-screener")

    await executor.execute_llm_agent(spec, {"brand": "Corelle"})

    assert captured["allowlist"] == ["browser_gather"]
    assert "- browser_gather:" in captured["system"]
    assert "- keepa_download:" not in captured["system"]
    assert "- db_write:" not in captured["system"]
