from __future__ import annotations

import json

import pytest

from agentic_factory import executor
from agentic_factory.loader import get_agent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_llm_agent_passes_compiled_prompt_to_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    async def fake_fetch_box_tools(project, allowlist):
        captured["project"] = project
        captured["allowlist"] = allowlist
        return [
            {
                "name": "browser_gather",
                "description": "Gather browser page data",
                "params": {"query": {"type": "string"}},
            }
        ]

    async def fake_call_llm(model, messages, spec):
        captured["model"] = model
        captured["messages"] = messages
        return json.dumps(
            {
                "final_answer": {
                    "brand": "Corelle",
                    "amazon_catalog_size_estimate": "500-2000",
                    "typical_price_band": "$10-$30",
                    "hazmat_heavy": False,
                    "market_type": "consumer_retail",
                    "distribution_model": "unknown",
                    "verdict": "PROCEED",
                    "verdict_reason": "Enough Amazon consumer retail signal.",
                    "confidence": "medium",
                }
            }
        )

    monkeypatch.setattr(executor, "_fetch_box_tools", fake_fetch_box_tools)
    monkeypatch.setattr(executor, "_call_llm", fake_call_llm)
    spec = get_agent("brand-amazon-screener")

    result = await executor.execute_llm_agent(spec, {"brand": "Corelle"})

    assert captured["project"] == "fba"
    assert captured["allowlist"] == ["browser_gather"]
    assert captured["model"] == "openai:gpt-5.4"
    assert captured["messages"][0]["role"] == "system"
    assert "# System Rules" not in captured["messages"][0]["content"]
    assert "browser_gather" in captured["messages"][0]["content"]
    assert captured["messages"][1]["role"] == "user"
    assert "Search for Corelle" in captured["messages"][1]["content"]
    assert result.output["brand"] == "Corelle"
    assert result.tokens_input is None
    assert result.tokens_output is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_execute_service_agent_posts_to_configured_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "status": "success",
                    "results": [
                        {
                            "url_index": 0,
                            "file": "~/keepa-downloads/batch_001_0.csv",
                            "rows": 5000,
                            "size_mb": 37.5,
                            "elapsed_s": 45.2,
                        }
                    ],
                    "errors": [],
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("KEEPA_DOWNLOADER_URL", "http://mac.test:8095")
    monkeypatch.setattr(executor.urllib.request, "urlopen", fake_urlopen)
    spec = get_agent("keepa-downloader")

    result = await executor.execute_service_agent(
        spec,
        {"urls": ["https://keepa.com/#!finder/test"], "output_dir": "~/keepa-downloads"},
    )

    assert captured["url"] == "http://mac.test:8095/download"
    assert captured["payload"] == {
        "urls": ["https://keepa.com/#!finder/test"],
        "output_dir": "~/keepa-downloads",
    }
    assert captured["timeout"] == 900
    assert result.output["total_urls"] == 1
    assert result.output["successful"] == 1
    assert result.output["total_rows"] == 5000
