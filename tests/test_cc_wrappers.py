from __future__ import annotations

import pandas as pd
import pytest

from agentic_factory.tools import cc_wrappers


@pytest.mark.asyncio
async def test_keepa_analyze_wrapper_calls_cc_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeSettingsManager:
        pass

    class FakeKeepaAnalyzer:
        def __init__(self, settings, log_callback=None):
            calls["settings"] = settings
            calls["log_callback"] = log_callback

        def analyze(self, input_path, filter_profile=None):
            calls["input_path"] = input_path
            calls["filter_profile"] = filter_profile
            return {"total": 1, "passing": 1, "output": "/tmp/out.xlsx"}

    fake_module = type(
        "FakeKeepaAnalyzerModule",
        (),
        {"SettingsManager": FakeSettingsManager, "KeepaAnalyzer": FakeKeepaAnalyzer},
    )
    monkeypatch.setattr(cc_wrappers, "_import_cc_module", lambda name: fake_module)

    result = await cc_wrappers.keepa_analyze_handler({"csv_path": "/tmp/input.csv", "filter_profile": "product_finder"})

    assert calls["input_path"] == "/tmp/input.csv"
    assert calls["filter_profile"] == "product_finder"
    assert result["total"] == 1
    assert result["source_csv"] == "/tmp/input.csv"


@pytest.mark.asyncio
async def test_scoring_engine_wrapper_calls_cc_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeScoringEngine:
        def score(self, df, subcategory=None, weights=None):
            calls["rows"] = len(df)
            calls["subcategory"] = subcategory
            scored = df.copy()
            scored["final_score"] = [91.5]
            scored["grade"] = ["A"]
            return scored

        def get_score_summary(self, df):
            return {"total": len(df)}

    fake_module = type("FakeScoringModule", (), {"ScoringEngine": FakeScoringEngine})
    monkeypatch.setattr(cc_wrappers, "_import_cc_module", lambda name: fake_module)

    result = await cc_wrappers.scoring_engine_handler(
        {
            "analysis": {
                "data": [
                    {
                        "ASIN": "B000TEST",
                        "Brand": "Corelle",
                        "Price": 20,
                    }
                ]
            },
            "subcategory": "Home & Kitchen",
        }
    )

    assert calls == {"rows": 1, "subcategory": "Home & Kitchen"}
    assert result["total_products"] == 1
    assert result["scored_products"][0]["final_score"] == 91.5
    assert result["summary"] == {"total": 1}
