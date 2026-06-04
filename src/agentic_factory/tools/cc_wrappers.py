"""Thin wrappers around the existing CommandAndControl FBA analysis stack."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any


CC_ROOT = Path("/home/dev/projects/ComandAndControl")


def _import_cc_module(module_name: str):
    if str(CC_ROOT) not in sys.path:
        sys.path.insert(0, str(CC_ROOT))
    return importlib.import_module(module_name)


async def keepa_analyze_handler(
    input: dict[str, Any] | None = None,
    csv_path: str | None = None,
    csv_paths: list[str] | None = None,
    filter_profile: str | None = None,
) -> dict[str, Any]:
    """Wrap KeepaAnalyzer.analyze for one or more Keepa CSV paths."""
    payload = input or {}
    paths = _resolve_csv_paths(payload, csv_path, csv_paths)
    module = _import_cc_module("keepa_analyzer")
    settings = module.SettingsManager()
    analyzer = module.KeepaAnalyzer(settings, log_callback=lambda _message: None)

    results = []
    for path in paths:
        result = analyzer.analyze(path, filter_profile=filter_profile or payload.get("filter_profile"))
        if result is None:
            result = {"error": "KeepaAnalyzer returned no result"}
        result = dict(result)
        result["source_csv"] = path
        results.append(_jsonable(result))

    if len(results) == 1:
        return results[0]
    return {"results": results, "total_files": len(results)}


async def scoring_engine_handler(
    input: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    subcategory: str | None = None,
) -> dict[str, Any]:
    """Wrap ScoringEngine.score for analysis data or source CSVs."""
    payload = input or {}
    analysis_payload = analysis or payload.get("analysis") or payload
    module = _import_cc_module("scoring_engine")
    scored_frames = []

    for item in _iter_analysis_items(analysis_payload):
        df = _analysis_to_dataframe(item)
        engine = module.ScoringEngine()
        scored = engine.score(df, subcategory=subcategory or payload.get("subcategory"))
        scored_frames.append(scored)

    if not scored_frames:
        return {"total_products": 0, "scored_products": []}

    pd = _import_pandas()
    combined = pd.concat(scored_frames, ignore_index=True) if len(scored_frames) > 1 else scored_frames[0]
    records = [_jsonable(record) for record in combined.to_dict(orient="records")]
    return {
        "total_products": len(records),
        "scored_products": records,
        "summary": module.ScoringEngine().get_score_summary(combined) if hasattr(module.ScoringEngine, "get_score_summary") else {},
    }


def _resolve_csv_paths(
    payload: dict[str, Any],
    csv_path: str | None,
    csv_paths: list[str] | None,
) -> list[str]:
    value = csv_paths or payload.get("csv_paths")
    if value is None:
        single = csv_path or payload.get("csv_path")
        value = [single] if single else []
    if isinstance(value, str):
        value = [value]
    paths = [str(path) for path in value if str(path).strip()]
    if not paths:
        raise ValueError("keepa_analyze requires input.csv_path or input.csv_paths")
    return paths


def _iter_analysis_items(analysis_payload: Any) -> list[dict[str, Any]]:
    if isinstance(analysis_payload, dict) and isinstance(analysis_payload.get("results"), list):
        return [item for item in analysis_payload["results"] if isinstance(item, dict)]
    if isinstance(analysis_payload, dict):
        return [analysis_payload]
    raise ValueError("scoring_engine requires analysis data as an object")


def _analysis_to_dataframe(analysis: dict[str, Any]):
    pd = _import_pandas()
    if isinstance(analysis.get("data"), list):
        return pd.DataFrame(analysis["data"])
    if isinstance(analysis.get("products"), list):
        return pd.DataFrame(analysis["products"])
    csv_path = analysis.get("source_csv") or analysis.get("csv_path")
    if csv_path:
        return pd.read_csv(str(csv_path), on_bad_lines="skip", encoding="utf-8", low_memory=False)
    raise ValueError("scoring_engine requires analysis.data, analysis.products, or analysis.source_csv")


def _import_pandas():
    return importlib.import_module("pandas")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
