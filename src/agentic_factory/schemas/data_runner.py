"""Structured output schema for data-runner."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoredProduct(BaseModel):
    asin: str
    brand: str
    score: float
    recommendation: str
    metrics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> dict[str, Any]:
        field_map = {
            "ASIN": "asin",
            "Brand": "brand",
            "final_score": "score",
            "grade": "recommendation",
        }
        mapped: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        for key, value in raw.items():
            if key in field_map:
                mapped[field_map[key]] = value
            elif key.lower() in {"asin", "brand", "score", "recommendation"}:
                mapped[key.lower()] = value
            else:
                metrics[key] = value

        mapped.setdefault("asin", raw.get("ASIN", raw.get("asin", "")))
        mapped.setdefault("brand", raw.get("Brand", raw.get("brand", "")))
        mapped.setdefault("score", raw.get("final_score", raw.get("score", 0.0)))
        mapped.setdefault("recommendation", raw.get("grade", raw.get("recommendation", "unknown")))
        mapped["metrics"] = metrics
        return mapped


class DataRunnerResult(BaseModel):
    brand: str
    total_products_analyzed: int
    scored_products: list[ScoredProduct]
    errors: list[str | dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def normalize(cls, raw: dict[str, Any]) -> dict[str, Any]:
        result = dict(raw)
        scored_products = result.get("scored_products")
        if isinstance(scored_products, list):
            result["scored_products"] = [
                ScoredProduct.from_raw(product) if isinstance(product, dict) and "ASIN" in product else product
                for product in scored_products
            ]
        return result
