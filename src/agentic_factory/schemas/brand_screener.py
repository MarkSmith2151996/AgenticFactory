"""Structured output schema for brand-amazon-screener."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BrandScreenerResult(BaseModel):
    brand: str
    amazon_catalog_size_estimate: Literal["<50", "50-500", "500-2000", "2000-5000", "5000+"]
    typical_price_band: str = Field(description="e.g. '$10-$30'")
    hazmat_heavy: bool
    market_type: Literal["consumer_retail", "B2B_industrial", "mixed", "unknown"]
    distribution_model: Literal["brand_direct", "multi_distributor", "authorized_dealer_lockup", "unknown"]
    verdict: Literal["PROCEED", "SKIP", "NEEDS_MANUAL"]
    verdict_reason: str
    confidence: Literal["high", "medium", "low"]
