"""brand-amazon-screener agent handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent

from agentic_factory.schemas.brand_screener import BrandScreenerResult
from agentic_factory.tools.browser_search import browser_gather


MODEL_NAME = "openai:gpt-5.4"

SYSTEM_PROMPT = """You are a wholesale FBA brand screener. You receive raw Amazon search data about a brand and produce a structured screening verdict.

DECISION FRAMEWORK — asymmetric cost:
- False SKIP costs real money (missed profitable brand)
- False PROCEED costs one Keepa download (~$0.10 and 2 minutes)
- Therefore: BIAS TOWARD PROCEED

PROCEED: >50 Amazon results AND (consumer_retail or mixed market_type) AND not hazmat_heavy AND typical price band overlaps $15-$500 range
SKIP: <50 Amazon results, OR hazmat_heavy, OR B2B_industrial only, OR brand does not exist on Amazon
NEEDS_MANUAL: Only when signals are genuinely contradictory (e.g., high result count but all results are B2B/industrial)

distribution_model is always "unknown" — do not guess. A downstream agent handles this.

If browser data is missing or errored, return NEEDS_MANUAL with confidence "low". Never SKIP on incomplete data.
"""


@dataclass(frozen=True)
class BrandScreenerInvocation:
    """Structured result plus model usage from the analysis phase."""

    output: BrandScreenerResult
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: float | None = None


def _manual_result(brand: str, reason: str) -> BrandScreenerInvocation:
    return BrandScreenerInvocation(
        output=BrandScreenerResult(
            brand=brand,
            amazon_catalog_size_estimate="<50",
            typical_price_band="unknown",
            hazmat_heavy=False,
            market_type="unknown",
            distribution_model="unknown",
            verdict="NEEDS_MANUAL",
            verdict_reason=reason,
            confidence="low",
        )
    )


def _normalize_result(output: BrandScreenerResult) -> BrandScreenerResult:
    """Enforce hard screening rules that the model can otherwise soften."""
    updates: dict[str, Any] = {"distribution_model": "unknown"}

    if output.confidence != "low":
        if output.hazmat_heavy:
            updates.update(
                verdict="SKIP",
                verdict_reason="SKIP because Amazon results indicate hazmat-heavy or restricted goods.",
            )
        elif output.market_type == "B2B_industrial":
            updates.update(
                verdict="SKIP",
                verdict_reason="SKIP because Amazon results are primarily B2B/industrial supplies.",
            )
        elif output.amazon_catalog_size_estimate == "<50":
            updates.update(
                verdict="SKIP",
                verdict_reason="SKIP because Amazon catalog size is below 50 results.",
            )

    return output.model_copy(update=updates)


async def screen_brand(brand: str) -> BrandScreenerInvocation:
    """Gather browser evidence for a brand and produce a structured verdict."""
    amazon_result = await browser_gather(
        f"""Go to amazon.com and search for '{brand}'. Report:
1. Total number of results (exact number if shown, or estimate like "over 1,000")
2. Price range on the first page (lowest and highest price)
3. Top 3 product categories visible
4. Whether any results show hazmat, dangerous goods, or restricted item warnings
5. Whether results are primarily consumer products or industrial/B2B supplies"""
    )

    user_message = f"""Brand: {brand}

Amazon research:
{amazon_result}
"""

    try:
        agent = Agent(
            MODEL_NAME,
            output_type=BrandScreenerResult,
            system_prompt=SYSTEM_PROMPT,
            output_retries=2,
        )
        run_result = await agent.run(user_message)
    except Exception as exc:
        return _manual_result(brand, f"Structured analysis failed: {exc}")

    usage = run_result.usage
    output = _normalize_result(run_result.output)
    return BrandScreenerInvocation(
        output=output,
        tokens_input=usage.input_tokens or None,
        tokens_output=usage.output_tokens or None,
        cost_usd=None,
    )


async def invoke_brand_amazon_screener(input_payload: dict[str, Any]) -> Any:
    """Runtime entrypoint for brand-amazon-screener."""
    brand = str(input_payload.get("brand", "")).strip()
    if not brand:
        raise ValueError("brand-amazon-screener requires input.brand")

    result = await screen_brand(brand)

    from agentic_factory.agents import AgentInvocationResult

    return AgentInvocationResult(
        output=result.output.model_dump(),
        tokens_input=result.tokens_input,
        tokens_output=result.tokens_output,
        cost_usd=result.cost_usd,
    )
