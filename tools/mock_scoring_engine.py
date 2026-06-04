TOOL_NAME = "scoring_engine"
TOOL_DESCRIPTION = "Mock: scores analyzed Keepa products."
TOOL_PARAMS = {"analysis": {"type": "object", "required": True}}


async def handler(params):
    analysis = params.get("analysis") or {}
    products = analysis.get("products") or []
    return {
        "scored_products": [
            {
                "asin": product.get("asin", "B000TEST"),
                "brand": product.get("brand", "TestBrand"),
                "score": 87.5,
                "recommendation": "A",
                "metrics": {"roi": 0.42, "monthly_sales": product.get("monthly_sales", 0)},
            }
            for product in products
        ]
    }
