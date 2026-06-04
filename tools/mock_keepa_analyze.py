TOOL_NAME = "keepa_analyze"
TOOL_DESCRIPTION = "Mock: returns fake product metrics from a Keepa CSV."
TOOL_PARAMS = {"csv_path": {"type": "string", "required": True}}


async def handler(params):
    return {
        "csv_path": params.get("csv_path"),
        "total_products": 1,
        "products": [
            {
                "asin": "B000TEST",
                "brand": "TestBrand",
                "price": 24.99,
                "monthly_sales": 500,
            }
        ],
    }
