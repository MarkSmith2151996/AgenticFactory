TOOL_NAME = "keepa_download"
TOOL_DESCRIPTION = "Mock: returns a fake Keepa CSV path."
TOOL_PARAMS = {"urls": {"type": "array", "items": {"type": "string"}, "required": True}}


async def handler(params):
    brand = params.get("brand") or "TestBrand"
    return {"csv_path": f"/data/{brand}.csv", "rows": 500}
