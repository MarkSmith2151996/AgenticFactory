TOOL_NAME = "db_write"
TOOL_DESCRIPTION = "Mock: pretends to write scored products to a database table."
TOOL_PARAMS = {
    "table": {"type": "string", "required": True},
    "data": {"type": "object", "required": True},
}


async def handler(params):
    data = params.get("data") or {}
    rows = data.get("scored_products") or []
    return {"table": params.get("table"), "rows_written": len(rows), "ok": True}
