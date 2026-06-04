def handle(params):
    verdict = params.get("verdict")
    valid = {"PROCEED", "SKIP", "NEEDS_MANUAL"}
    if verdict not in valid:
        return {"ok": False, "error": "invalid verdict %r, must be one of %s" % (verdict, sorted(valid))}
    conf = params.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except (TypeError, ValueError):
        conf = None
    flags = params.get("flags") or []
    if isinstance(flags, str):
        flags = [flags]
    return {
        "ok": True,
        "verdict": verdict,
        "confidence": conf,
        "reasoning": params.get("reasoning", ""),
        "flags": list(flags),
    }
