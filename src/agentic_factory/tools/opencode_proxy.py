"""OpenAI Chat Completions -> ChatGPT Codex backend proxy.

Accepts standard /v1/chat/completions requests (what browser-use sends via
its bundled OpenAI client) and forwards them through the OAuth token that
OpenCode keeps in ~/.local/share/opencode/auth.json. The upstream endpoint
is https://chatgpt.com/backend-api/codex/responses, which speaks the OpenAI
Responses API over SSE. We translate request + response shapes both ways.

Run:
    .venv/bin/python -m agentic_factory.tools.opencode_proxy
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


AUTH_PATH = Path(os.path.expanduser("~/.local/share/opencode/auth.json"))
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"

MODEL_MAP = {
    "gpt-4o-mini": "gpt-5.4-mini",
    "gpt-4o": "gpt-5.4",
    "gpt-4.1-mini": "gpt-5.4-mini",
    "gpt-4.1": "gpt-5.4",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.5": "gpt-5.5",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.3-codex": "gpt-5.3-codex",
}

LOGGER = logging.getLogger("opencode_proxy")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _load_auth() -> dict[str, Any]:
    return json.loads(AUTH_PATH.read_text())


def _save_auth(auth: dict[str, Any]) -> None:
    AUTH_PATH.write_text(json.dumps(auth, indent=2))
    os.chmod(AUTH_PATH, 0o600)


async def _refresh_token(auth: dict[str, Any]) -> dict[str, Any]:
    refresh = auth["openai"]["refresh"]
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            OAUTH_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": CLIENT_ID,
            },
        )
    resp.raise_for_status()
    data = resp.json()
    auth["openai"]["access"] = data["access_token"]
    if "refresh_token" in data:
        auth["openai"]["refresh"] = data["refresh_token"]
    expires_in = data.get("expires_in", 3600)
    auth["openai"]["expires"] = int((time.time() + expires_in) * 1000)
    _save_auth(auth)
    LOGGER.info("refreshed OAuth token; new expiry %s", auth["openai"]["expires"])
    return auth


async def _get_token_and_account() -> tuple[str, str]:
    auth = _load_auth()
    oa = auth["openai"]
    if oa["expires"] / 1000 <= time.time() + 60:
        auth = await _refresh_token(auth)
        oa = auth["openai"]
    return oa["access"], oa["accountId"]


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                out.append(part.get("text", ""))
            elif isinstance(part, str):
                out.append(part)
        return "".join(out)
    return str(content)


def _content_to_response_parts(content: Any, role: str) -> list[dict[str, Any]]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": text_type, "text": content}] if content else []
    if not isinstance(content, list):
        text = str(content)
        return [{"type": text_type, "text": text}] if text else []

    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, str):
            if part:
                parts.append({"type": text_type, "text": part})
            continue
        if not isinstance(part, dict):
            continue

        part_kind = part.get("type")
        if part_kind in ("text", "input_text", "output_text"):
            text = part.get("text", "")
            if text:
                parts.append({"type": text_type, "text": text})
        elif role == "user" and part_kind == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if image_url:
                parts.append({"type": "input_image", "image_url": image_url})
        elif role == "user" and part_kind == "input_image":
            image_url = part.get("image_url") or part.get("image")
            if image_url:
                parts.append({"type": "input_image", "image_url": image_url})
    return parts


def _translate_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        text = _content_to_text(msg.get("content"))
        if role == "system":
            if text:
                instructions_parts.append(text)
            continue
        if role not in ("user", "assistant"):
            continue
        content_parts = _content_to_response_parts(msg.get("content"), role)
        if content_parts:
            input_items.append({
                "type": "message",
                "role": role,
                "content": content_parts,
            })
    instructions = "\n\n".join(p for p in instructions_parts if p) or "You are a helpful assistant."
    return instructions, input_items


def _translate_response_format(rf: dict[str, Any] | None) -> dict[str, Any] | None:
    if not rf:
        return None
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema", {})
        return {
            "format": {
                "type": "json_schema",
                "name": js.get("name", "response"),
                "strict": js.get("strict", True),
                "schema": js.get("schema", {}),
            }
        }
    if rf.get("type") == "json_object":
        return {"format": {"type": "json_object"}}
    return None


def _build_codex_payload(req: dict[str, Any]) -> dict[str, Any]:
    requested = req.get("model", "gpt-5.4-mini")
    model = MODEL_MAP.get(requested, requested)
    instructions, input_items = _translate_messages(req.get("messages", []))
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "stream": True,
        "store": False,
    }
    text_block = _translate_response_format(req.get("response_format"))
    if text_block:
        payload["text"] = text_block
    return payload


async def _call_codex(payload: dict[str, Any], structured: bool = False) -> dict[str, Any]:
    access, account = await _get_token_and_account()
    headers = {
        "Authorization": f"Bearer {access}",
        "Content-Type": "application/json",
        "chatgpt-account-id": account,
        "OpenAI-Beta": "responses=experimental",
        "Accept": "text/event-stream",
    }
    text_parts: list[str] = []
    finish = "stop"
    usage: dict[str, Any] = {}
    response_id = ""
    created_at = int(time.time())
    model_returned = payload["model"]
    error: dict[str, Any] | None = None

    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream("POST", CODEX_URL, headers=headers, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"codex backend rejected: {body.decode('utf-8', 'replace')[:500]}",
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    evt = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type", "")
                if etype == "response.output_text.delta":
                    text_parts.append(evt.get("delta", ""))
                elif etype == "response.completed":
                    r = evt.get("response", {})
                    response_id = r.get("id", response_id)
                    created_at = r.get("created_at", created_at)
                    model_returned = r.get("model", model_returned)
                    u = r.get("usage") or {}
                    usage = {
                        "prompt_tokens": u.get("input_tokens", 0),
                        "completion_tokens": u.get("output_tokens", 0),
                        "total_tokens": u.get("total_tokens", 0),
                        "prompt_tokens_details": {
                            "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
                        },
                    }
                    incomplete = r.get("incomplete_details")
                    if incomplete and "max" in str(incomplete.get("reason", "")):
                        finish = "length"
                elif etype in ("response.failed", "error"):
                    error = evt.get("response", {}).get("error") or evt.get("error") or {"message": str(evt)}

    if error:
        raise HTTPException(status_code=502, detail=f"codex backend error: {error}")
    if not text_parts:
        raise HTTPException(status_code=502, detail="codex backend returned no text")

    return {
        "id": response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created_at,
        "model": model_returned,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _final_content("".join(text_parts), structured)},
                "finish_reason": finish,
            }
        ],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _extract_json_object(text: str) -> str:
    """Return the first complete JSON value in text, stripping any trailing junk.

    gpt-5.4-mini occasionally appends stray tokens after the schema-validated
    object; pydantic strict mode then refuses the whole string. raw_decode
    walks the JSON exactly and returns the slice up to the boundary.
    """
    stripped = text.lstrip()
    leading = len(text) - len(stripped)
    decoder = json.JSONDecoder()
    try:
        _value, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return text
    return text[:leading + end]


def _final_content(text: str, structured: bool) -> str:
    if structured:
        return _extract_json_object(text)
    return text


app = FastAPI(title="opencode-proxy")


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    now = int(time.time())
    ids = sorted(set(MODEL_MAP.values()))
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": now, "owned_by": "opencode-proxy"} for m in ids],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    if body.get("stream"):
        raise HTTPException(status_code=400, detail="proxy does not implement streaming")
    payload = _build_codex_payload(body)
    LOGGER.info(
        "chat.completions model=%s -> codex model=%s msgs=%d resp_fmt=%s",
        body.get("model"),
        payload["model"],
        len(body.get("messages", [])),
        body.get("response_format", {}).get("type") if body.get("response_format") else None,
    )
    result = await _call_codex(payload, structured=bool(body.get("response_format")))
    return JSONResponse(result)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    auth = _load_auth()
    oa = auth.get("openai", {})
    return {
        "ok": True,
        "token_expires_in_hours": round((oa.get("expires", 0) / 1000 - time.time()) / 3600, 2),
        "models": sorted(set(MODEL_MAP.values())),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4096, log_level="info")
