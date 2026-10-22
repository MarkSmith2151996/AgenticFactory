"""Playwright connection helper for self-hosted Steel Browser sessions."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from types import TracebackType
from typing import Any


STEEL_API_URL = "http://localhost:3010"
DOWNLOAD_PATH = "/data/downloads"
STEEL_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class SteelPlaywrightSession:
    """Async context manager for one Steel Browser + Playwright CDP session."""

    def __init__(
        self,
        steel_api_url: str = STEEL_API_URL,
        download_path: str = DOWNLOAD_PATH,
        user_agent: str = STEEL_USER_AGENT,
    ) -> None:
        self.steel_api_url = steel_api_url.rstrip("/")
        self.download_path = download_path
        self.user_agent = user_agent
        self.session_id: str | None = None
        self.raw_ws_url: str | None = None
        self.normalized_ws_url: str | None = None
        self.playwright: Any = None
        self.browser: Any = None

    async def __aenter__(self) -> "SteelPlaywrightSession":
        from playwright.async_api import async_playwright

        session = await asyncio.to_thread(self._create_steel_session)
        self.session_id = str(session.get("id") or "")
        if not self.session_id:
            raise RuntimeError(f"Steel session response missing id: {session}")

        self.raw_ws_url = self._extract_ws_url(session)
        self.normalized_ws_url = self._normalize_ws_url(self.raw_ws_url)
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.connect_over_cdp(self.normalized_ws_url)
        await self._set_download_behavior()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.browser is not None:
            await self.browser.close()
        if self.playwright is not None:
            await self.playwright.stop()
        if self.session_id:
            await asyncio.to_thread(self._release_steel_session, self.session_id)

    async def new_page(self) -> Any:
        """Create a new Playwright page in the connected Steel browser."""
        if self.browser is None:
            raise RuntimeError("SteelPlaywrightSession is not connected")
        context = self.browser.contexts[0] if self.browser.contexts else await self.browser.new_context()
        return await context.new_page()

    async def _set_download_behavior(self) -> None:
        if self.browser is None:
            raise RuntimeError("SteelPlaywrightSession is not connected")
        cdp = await self.browser.new_browser_cdp_session()
        await cdp.send(
            "Browser.setDownloadBehavior",
            {"behavior": "allow", "downloadPath": self.download_path},
        )

    def _create_steel_session(self) -> dict[str, Any]:
        body = json.dumps({"userAgent": self.user_agent}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.steel_api_url}/v1/sessions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError("Steel session API returned a non-object JSON response")
        return decoded

    def _release_steel_session(self, session_id: str) -> None:
        request = urllib.request.Request(
            f"{self.steel_api_url}/v1/sessions/{session_id}",
            method="DELETE",
        )
        try:
            urllib.request.urlopen(request, timeout=30).close()
            return
        except urllib.error.HTTPError:
            # Self-hosted Steel currently returns HTTP 500 even when release succeeds.
            pass
        except Exception:
            pass

        fallback = urllib.request.Request(
            f"{self.steel_api_url}/v1/sessions/{session_id}/release",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(fallback, timeout=30).close()
        except Exception:
            return

    @staticmethod
    def _extract_ws_url(session: dict[str, Any]) -> str:
        for key in ("websocketUrl", "webSocketUrl", "ws", "wsUrl", "cdpUrl", "debuggerUrl"):
            value = session.get(key)
            if isinstance(value, str) and value.startswith(("ws://", "wss://")):
                return value
        raise RuntimeError(f"Steel session response missing WebSocket URL: {session}")

    def _normalize_ws_url(self, ws_url: str) -> str:
        parsed_steel = urllib.parse.urlparse(self.steel_api_url)
        parsed_ws = urllib.parse.urlparse(ws_url)
        if parsed_ws.hostname == "0.0.0.0" and parsed_ws.port == 3000:
            host = parsed_steel.hostname or "localhost"
            port = parsed_steel.port or (443 if parsed_steel.scheme == "https" else 80)
            scheme = "wss" if parsed_steel.scheme == "https" else "ws"
            path = parsed_ws.path or "/"
            query = f"?{parsed_ws.query}" if parsed_ws.query else ""
            return f"{scheme}://{host}:{port}{path}{query}"
        return ws_url


async def _smoke_test() -> None:
    async with SteelPlaywrightSession() as session:
        print(f"raw_ws_url={session.raw_ws_url}")
        print(f"normalized_ws_url={session.normalized_ws_url}")
        page = await session.new_page()
        await page.goto("https://httpbin.org/get")
        body = await page.text_content("body")
        print(f"loaded_httpbin_origin={'origin' in (body or '')}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
