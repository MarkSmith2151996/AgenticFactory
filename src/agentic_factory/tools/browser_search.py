"""Reusable browser-use research wrapper."""

from __future__ import annotations

import logging
import os

from browser_use import Agent as BrowserAgent, Browser
from browser_use.llm.openai.chat import ChatOpenAI


LOGGER = logging.getLogger(__name__)

# Route through the local opencode-proxy that bridges browser-use's
# OpenAI-shaped calls to OpenCode's ChatGPT Codex OAuth credentials.
# Override either piece via env if you ever want to point at a real API key.
BROWSER_MODEL = os.environ.get("BROWSER_USE_MODEL", "gpt-4o-mini")
PROXY_BASE_URL = os.environ.get("BROWSER_USE_BASE_URL", "http://127.0.0.1:4096/v1")
PROXY_API_KEY = os.environ.get("BROWSER_USE_API_KEY", "opencode-proxy")


async def browser_gather(task: str, max_steps: int = 15) -> str:
    """Run a headless browser-use task and return the extracted result text."""
    browser = None
    try:
        browser = Browser(headless=True, use_cloud=False)
        llm = ChatOpenAI(
            model=BROWSER_MODEL,
            api_key=PROXY_API_KEY,
            base_url=PROXY_BASE_URL,
        )
        agent = BrowserAgent(task=task, llm=llm, browser=browser, max_failures=1)
        history = await agent.run(max_steps=max_steps)
        if history.has_errors() or history.is_successful() is False:
            errors = "; ".join(error for error in history.errors() if error)
            return f"ERROR: browser_gather failed: {errors or 'browser-use did not complete the task'}"
        result = history.final_result()
        if result:
            return str(result)
        extracted = history.extracted_content()
        return "\n".join(str(item) for item in extracted if item)
    except Exception as exc:
        message = f"ERROR: browser_gather failed: {exc}"
        LOGGER.warning(message)
        return message
    finally:
        if browser is not None:
            try:
                await browser.stop()
            except Exception as exc:
                LOGGER.warning("Failed to close browser-use session: %s", exc)
