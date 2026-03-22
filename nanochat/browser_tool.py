import asyncio
import os
import json
import logging
from typing import Dict, Any, List, Optional
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Configure logging
logger = logging.getLogger(__name__)

class BrowserManager:
    """Manages browser sessions and automation for NanoChat."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def start(self):
        """Initialize Playwright and launch the browser."""
        if not self._playwright:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="NanoChatBrowser/1.0 (Automated Tool)"
            )
            self._page = await self._context.new_page()
            logger.info("Browser session started (Headless: %s)", self.headless)

    async def close(self):
        """Close browser and clean up Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser session closed.")

    async def navigate(self, url: str) -> str:
        """Navigate to a URL and return a simplified page summary."""
        if not self._page:
            await self.start()
        try:
            await self._page.goto(url, wait_until="networkidle", timeout=30000)
            title = await self._page.title()
            return f"Successfully navigated to '{url}'. Page title: '{title}'."
        except Exception as e:
            return f"Navigation error: {str(e)}"

    async def click(self, selector: str) -> str:
        """Click an element matching the selector."""
        try:
            await self._page.click(selector, timeout=5000)
            return f"Clicked element matching '{selector}'."
        except Exception as e:
            return f"Click error: {str(e)}"

    async def type_text(self, selector: str, text: str, press_enter: bool = False) -> str:
        """Type text into a field and optionally press Enter."""
        try:
            await self._page.fill(selector, text, timeout=5000)
            if press_enter:
                await self._page.press(selector, "Enter")
                return f"Typed '{text}' and pressed Enter in '{selector}'."
            return f"Typed '{text}' into '{selector}'."
        except Exception as e:
            return f"Typing error: {str(e)}"

    async def get_page_summary(self) -> str:
        """Extract a text-based summary of the current page for model context."""
        try:
            # Simple heuristic: get interactive elements and headers
            elements = await self._page.evaluate("""() => {
                const results = [];
                // Get all buttons, links, and inputs
                document.querySelectorAll('button, a, input, h1, h2, h3').forEach(el => {
                    const rect = el.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        results.push({
                            tag: el.tagName,
                            text: el.innerText.substring(0, 50).trim(),
                            id: el.id,
                            role: el.role,
                            type: el.type
                        });
                    }
                });
                return results.slice(0, 20); // Limit to top 20 interactive elements
            }""")
            summary = f"Page Summary: {await self._page.title()}\n"
            summary += "Top elements:\n"
            for el in elements:
                summary += f"- [{el['tag']}] {el['text']} (Selector hint: {el['id'] or el['text']})\n"
            return summary
        except Exception as e:
            return f"Summary extraction error: {str(e)}"

async def execute_browser_tool(action: str, params: Dict[str, Any]) -> str:
    """Entry point for the inference engine to call browser tools."""
    # This would typically be held in a persistent session state in chat_web.py
    manager = BrowserManager()
    await manager.start()
    try:
        if action == "navigate":
            return await manager.navigate(params["url"])
        elif action == "click":
            return await manager.click(params["selector"])
        elif action == "type":
            return await manager.type_text(params["selector"], params["text"], params.get("enter", False))
        elif action == "summarize":
            return await manager.get_page_summary()
        else:
            return f"Unknown action: {action}"
    finally:
        await manager.close()
