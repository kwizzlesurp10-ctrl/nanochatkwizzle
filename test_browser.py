import asyncio
from nanochat.browser_tool import BrowserManager

async def test():
    manager = BrowserManager(headless=True)
    print("Starting browser...")
    await manager.start()
    print("Navigating to example.com...")
    result = await manager.navigate("https://example.com")
    print(f"Result: {result}")
    print("Getting page summary...")
    summary = await manager.get_page_summary()
    print(f"Summary: {summary}")
    await manager.close()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(test())
