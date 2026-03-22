#!/usr/bin/env python3
"""Create tables and insert sample rows (requires DATABASE_URL)."""

from __future__ import annotations

import asyncio
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from nanochat.db.models import Base
from nanochat.db.repository import ensure_chat_session, insert_metric_event, insert_training_text
from nanochat.db.session import create_async_engine_from_url, make_session_factory


async def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("Set DATABASE_URL (e.g. postgresql+asyncpg://user:pass@localhost/nanochat)", file=sys.stderr)
        sys.exit(1)
    engine = create_async_engine_from_url(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = make_session_factory(engine)
    async with factory() as session:
        await ensure_chat_session(
            session,
            session_id="sess_seed",
            ab_group="A",
            model="llama3.2",
            backend="ollama",
        )
        await insert_metric_event(
            session,
            session_id="sess_seed",
            metric_name="tokens_per_second",
            metric_value=42.0,
            ab_group="A",
            model="llama3.2",
        )
        await insert_training_text(session, "Seed document for pretraining smoke test.", source="seed")
        await session.commit()
    await engine.dispose()
    print("OK: schema ensured + seed rows committed.")


if __name__ == "__main__":
    asyncio.run(main())
