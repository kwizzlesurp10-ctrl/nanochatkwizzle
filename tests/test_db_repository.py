from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from nanochat.db.models import AnalyticsEvent, Base, ChatSession, UserTrainingText
from nanochat.db.repository import (
    ensure_chat_session,
    insert_feedback_event,
    insert_metric_event,
    insert_training_text,
)


async def _with_memory_session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return engine, factory


def test_ensure_chat_session_creates_and_updates():
    async def body():
        engine, factory = await _with_memory_session()
        try:
            async with factory() as session:
                await ensure_chat_session(
                    session,
                    session_id="s1",
                    ab_group="A",
                    model="m1",
                    backend="ollama",
                )
                await session.commit()

            async with factory() as session:
                await ensure_chat_session(
                    session,
                    session_id="s1",
                    ab_group="B",
                    model="m2",
                    backend="nanochat",
                )
                await session.commit()

            async with factory() as session:
                r = await session.execute(select(ChatSession).where(ChatSession.id == "s1"))
                row = r.scalar_one()
                assert row.ab_group == "B"
                assert row.model == "m2"
                assert row.backend == "nanochat"
        finally:
            await engine.dispose()

    asyncio.run(body())


def test_insert_metric_and_feedback():
    async def body():
        engine, factory = await _with_memory_session()
        try:
            async with factory() as session:
                await ensure_chat_session(
                    session,
                    session_id="s2",
                    ab_group="A",
                    model="llama3.2",
                    backend="ollama",
                )
                await insert_metric_event(
                    session,
                    session_id="s2",
                    metric_name="total_tokens",
                    metric_value=10.0,
                    ab_group="A",
                    model="llama3.2",
                )
                await insert_feedback_event(
                    session,
                    session_id="s2",
                    message_index=0,
                    feedback="thumb_up",
                    ab_group="A",
                    model="llama3.2",
                )
                await session.commit()

            async with factory() as session:
                r = await session.execute(select(AnalyticsEvent).where(AnalyticsEvent.session_id == "s2"))
                rows = r.scalars().all()
                assert len(rows) == 2
                assert {x.event_type for x in rows} == {"metric", "feedback"}
        finally:
            await engine.dispose()

    asyncio.run(body())


def test_insert_training_text():
    async def body():
        engine, factory = await _with_memory_session()
        try:
            async with factory() as session:
                await insert_training_text(session, "hello world", source="test")
                await session.commit()

            async with factory() as session:
                r = await session.execute(select(UserTrainingText))
                row = r.scalar_one()
                assert row.body == "hello world"
                assert row.source == "test"
        finally:
            await engine.dispose()

    asyncio.run(body())
