from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nanochat.db.models import AnalyticsEvent, ChatSession, UserTrainingText


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def ensure_chat_session(
    db: AsyncSession,
    *,
    session_id: str,
    ab_group: str,
    model: str,
    backend: str,
) -> None:
    now = _utcnow()
    r = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    row = r.scalar_one_or_none()
    if row:
        row.last_seen_at = now
        row.ab_group = ab_group
        row.model = model
        row.backend = backend
    else:
        db.add(
            ChatSession(
                id=session_id,
                ab_group=ab_group,
                model=model,
                backend=backend,
                created_at=now,
                last_seen_at=now,
            )
        )


async def insert_metric_event(
    db: AsyncSession,
    *,
    session_id: str,
    metric_name: str,
    metric_value: float,
    ab_group: str,
    model: str,
) -> None:
    ts = _utcnow().isoformat()
    payload = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "ab_group": ab_group,
        "model": model,
        "timestamp": ts,
    }
    db.add(
        AnalyticsEvent(
            event_type="metric",
            session_id=session_id,
            payload=payload,
            created_at=_utcnow(),
        )
    )


async def insert_feedback_event(
    db: AsyncSession,
    *,
    session_id: str,
    message_index: int,
    feedback: str,
    ab_group: str,
    model: str,
) -> None:
    ts = _utcnow().isoformat()
    payload = {
        "message_index": message_index,
        "feedback": feedback,
        "ab_group": ab_group,
        "model": model,
        "timestamp": ts,
    }
    db.add(
        AnalyticsEvent(
            event_type="feedback",
            session_id=session_id,
            payload=payload,
            created_at=_utcnow(),
        )
    )


async def insert_training_text(
    db: AsyncSession,
    body: str,
    *,
    source: str = "save_to_training",
) -> None:
    db.add(
        UserTrainingText(
            body=body,
            source=source,
            created_at=_utcnow(),
        )
    )
