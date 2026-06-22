"""
Collects shadow traffic metrics over the 48h window.
Stores per-request quality deltas for statistical analysis.
"""

from datetime import datetime, timedelta
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.config.settings import settings

log = structlog.get_logger()


class ABCollector:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def record(
        self,
        challenger_version: str,
        prompt: str,
        production_output: str,
        challenger_output: str,
        quality_delta: float,
    ) -> None:
        await self._db.execute(
            text("""
                INSERT INTO shadow_logs
                    (challenger_version, prompt, production_output, challenger_output, quality_delta)
                VALUES (:version, :prompt, :prod, :chal, :delta)
            """),
            {
                "version": challenger_version,
                "prompt": prompt[:2000],
                "prod": production_output[:2000],
                "chal": challenger_output[:2000],
                "delta": quality_delta,
            },
        )

    async def cleanup_old(self, retention_days: int) -> int:
        """Delete shadow_logs rows older than retention_days (#S3). The table is
        append-only per shadow request; without this it grows unboundedly. The
        existing (challenger_version, created_at) index keeps collect_window fast,
        so this is purely about storage. Returns rows deleted."""
        result = await self._db.execute(
            text("DELETE FROM shadow_logs WHERE created_at < NOW() - make_interval(days => :days)"),
            {"days": retention_days},
        )
        return result.rowcount or 0

    async def collect_window(self, challenger_version: str) -> dict:
        """
        Collect all shadow log deltas for a challenger version.
        Returns stats needed for promotion gate decision.
        """
        since = datetime.utcnow() - timedelta(hours=settings.ab_min_hours)
        result = await self._db.execute(
            text("""
                SELECT quality_delta, created_at
                FROM shadow_logs
                WHERE challenger_version = :version
                  AND created_at >= :since
                ORDER BY created_at ASC
            """),
            {"version": challenger_version, "since": since},
        )
        rows = result.fetchall()

        deltas = [row.quality_delta for row in rows if row.quality_delta is not None]
        n = len(rows)
        elapsed_h = (datetime.utcnow() - since).total_seconds() / 3600

        return {
            "n_requests": n,
            "elapsed_hours": elapsed_h,
            "quality_deltas": deltas,
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "ready": (
                n >= settings.ab_min_requests
                and elapsed_h >= settings.ab_min_hours
            ),
        }
