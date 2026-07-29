"""
Collects shadow traffic metrics over the 48h window.
Stores per-request quality deltas for statistical analysis.
"""

from datetime import datetime, timezone
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

    async def collect_window(
        self, challenger_version: str, started_at: datetime | None = None
    ) -> dict:
        """
        Collect all shadow log deltas for a challenger version.
        Returns stats needed for promotion gate decision.

        ``started_at`` is when the shadow test began (from the router's Redis
        stamp). Elapsed time MUST be measured from it: the previous version
        computed ``now - (now - ab_min_hours)``, which is always exactly
        ``ab_min_hours``, so the "48h window" gate was a tautology that always
        passed. When no stamp exists we fall back to the oldest recorded sample,
        and to 0.0 when nothing has been recorded at all.

        Every row for the challenger counts — the version tag is unique per
        shadow test, so a lookback filter would only discard valid early samples.
        """
        result = await self._db.execute(
            text("""
                SELECT quality_delta, created_at
                FROM shadow_logs
                WHERE challenger_version = :version
                ORDER BY created_at ASC
            """),
            {"version": challenger_version},
        )
        rows = result.fetchall()

        deltas = [row.quality_delta for row in rows if row.quality_delta is not None]
        n = len(rows)

        window_start = started_at
        if window_start is None and rows:
            window_start = rows[0].created_at
        elapsed_h = self._hours_since(window_start)

        ready = n >= settings.ab_min_requests and elapsed_h >= settings.ab_min_hours
        # A challenger that never reaches the request floor must not hold the
        # graph in ab_test_node forever; force a decision past the ceiling.
        timed_out = (not ready) and elapsed_h >= settings.ab_max_wait_hours

        return {
            "n_requests": n,
            "elapsed_hours": elapsed_h,
            "quality_deltas": deltas,
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "ready": ready,
            "timed_out": timed_out,
        }

    @staticmethod
    def _hours_since(start: datetime | None) -> float:
        """Hours between ``start`` and now, tolerating naive/aware mixing (rows
        come back naive-UTC from asyncpg, the Redis stamp is tz-aware)."""
        if start is None:
            return 0.0
        now = datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        return max(0.0, (now - start).total_seconds() / 3600)
