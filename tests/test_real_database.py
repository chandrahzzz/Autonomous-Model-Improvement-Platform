"""
Integration tests that run against a REAL PostgreSQL.

Everything under `tests/integration/` is mocked — `conftest`'s `db` is an
`AsyncMock`, so `db.execute()` swallows whatever string it is handed. That means
the suite has never verified that the migrations apply, that the audit trail is
actually immutable, or that the hand-written `text()` SQL in the repositories is
even valid SQL. Those are exactly the claims that matter most.

Run with infrastructure up:

    docker compose up -d postgres
    uv run alembic -c alembic/alembic.ini upgrade head
    uv run pytest tests/test_real_database.py -m realdb

Auto-skips when Postgres is unreachable, so the default suite stays green on a
laptop with nothing running.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.config.settings import settings

pytestmark = [pytest.mark.realdb, pytest.mark.asyncio]

CONNECT_TIMEOUT_SECONDS = 3


async def _engine_or_skip():
    engine = create_async_engine(
        settings.database_url,
        connect_args={"timeout": CONNECT_TIMEOUT_SECONDS},
        poolclass=None,
    )
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL unreachable ({type(exc).__name__}) — start it with "
                    f"`docker compose up -d postgres`")
    return engine


@pytest_asyncio.fixture
async def engine():
    eng = await _engine_or_skip()
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session(engine):
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
        await s.rollback()


# ── Migrations actually applied ──────────────────────────────────────────────
EXPECTED_TABLES = [
    "llm_logs",
    "training_examples",
    "training_runs",
    "model_versions",
    "audit_trail",
    "eval_set",
    "eval_runs",
    "shadow_logs",
    "drift_baselines",
    "failure_classifications",
    "knowledge_documents",
    "pipeline_metrics",
]


@pytest.mark.parametrize("table", EXPECTED_TABLES)
async def test_migrations_created_every_table(session, table):
    found = await session.execute(
        text("SELECT to_regclass(:t)"), {"t": f"public.{table}"}
    )
    assert found.scalar() is not None, (
        f"table {table!r} missing — run `alembic -c alembic/alembic.ini upgrade head`"
    )


async def test_alembic_is_at_head(session):
    result = await session.execute(text("SELECT version_num FROM alembic_version"))
    version = result.scalar()
    assert version is not None
    assert version == "013", f"migrations not at head (at {version!r}, expected '013')"


# ── The audit trail is genuinely append-only ─────────────────────────────────
# This is the project's central integrity claim and nothing verified it: RLS is
# declared in migration 002 and was never exercised.
async def _insert_audit_row(session) -> int:
    result = await session.execute(
        text("""
            INSERT INTO audit_trail (event_type, decision, rationale, state_snapshot, hmac_sha256)
            VALUES ('test_event', 'integrity probe', '{}'::jsonb, '{}'::jsonb, :sig)
            RETURNING id
        """),
        {"sig": "deadbeef"},
    )
    row_id = result.scalar()
    await session.commit()
    return row_id


async def test_audit_row_can_be_inserted_and_read(session):
    row_id = await _insert_audit_row(session)
    got = await session.execute(
        text("SELECT decision FROM audit_trail WHERE id = :id"), {"id": row_id}
    )
    assert got.scalar() == "integrity probe"


async def test_audit_trail_rls_is_enabled_and_forced(session):
    """RLS must be ENABLED *and* FORCED.

    Enabled alone is not enough: a table owner bypasses RLS unless the table is
    FORCE ROW LEVEL SECURITY. Without the force flag the policies below are
    decorative for whoever owns the table (migration 013).
    """
    row = await session.execute(
        text(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE relname = 'audit_trail'"
        )
    )
    enabled, forced = row.one()
    assert enabled is True, "row-level security is NOT enabled on audit_trail"
    assert forced is True, (
        "RLS is enabled but not FORCED — the table owner bypasses every policy"
    )


async def test_audit_writer_role_can_login_and_is_not_privileged(session):
    """The role the policies target must be usable, and must not hold a bypass."""
    row = await session.execute(
        text(
            "SELECT rolcanlogin, rolsuper, rolbypassrls FROM pg_roles "
            "WHERE rolname = 'pipeline_writer'"
        )
    )
    result = row.one_or_none()
    assert result is not None, "pipeline_writer role does not exist"
    canlogin, is_super, bypass = result
    assert canlogin is True, "pipeline_writer cannot log in — the policies are unreachable"
    assert is_super is False, "pipeline_writer is a superuser; RLS cannot constrain it"
    assert bypass is False, "pipeline_writer has BYPASSRLS; RLS cannot constrain it"


async def test_audit_trail_policies_allow_no_mutation(session):
    """The INSERT/SELECT-only policies must be present."""

    policies = await session.execute(
        text("SELECT cmd FROM pg_policies WHERE tablename = 'audit_trail'")
    )
    cmds = {row[0].upper() for row in policies.fetchall()}
    assert cmds, "no RLS policies on audit_trail"
    assert "UPDATE" not in cmds, "an UPDATE policy exists — the trail is mutable"
    assert "DELETE" not in cmds, "a DELETE policy exists — the trail is erasable"


async def test_audit_trail_is_immutable_for_the_pipeline_role(engine):
    """Under `pipeline_writer`, UPDATE and DELETE must not alter the trail.

    Superusers bypass RLS, so this connects as the restricted role the
    application is meant to use. Skips (loudly) if that role has no login.
    """
    row_id = None
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as setup:
        row_id = await _insert_audit_row(setup)

    # Swap only the username; migration 013 gives pipeline_writer the password
    # from AUDIT_WRITER_PASSWORD (falling back to POSTGRES_PASSWORD), which on a
    # dev box is the same one already in DATABASE_URL.
    import os
    import re

    password = os.getenv("AUDIT_WRITER_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    if password:
        writer_url = re.sub(
            r"://[^:]+:[^@]+@", f"://pipeline_writer:{password}@", settings.database_url
        )
    else:
        writer_url = re.sub(r"://[^:]+:", "://pipeline_writer:", settings.database_url, count=1)

    writer_engine = create_async_engine(
        writer_url, connect_args={"timeout": CONNECT_TIMEOUT_SECONDS}
    )
    try:
        async with writer_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        await writer_engine.dispose()
        pytest.skip(f"pipeline_writer role cannot log in ({type(exc).__name__}); "
                    f"grant it a password to exercise RLS end to end")

    try:
        writer_maker = async_sessionmaker(writer_engine, expire_on_commit=False)

        # The role must still be able to APPEND — an audit trail nothing can
        # write to is useless.
        async with writer_maker() as w:
            await w.execute(
                text("""
                    INSERT INTO audit_trail
                        (event_type, decision, rationale, state_snapshot, hmac_sha256)
                    VALUES ('rls_probe', 'append works', '{}'::jsonb, '{}'::jsonb, 'sig')
                """)
            )
            await w.commit()

        # ...but never rewrite or erase it.
        async with writer_maker() as w:
            with pytest.raises(Exception):
                await w.execute(
                    text("UPDATE audit_trail SET decision = 'tampered' WHERE id = :id"),
                    {"id": row_id},
                )
                await w.commit()
            await w.rollback()

        async with writer_maker() as w:
            with pytest.raises(Exception):
                await w.execute(
                    text("DELETE FROM audit_trail WHERE id = :id"), {"id": row_id}
                )
                await w.commit()
            await w.rollback()

        # Whether the statements errored or silently affected zero rows, the
        # stored record must be unchanged and still present.
        async with maker() as check:
            result = await check.execute(
                text("SELECT decision FROM audit_trail WHERE id = :id"), {"id": row_id}
            )
            surviving = result.scalar()
        assert surviving == "integrity probe", "audit row was mutated or deleted"
    finally:
        await writer_engine.dispose()


# ── Hand-written repository SQL is valid against a real server ───────────────
# Every one of these is raw text() that a mocked session would happily accept
# regardless of syntax or column names.
async def test_known_good_sample_sql_is_valid(session):
    from src.db.repositories.llm_logs import LLMLogRepository

    rows = await LLMLogRepository(session).get_known_good_sample(limit=5)
    assert isinstance(rows, list)


async def test_known_good_candidates_sql_is_valid(session):
    from src.db.repositories.llm_logs import LLMLogRepository

    rows = await LLMLogRepository(session).get_known_good_candidates(
        limit=5, model_version="v7"
    )
    assert isinstance(rows, list)


async def test_shadow_window_roundtrip(session):
    """Insert shadow observations and read them back through collect_window —
    the path that decides promotion."""
    from src.shadow.ab_collector import ABCollector

    version = f"itest-{uuid.uuid4().hex[:8]}"
    collector = ABCollector(session)
    for i in range(5):
        await collector.record(
            challenger_version=version,
            prompt=f"prompt {i}",
            production_output="prod",
            challenger_output="chal",
            quality_delta=0.1,
        )
    await session.commit()

    started = datetime.now(timezone.utc) - timedelta(hours=2)
    data = await collector.collect_window(version, started_at=started)

    assert data["n_requests"] == 5
    assert data["quality_deltas"] == [pytest.approx(0.1)] * 5
    assert 1.5 < data["elapsed_hours"] < 2.5


async def test_shadow_logs_cleanup_sql_is_valid(session):
    from src.shadow.ab_collector import ABCollector

    deleted = await ABCollector(session).cleanup_old(retention_days=36500)
    assert deleted == 0  # nothing is that old; the point is the SQL parses


async def test_lifetime_cycle_counter_sql_is_valid(session):
    """The runner's UPDATE ... RETURNING against pipeline_metrics."""
    result = await session.execute(
        text(
            "UPDATE pipeline_metrics SET lifetime_cycles_completed = "
            "lifetime_cycles_completed + 1, updated_at = NOW() WHERE id = 1 "
            "RETURNING lifetime_cycles_completed"
        )
    )
    total = result.scalar()
    await session.rollback()
    assert total is not None, "pipeline_metrics row id=1 is missing (seeded by migration 011)"


async def test_next_version_tag_against_real_registry(session):
    """Confirms the rollback-collision fix works against real rows."""
    from src.db.repositories.model_versions import ModelRepository

    tag = await ModelRepository(session).next_version_tag()
    assert tag.startswith("v")
    existing = await session.execute(
        text("SELECT 1 FROM model_versions WHERE version_tag = :t"), {"t": tag}
    )
    assert existing.scalar() is None, f"next_version_tag returned an existing tag {tag!r}"
