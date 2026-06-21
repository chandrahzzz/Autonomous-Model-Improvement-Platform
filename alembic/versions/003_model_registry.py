"""Model registry indexes and eval set table

Revision ID: 003
Revises: 002
Create Date: 2026-06-14
"""

from alembic import op

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Held-out evaluation set.
    # NOTE: this table must contain at least settings.min_eval_examples (default 50)
    # rows before any promotion decision is trusted. Postgres cannot express a
    # row-count CHECK, so the floor is enforced in code (eval_runner_node) and
    # seeded via scripts/seed_eval_set.py.
    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_set (
            id              SERIAL PRIMARY KEY,
            version         TEXT NOT NULL DEFAULT 'v1',
            question        TEXT NOT NULL,
            context         TEXT NOT NULL,
            ground_truth    TEXT NOT NULL,
            domain          TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_eval_set_version ON eval_set (version)")

    # Shadow traffic log
    op.execute("""
        CREATE TABLE IF NOT EXISTS shadow_logs (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            challenger_version  TEXT NOT NULL,
            prompt              TEXT NOT NULL,
            production_output   TEXT NOT NULL,
            challenger_output   TEXT NOT NULL,
            quality_delta       FLOAT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_shadow_version ON shadow_logs (challenger_version, created_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS shadow_logs CASCADE")
    op.execute("DROP TABLE IF EXISTS eval_set CASCADE")
