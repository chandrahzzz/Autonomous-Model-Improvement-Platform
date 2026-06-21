"""Hallucination grounding context, dataset versioning, calibration history

Revision ID: 004
Revises: 003
Create Date: 2026-06-19
"""

from alembic import op

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # #2 Hallucination grounding: store retrieved RAG context so the NLI detector
    # can check factual consistency against real context, not just the prompt.
    op.execute("ALTER TABLE llm_logs ADD COLUMN IF NOT EXISTS retrieved_context TEXT")

    # Dataset versioning: durable, reproducible pointer to the exact training set
    # used for a run.
    op.execute("ALTER TABLE training_runs ADD COLUMN IF NOT EXISTS dataset_uri TEXT")

    # Confidence calibration: append-only history of threshold suggestions and
    # (optionally) applied changes.
    op.execute("""
        CREATE TABLE IF NOT EXISTS calibration_history (
            id              SERIAL PRIMARY KEY,
            metric          TEXT NOT NULL,
            current_value   FLOAT NOT NULL,
            suggested_value FLOAT NOT NULL,
            signal          TEXT NOT NULL,
            applied         BOOLEAN NOT NULL DEFAULT FALSE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_calibration_metric ON calibration_history (metric)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS calibration_history CASCADE")
    op.execute("ALTER TABLE training_runs DROP COLUMN IF EXISTS dataset_uri")
    op.execute("ALTER TABLE llm_logs DROP COLUMN IF EXISTS retrieved_context")
