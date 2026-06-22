"""Single-row pipeline_metrics table for lifetime counters (#L4)

Separates a durable lifetime cycle count from the per-session graph state, which
resets on restart and (via the 5-min Redis TTL) is unreliable as a lifetime total.

Revision ID: 011
Revises: 010
Create Date: 2026-06-22
"""

from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_metrics (
            id                       INT PRIMARY KEY DEFAULT 1,
            lifetime_cycles_completed BIGINT NOT NULL DEFAULT 0,
            updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT pipeline_metrics_singleton CHECK (id = 1)
        )
    """)
    op.execute("INSERT INTO pipeline_metrics (id) VALUES (1) ON CONFLICT (id) DO NOTHING")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pipeline_metrics")
