"""Record replay-buffer version distribution on training runs (#T2)

Adds training_runs.replay_distribution so each run audits which production model
versions its known-good replay examples came from (recency-weighted sampling).

Revision ID: 010
Revises: 009
Create Date: 2026-06-22
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "training_runs",
        sa.Column("replay_distribution", postgresql.JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("training_runs", "replay_distribution")
