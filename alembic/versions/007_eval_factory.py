"""Continuous eval factory: living benchmark columns on eval_set (RFC-002)

Revision ID: 007
Revises: 006
Create Date: 2026-06-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("eval_set", sa.Column("source", sa.String, server_default="seed", nullable=False))
    op.add_column("eval_set", sa.Column("cluster_id", sa.Integer, nullable=True))
    op.add_column("eval_set", sa.Column("cluster_label", sa.String, nullable=True))
    op.add_column("eval_set", sa.Column("factory_confidence", sa.Float, nullable=True))
    op.add_column("eval_set", sa.Column("access_count", sa.Integer, server_default="0", nullable=False))
    op.add_column("eval_set", sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("eval_set", sa.Column("evicted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("eval_set", sa.Column("embedding", postgresql.JSONB, nullable=True))

    op.create_index("ix_eval_set_source_evicted", "eval_set", ["source", "evicted_at"])
    op.create_index("ix_eval_set_last_accessed", "eval_set", ["last_accessed_at"])


def downgrade() -> None:
    op.drop_index("ix_eval_set_last_accessed", table_name="eval_set")
    op.drop_index("ix_eval_set_source_evicted", table_name="eval_set")
    for col in (
        "embedding", "evicted_at", "last_accessed_at", "access_count",
        "factory_confidence", "cluster_label", "cluster_id", "source",
    ):
        op.drop_column("eval_set", col)
