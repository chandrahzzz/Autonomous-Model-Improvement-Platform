"""Failure attribution: failure_attributions table (RFC-003)

Revision ID: 008
Revises: 007
Create Date: 2026-06-21

Note: training_examples.retracted_at already exists (added in migration 005),
so this migration only creates the failure_attributions table.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "failure_attributions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("log_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("llm_logs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("failure_classification_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("failure_classifications.id", ondelete="SET NULL"), nullable=True),
        sa.Column("model_version", sa.String, nullable=False),
        # training_runs.id is integer (autoincrement) → integer FK, not UUID.
        sa.Column("training_run_id", sa.Integer,
                  sa.ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("top_k_examples", postgresql.JSONB, nullable=False),
        sa.Column("total_candidates_scored", sa.Integer, nullable=False),
        sa.Column("backend_used", sa.String, nullable=False, server_default="embedding_cosine"),
        sa.Column("computation_ms", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_failure_attr_log_id", "failure_attributions", ["log_id"])
    op.create_index("ix_failure_attr_model_version", "failure_attributions", ["model_version"])
    op.create_index("ix_failure_attr_created_at", "failure_attributions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_failure_attr_created_at", table_name="failure_attributions")
    op.drop_index("ix_failure_attr_model_version", table_name="failure_attributions")
    op.drop_index("ix_failure_attr_log_id", table_name="failure_attributions")
    op.drop_table("failure_attributions")
