"""Predictive drift early warning: drift_trend_history table (RFC-001)

Revision ID: 006
Revises: 005
Create Date: 2026-06-21
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drift_trend_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("cycle_id", sa.String, nullable=True),
        sa.Column("model_version", sa.String, nullable=True),
        sa.Column("current_score", sa.Float, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("slope_per_cycle", sa.Float, nullable=False),
        sa.Column("r_squared", sa.Float, nullable=False),
        sa.Column("predicted_trigger_hours", sa.Float, nullable=True),
        sa.Column("window_size", sa.Integer, nullable=False),
        sa.Column("trend_direction", sa.String, nullable=False),
        sa.Column("is_alarming", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("alert_sent", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_drift_trend_created_at", "drift_trend_history", ["created_at"])
    op.create_index("ix_drift_trend_is_alarming", "drift_trend_history", ["is_alarming"])


def downgrade() -> None:
    op.drop_index("ix_drift_trend_is_alarming", table_name="drift_trend_history")
    op.drop_index("ix_drift_trend_created_at", table_name="drift_trend_history")
    op.drop_table("drift_trend_history")
