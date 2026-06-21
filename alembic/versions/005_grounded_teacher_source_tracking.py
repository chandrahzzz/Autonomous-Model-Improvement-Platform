"""Grounded teacher: grounding_score + grounding_sources + retraction

Revision ID: 005
Revises: 004
Create Date: 2026-06-21
"""

from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # RAG-grounding metadata for teacher corrections.
    op.execute("ALTER TABLE training_examples ADD COLUMN IF NOT EXISTS grounding_score FLOAT")
    op.execute("ALTER TABLE training_examples ADD COLUMN IF NOT EXISTS grounding_sources TEXT[]")
    op.execute("ALTER TABLE training_examples ADD COLUMN IF NOT EXISTS retracted_at TIMESTAMPTZ")

    # Partial index: fast lookup of weakly-grounded examples for review/removal.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_te_grounding_score "
        "ON training_examples (grounding_score) WHERE grounding_score IS NOT NULL"
    )
    # GIN index so "which examples were grounded on source X" is fast (array containment).
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_te_grounding_sources "
        "ON training_examples USING GIN (grounding_sources)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_te_grounding_sources")
    op.execute("DROP INDEX IF EXISTS ix_te_grounding_score")
    op.execute("ALTER TABLE training_examples DROP COLUMN IF EXISTS retracted_at")
    op.execute("ALTER TABLE training_examples DROP COLUMN IF EXISTS grounding_sources")
    op.execute("ALTER TABLE training_examples DROP COLUMN IF EXISTS grounding_score")
