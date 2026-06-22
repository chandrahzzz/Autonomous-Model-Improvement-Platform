"""Nullable tenant_id scaffold on core tables (#I4)

The system is intentionally single-tenant today. Adding a nullable, unenforced
tenant_id now makes an eventual multi-tenancy migration ADDITIVE (backfill +
add indexes/constraints) rather than a destructive rewrite of every table. No
behaviour changes: nothing reads or writes these columns yet.

Revision ID: 012
Revises: 011
Create Date: 2026-06-22
"""

from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None

_TABLES = [
    "llm_logs", "failure_classifications", "training_examples", "model_versions",
    "training_runs", "eval_runs", "audit_trail", "shadow_logs",
]


def upgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS tenant_id UUID")


def downgrade() -> None:
    for table in _TABLES:
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS tenant_id")
