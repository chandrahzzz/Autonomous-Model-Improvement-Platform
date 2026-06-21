"""Audit trail row-level security

Revision ID: 002
Revises: 001
Create Date: 2026-06-14
"""

from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable row-level security on audit_trail (fail closed on UPDATE/DELETE)
    op.execute("ALTER TABLE audit_trail ENABLE ROW LEVEL SECURITY")
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pipeline_writer') THEN
                CREATE ROLE pipeline_writer;
            END IF;
        END
        $$
    """)
    op.execute("GRANT INSERT, SELECT ON audit_trail TO pipeline_writer")
    op.execute("GRANT USAGE, SELECT ON SEQUENCE audit_trail_id_seq TO pipeline_writer")
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE tablename='audit_trail' AND policyname='audit_insert_only'
            ) THEN
                CREATE POLICY audit_insert_only ON audit_trail
                    FOR INSERT TO pipeline_writer WITH CHECK (true);
            END IF;
        END
        $$
    """)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE tablename='audit_trail' AND policyname='audit_select_only'
            ) THEN
                CREATE POLICY audit_select_only ON audit_trail
                    FOR SELECT TO pipeline_writer USING (true);
            END IF;
        END
        $$
    """)


def downgrade() -> None:
    op.execute("ALTER TABLE audit_trail DISABLE ROW LEVEL SECURITY")
