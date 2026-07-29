"""Make the audit trail genuinely immutable.

Migration 002 enabled RLS on `audit_trail` and created INSERT/SELECT-only
policies for `pipeline_writer`. Verified against a real PostgreSQL, that
guarantee was never in force:

  * `audit_trail` is owned by `pipeline`, and a table owner BYPASSES RLS unless
    the table is marked FORCE ROW LEVEL SECURITY (it was not:
    `relforcerowsecurity = f`).
  * The application connects as `pipeline`, which is `rolsuper = t` and
    `rolbypassrls = t` — RLS cannot constrain it under any configuration.
  * `pipeline_writer`, the role the policies target, had `rolcanlogin = f`, so
    nothing could ever connect as it.

Running as the application's own user, `UPDATE audit_trail` and
`DELETE FROM audit_trail` both succeeded.

This migration makes the restricted role usable and the table tamper-resistant:
  1. `pipeline_writer` gets LOGIN plus a password (from AUDIT_WRITER_PASSWORD,
     falling back to POSTGRES_PASSWORD so a dev box keeps working).
  2. It gets full DML on the application tables, but only INSERT/SELECT on
     `audit_trail` — enforced at the GRANT level as well as by RLS, so the
     restriction holds even if a policy is later changed.
  3. `audit_trail` is set to FORCE ROW LEVEL SECURITY, so ownership alone no
     longer confers a bypass.

REQUIRED FINAL STEP — this migration alone changes nothing for a running
system. The application must actually connect as the restricted role:

    DATABASE_URL=postgresql+asyncpg://pipeline_writer:<password>@host:5432/finetuning_pipeline

While DATABASE_URL still points at a superuser, the audit trail remains mutable
no matter what policies exist.

Revision ID: 013
Revises: 012
"""

import os

from alembic import op

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None

AUDIT_TABLE = "audit_trail"


def _writer_password() -> str:
    return (
        os.getenv("AUDIT_WRITER_PASSWORD")
        or os.getenv("POSTGRES_PASSWORD")
        or "password"
    )


def upgrade() -> None:
    password = _writer_password().replace("'", "''")

    # 1. Make the restricted role connectable.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pipeline_writer') THEN
                CREATE ROLE pipeline_writer;
            END IF;
        END $$;
        """
    )
    op.execute(f"ALTER ROLE pipeline_writer WITH LOGIN PASSWORD '{password}'")
    # Defence in depth: never let this role be granted an RLS bypass.
    op.execute("ALTER ROLE pipeline_writer WITH NOBYPASSRLS NOSUPERUSER")

    # 2. Privileges: full DML on the application tables...
    op.execute("GRANT USAGE ON SCHEMA public TO pipeline_writer")
    op.execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public "
        "TO pipeline_writer"
    )
    op.execute(
        "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO pipeline_writer"
    )
    # ...but the audit trail is append-only, at the GRANT level too.
    op.execute(f"REVOKE UPDATE, DELETE, TRUNCATE ON {AUDIT_TABLE} FROM pipeline_writer")

    # Future tables created by the owner must be reachable by the app role.
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO pipeline_writer"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        "GRANT USAGE, SELECT ON SEQUENCES TO pipeline_writer"
    )

    # 3. Ownership no longer grants an RLS bypass.
    op.execute(f"ALTER TABLE {AUDIT_TABLE} FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute(f"ALTER TABLE {AUDIT_TABLE} NO FORCE ROW LEVEL SECURITY")
    op.execute(f"GRANT UPDATE, DELETE ON {AUDIT_TABLE} TO pipeline_writer")
    op.execute("ALTER ROLE pipeline_writer WITH NOLOGIN")
