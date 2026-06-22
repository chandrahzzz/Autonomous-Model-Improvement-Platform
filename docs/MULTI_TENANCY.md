# Multi-tenancy migration path (#I4)

The system is **intentionally single-tenant** today (see README). Every query,
index, and audit record is tenant-unscoped. If multi-tenancy is ever required,
retrofitting a `tenant_id` onto every table after the fact is a large, destructive
migration.

## What we did now (zero behaviour change)

Migration `012_tenant_scaffold.py` adds a **nullable, unenforced `tenant_id UUID`**
column to the core tables (`llm_logs`, `failure_classifications`,
`training_examples`, `model_versions`, `training_runs`, `eval_runs`, `audit_trail`,
`shadow_logs`). Nothing reads or writes it yet — it is purely structural so the
eventual migration is **additive**.

## Eventual multi-tenancy migration (when needed)

1. Backfill `tenant_id` for existing rows (e.g. a single default tenant).
2. `ALTER COLUMN tenant_id SET NOT NULL` once backfilled.
3. Add composite indexes `(tenant_id, created_at)` / `(tenant_id, <existing keys>)`.
4. Add `tenant_id` to repository method signatures and a `WHERE tenant_id = :tid`
   filter (or Postgres Row-Level Security policies) — additive, not a rewrite.
5. Scope the audit trail and HMAC chain per tenant.

Because the column already exists everywhere, none of the above requires a
table rewrite or a backward-incompatible change.
