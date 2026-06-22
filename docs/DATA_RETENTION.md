# Data retention & table growth

## Correction: `llm_logs` is NOT partitioned

A review flagged that `llm_logs` is "partitioned quarterly with no automated
partition creation," warning that inserts would fail at a quarter boundary.

**This is not the case.** Migration `001_initial_schema.py` creates `llm_logs` as
a **plain table** (`CREATE TABLE llm_logs (...)`, no `PARTITION BY`). There is no
partitioning anywhere in the schema (`grep -r PARTITION alembic/versions` → none).
So there is **no partition boundary that can break inserts** — the failure mode
described cannot occur.

What *is* real is unbounded growth over time. We address that as follows.

## `shadow_logs` — retention cleanup (implemented, #S3)

`shadow_logs` is append-only (one row per shadowed request). It already has a
`(challenger_version, created_at DESC)` index that keeps `ABCollector.collect_window`
fast regardless of table size, so this is a storage concern, not a query one.

- `ABCollector.cleanup_old(retention_days)` deletes rows older than the window.
- The runner calls it every `SHADOW_LOGS_CLEANUP_INTERVAL_CYCLES` cycles, pruning
  rows older than `SHADOW_LOGS_RETENTION_DAYS` (default 30). Metric: `shadow_logs_pruned_total`.

## `llm_logs` — recommended retention

`llm_logs` feeds drift baselines and the replay buffer, so deletion must be
deliberate. Retention is **not** auto-enabled. Recommended production setup:

1. **Time-based partitioning** (if/when volume warrants): convert `llm_logs` to
   `PARTITION BY RANGE (created_at)` and use **pg_partman** for automatic
   partition creation + retention drop. This is the scalable path.
2. **Or** a scheduled `DELETE FROM llm_logs WHERE created_at < NOW() - INTERVAL '90 days'`
   via cron / pg_cron, sized so it never deletes data still inside the drift
   baseline / replay window.

Either way, add a Prometheus alert on table size / age before enabling deletion.
