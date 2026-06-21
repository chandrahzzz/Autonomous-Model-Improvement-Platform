"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-06-14
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS llm_logs (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id      TEXT NOT NULL,
            user_cohort     TEXT,
            model_version   TEXT NOT NULL,
            prompt          TEXT NOT NULL,
            completion      TEXT NOT NULL,
            prompt_tokens   INT NOT NULL,
            completion_tokens INT NOT NULL,
            latency_ms      INT NOT NULL,
            finish_reason   TEXT NOT NULL,
            cost_usd        NUMERIC(12, 8) NOT NULL,
            embedding_hash  TEXT,
            metadata        JSONB DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_logs_created_at ON llm_logs (created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_logs_model_version ON llm_logs (model_version, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_llm_logs_session ON llm_logs (session_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS failure_classifications (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            llm_log_id      UUID NOT NULL REFERENCES llm_logs(id) ON DELETE CASCADE,
            failure_type    TEXT NOT NULL CHECK (failure_type IN (
                                'hallucination', 'semantic_drift',
                                'refusal_creep', 'format_regression')),
            score           FLOAT NOT NULL,
            cluster_id      INT,
            cluster_label   TEXT,
            metadata        JSONB DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_failure_type ON failure_classifications (failure_type, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_failure_cluster ON failure_classifications (cluster_id)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS training_examples (
            id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            failure_id           UUID REFERENCES failure_classifications(id),
            llm_log_id           UUID REFERENCES llm_logs(id),
            prompt               TEXT NOT NULL,
            bad_completion       TEXT NOT NULL,
            corrected_completion TEXT NOT NULL,
            failure_type         TEXT NOT NULL,
            cluster_id           INT,
            teacher_model        TEXT NOT NULL,
            teacher_confidence   FLOAT NOT NULL,
            pii_scrubbed         BOOLEAN NOT NULL DEFAULT FALSE,
            dedup_hash           TEXT NOT NULL,
            quality_score        FLOAT NOT NULL,
            included_in_run      INT,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_training_dedup ON training_examples (dedup_hash)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_training_run ON training_examples (included_in_run)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_training_failure_type ON training_examples (failure_type)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            id                SERIAL PRIMARY KEY,
            version_tag       TEXT NOT NULL UNIQUE,
            base_model        TEXT NOT NULL,
            lora_weights_path TEXT,
            is_production     BOOLEAN NOT NULL DEFAULT FALSE,
            is_archived       BOOLEAN NOT NULL DEFAULT FALSE,
            training_run_id   INT,
            promoted_at       TIMESTAMPTZ,
            rolled_back_at    TIMESTAMPTZ,
            metadata          JSONB DEFAULT '{}',
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_production ON model_versions (is_production) WHERE is_production = TRUE")

    op.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id              SERIAL PRIMARY KEY,
            version_tag     TEXT NOT NULL,
            modal_job_id    TEXT,
            status          TEXT NOT NULL DEFAULT 'submitted'
                            CHECK (status IN ('submitted','running','completed','failed','cancelled')),
            dataset_size    INT NOT NULL,
            dataset_path    TEXT,
            lora_config     JSONB NOT NULL,
            final_loss      FLOAT,
            wandb_run_id    TEXT,
            wandb_run_url   TEXT,
            error_message   TEXT,
            started_at      TIMESTAMPTZ,
            completed_at    TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id                  SERIAL PRIMARY KEY,
            training_run_id     INT REFERENCES training_runs(id),
            version_tag         TEXT NOT NULL,
            eval_type           TEXT NOT NULL CHECK (eval_type IN ('ragas','safety','ab_test')),
            status              TEXT NOT NULL DEFAULT 'running'
                                CHECK (status IN ('running','passed','failed','error')),
            faithfulness        FLOAT,
            answer_relevancy    FLOAT,
            context_recall      FLOAT,
            safety_score        FLOAT,
            ab_quality_delta    FLOAT,
            ab_pvalue           FLOAT,
            ab_cohens_d         FLOAT,
            ab_n_requests       INT,
            gate_passed         BOOLEAN,
            rationale           JSONB DEFAULT '{}',
            eval_set_version    TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            completed_at        TIMESTAMPTZ
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_eval_version ON eval_runs (version_tag, eval_type)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS audit_trail (
            id                   BIGSERIAL PRIMARY KEY,
            event_type           TEXT NOT NULL,
            decision             TEXT NOT NULL,
            rationale            JSONB NOT NULL,
            state_snapshot       JSONB NOT NULL,
            model_version_before TEXT,
            model_version_after  TEXT,
            operator             TEXT NOT NULL DEFAULT 'autonomous_pipeline',
            hmac_sha256          TEXT NOT NULL,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_trail (event_type, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_trail (created_at DESC)")

    op.execute("""
        CREATE TABLE IF NOT EXISTS drift_baselines (
            id              SERIAL PRIMARY KEY,
            model_version   TEXT NOT NULL,
            sample_size     INT NOT NULL,
            centroid        JSONB NOT NULL,
            covariance_inv  JSONB NOT NULL,
            computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_active       BOOLEAN NOT NULL DEFAULT FALSE
        )
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_baseline_active ON drift_baselines (is_active) WHERE is_active = TRUE")

    # Seed the initial model version "v7" as production baseline
    op.execute("""
        INSERT INTO model_versions (version_tag, base_model, is_production, metadata)
        VALUES ('v7', 'meta-llama/Meta-Llama-3-8B-Instruct', TRUE, '{"seeded": true}')
        ON CONFLICT (version_tag) DO NOTHING
    """)


def downgrade() -> None:
    for table in ["drift_baselines", "audit_trail", "eval_runs", "training_runs",
                  "model_versions", "training_examples", "failure_classifications", "llm_logs"]:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
