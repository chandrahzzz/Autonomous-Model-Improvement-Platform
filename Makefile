.PHONY: dev test migrate lint format typecheck topics seed clean

# ── Development ──────────────────────────────────────────────────────────────
dev:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dev-pipeline:
	uv run python -m src.graph.runner

infra-up:
	docker-compose up -d
	@echo "Waiting for services..."
	@sleep 10

infra-down:
	docker-compose down

infra-logs:
	docker-compose logs -f

# ── Database ──────────────────────────────────────────────────────────────────
migrate:
	uv run alembic upgrade head

migrate-down:
	uv run alembic downgrade -1

migrate-history:
	uv run alembic history

# ── Kafka ─────────────────────────────────────────────────────────────────────
topics:
	uv run python -c "from src.kafka.topics import create_topics; from src.config.settings import settings; create_topics(settings.kafka_bootstrap_servers)"

# ── Seeding ───────────────────────────────────────────────────────────────────
seed-eval:
	uv run python scripts/seed_eval_set.py

seed-baseline:
	uv run python scripts/seed_baseline.py

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy src

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-cov:
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# ── Audit ─────────────────────────────────────────────────────────────────────
verify-audit:
	uv run python scripts/verify_audit_chain.py

rollback:
	uv run python scripts/manual_rollback.py

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache

# ── Bootstrap ─────────────────────────────────────────────────────────────────
bootstrap: infra-up migrate topics seed-eval seed-baseline
	@echo "Pipeline bootstrapped and ready."
