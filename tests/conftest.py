"""
Pytest fixtures for unit and integration tests.
Uses testcontainers for PostgreSQL, Redis, Kafka.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock


# ── Event loop ────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Mock settings ─────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Override settings for tests — no real external services required."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setenv("MODAL_TOKEN_ID", "test")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test")
    monkeypatch.setenv("VAULT_TOKEN", "root")


# ── Mock DB session ───────────────────────────────────────────────────────────
@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.close = AsyncMock()
    return db


# ── Sample LLM events ─────────────────────────────────────────────────────────
@pytest.fixture
def sample_llm_events():
    return [
        {
            "id": f"event-{i}",
            "session_id": f"sess-{i}",
            "model_version": "v7",
            "prompt": f"What is the capital of country {i}?",
            "completion": f"The capital of country {i} is City{i}.",
            "prompt_tokens": 15,
            "completion_tokens": 12,
            "latency_ms": 250,
            "finish_reason": "stop",
            "cost_usd": 0.001,
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_failure_events():
    from src.detection.failure_classifier import FailureEvent
    return [
        FailureEvent(
            llm_log_id=f"log-{i}",
            prompt=f"Question {i}",
            completion="I cannot help with that request.",
            failure_type="refusal_creep",
            score=0.92,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_eval_set():
    return [
        {
            "question": "What is the capital of France?",
            "context": "France is a Western European country. Its capital city is Paris.",
            "ground_truth": "Paris",
        },
        {
            "question": "Who wrote Hamlet?",
            "context": "Hamlet is a tragedy written by William Shakespeare in approximately 1600.",
            "ground_truth": "William Shakespeare",
        },
    ]
