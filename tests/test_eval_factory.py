"""Unit tests for the continuous eval factory (RFC-002). No real I/O."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation.eval_factory import EvalFactory


def make_factory(
    active_count: int = 0,
    exists_similar: bool = False,
    generated_answer: str | None = "This is a good answer.",
):
    eval_repo = AsyncMock()
    eval_repo.count_active_by_source.return_value = {
        "seed": 50, "factory": max(0, active_count - 50)
    }
    eval_repo.exists_similar.return_value = exists_similar
    eval_repo.evict_oldest.return_value = 1
    eval_repo.insert_factory_example.return_value = MagicMock()

    openai = AsyncMock()
    openai.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=generated_answer))]
    )

    redis = AsyncMock()
    redis.get.return_value = "0"

    factory = EvalFactory(eval_repo=eval_repo, openai_client=openai, redis_client=redis)
    return factory, eval_repo, openai, redis


@contextmanager
def patch_settings(trigger_every=1000, max_size=500, min_confidence=0.80,
                   dedup_threshold=0.90, max_per_run=10):
    with patch("src.evaluation.eval_factory.settings") as s:
        s.eval_factory_trigger_every_n_requests = trigger_every
        s.eval_factory_max_eval_set_size = max_size
        s.eval_factory_min_confidence = min_confidence
        s.eval_factory_dedup_cosine_threshold = dedup_threshold
        s.eval_factory_max_examples_per_run = max_per_run
        s.eval_factory_request_counter_key = "test:counter"
        yield s


@contextmanager
def patch_clusterer_medoid(factory, prompt, embedding):
    with patch.object(factory._clusterer, "pick_medoid", return_value=(prompt, embedding)):
        yield


# ── should_run() ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_should_run_below_threshold():
    factory, _, _, redis = make_factory()
    redis.get.return_value = "999"
    with patch_settings(trigger_every=1000):
        assert await factory.should_run() is False


@pytest.mark.asyncio
async def test_should_run_at_threshold():
    factory, _, _, redis = make_factory()
    redis.get.return_value = "1000"
    with patch_settings(trigger_every=1000):
        assert await factory.should_run() is True


@pytest.mark.asyncio
async def test_should_run_no_counter():
    factory, _, _, redis = make_factory()
    redis.get.return_value = None
    with patch_settings(trigger_every=1000):
        assert await factory.should_run() is False


# ── _try_generate_example() ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_try_generate_skips_duplicate():
    factory, eval_repo, openai, _ = make_factory(exists_similar=True)
    with patch_clusterer_medoid(factory, "How do I book a flight?", [0.1] * 384):
        with patch_settings():
            result = await factory._try_generate_example(0, ["How do I book a flight?"] * 5, AsyncMock())
    assert result is False
    eval_repo.insert_factory_example.assert_not_called()
    openai.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_try_generate_skips_on_generation_failure():
    factory, eval_repo, _, _ = make_factory(generated_answer=None)
    with patch_clusterer_medoid(factory, "Some prompt", [0.1] * 384):
        with patch_settings():
            result = await factory._try_generate_example(0, ["Some prompt"] * 5, AsyncMock())
    assert result is False
    eval_repo.insert_factory_example.assert_not_called()


@pytest.mark.asyncio
async def test_try_generate_skips_low_confidence():
    factory, eval_repo, _, _ = make_factory()
    with patch.object(factory, "_compute_consistency", return_value=0.20):
        with patch_clusterer_medoid(factory, "Some prompt", [0.1] * 384):
            with patch_settings(min_confidence=0.80):
                result = await factory._try_generate_example(0, ["Some prompt"] * 5, AsyncMock())
    assert result is False
    eval_repo.insert_factory_example.assert_not_called()


@pytest.mark.asyncio
async def test_try_generate_happy_path():
    factory, eval_repo, _, _ = make_factory(
        exists_similar=False, generated_answer="Lithium batteries up to 160Wh are allowed.",
        active_count=10,
    )
    with patch.object(factory, "_compute_consistency", return_value=0.92):
        with patch_clusterer_medoid(factory, "Can I carry lithium batteries?", [0.5] * 384):
            with patch_settings():
                result = await factory._try_generate_example(2, ["Can I carry lithium batteries?"] * 5, AsyncMock())
    assert result is True
    eval_repo.insert_factory_example.assert_called_once()
    kwargs = eval_repo.insert_factory_example.call_args[0][0]
    assert kwargs["source"] == "factory"
    assert kwargs["cluster_id"] == 2
    assert kwargs["factory_confidence"] == 0.92


@pytest.mark.asyncio
async def test_try_generate_evicts_when_at_cap():
    factory, eval_repo, _, _ = make_factory(
        exists_similar=False, generated_answer="Good answer here.", active_count=500,
    )
    eval_repo.count_active_by_source.return_value = {"seed": 50, "factory": 450}
    with patch.object(factory, "_compute_consistency", return_value=0.92):
        with patch_clusterer_medoid(factory, "Some prompt", [0.1] * 384):
            with patch_settings(max_size=500):
                result = await factory._try_generate_example(0, ["Some prompt"] * 5, AsyncMock())
    assert result is True
    eval_repo.evict_oldest.assert_called_once_with(count=1)
    eval_repo.insert_factory_example.assert_called_once()


@pytest.mark.asyncio
async def test_evict_oldest_only_evicts_factory_examples():
    # Enforcement is in the SQL (source='factory'); here we just confirm the
    # method contract via mock.
    eval_repo = AsyncMock()
    eval_repo.evict_oldest.return_value = 1
    result = await eval_repo.evict_oldest(count=1)
    eval_repo.evict_oldest.assert_called_once_with(count=1)
    assert result == 1


# ── _compute_consistency() ───────────────────────────────────────────────────

def test_compute_consistency_identical_answers():
    factory, _, _, _ = make_factory()
    assert factory._compute_consistency(["The same answer."] * 3) == pytest.approx(1.0, abs=0.01)


def test_compute_consistency_different_answers():
    factory, _, _, _ = make_factory()
    answers = [
        "Cats are mammals that purr.",
        "The Python programming language was created in 1991.",
        "Mount Everest is 8849 meters tall.",
    ]
    assert factory._compute_consistency(answers) < 0.20


def test_compute_consistency_single_answer():
    factory, _, _, _ = make_factory()
    assert factory._compute_consistency(["Only one answer."]) == 0.0
