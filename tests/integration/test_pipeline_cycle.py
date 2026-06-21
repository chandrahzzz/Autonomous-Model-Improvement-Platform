"""
Integration test: full pipeline cycle with mocked external services.
Tests that the graph can execute a complete cycle without errors.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_failure_classifier_full_cycle(sample_llm_events):
    """Test full failure detection cycle with mocked detectors."""
    from src.detection.failure_classifier import FailureClassifier, FailureBatch
    from src.detection.refusal import RefusalDetector
    from src.detection.format_validator import FormatValidator

    # Use real refusal and format detectors but mock heavy ML models
    refusal = RefusalDetector()
    fmt = FormatValidator()

    with patch("src.detection.hallucination.CrossEncoder") as mock_ce, \
         patch("src.detection.drift.SentenceTransformer"):
        mock_ce.return_value.predict.return_value = [0.3] * len(sample_llm_events)

        from src.detection.hallucination import HallucinationDetector
        from src.detection.drift import DriftDetector

        hall = HallucinationDetector()
        drift = DriftDetector()

        classifier = FailureClassifier(hall, drift, refusal, fmt)
        batch = await classifier.classify_batch(sample_llm_events)

    assert isinstance(batch, FailureBatch)
    assert batch.total_processed == len(sample_llm_events)
    assert isinstance(batch.drift_score, float)


@pytest.mark.asyncio
async def test_quality_filter_and_dedup_pipeline():
    """Test curation sub-pipeline: quality filter + dedup working together."""
    from src.curation.quality_filter import QualityFilter
    from src.curation.deduplicator import Deduplicator

    qf = QualityFilter()
    dedup = Deduplicator()

    examples = [
        ("What is AI?", "I can't help.", "AI is Artificial Intelligence — a field studying intelligent machines.", 0.95),
        ("What is ML?", "I won't help.", "ML (Machine Learning) is a subset of AI that learns from data.", 0.90),
        # Duplicate
        ("What is AI?", "I can't help.", "AI is Artificial Intelligence — a field studying intelligent machines.", 0.95),
    ]

    passed = []
    for prompt, bad, corrected, confidence in examples:
        passes, score, reason = qf.passes(prompt, bad, corrected, confidence)
        if not passes:
            continue
        h = dedup.compute_hash(prompt, corrected)
        if dedup.is_duplicate(prompt, corrected):
            continue
        passed.append({"prompt": prompt, "corrected": corrected, "score": score})

    assert len(passed) == 2   # duplicate filtered out


@pytest.mark.asyncio
async def test_audit_logger_signs_entry(mock_db):
    """Test that audit logger produces a valid HMAC signature."""
    from src.audit.logger import AuditLogger
    from src.audit.schemas import AuditEvent
    from src.audit.hmac_signer import HMACSigner
    from unittest.mock import AsyncMock, MagicMock

    # Mock the repository
    mock_entry = MagicMock()
    mock_entry.id = 42
    mock_entry.event_type = "training_triggered"
    mock_entry.decision = "test decision"
    mock_entry.rationale = {}
    mock_entry.state_snapshot = {}
    mock_entry.operator = "autonomous_pipeline"
    mock_entry.created_at = __import__("datetime").datetime.utcnow()

    with patch("src.audit.logger.AuditRepository") as MockRepo:
        mock_repo_instance = AsyncMock()
        mock_repo_instance.insert = AsyncMock(return_value=mock_entry)
        MockRepo.return_value = mock_repo_instance

        logger = AuditLogger(mock_db)
        event = AuditEvent(
            event_type="training_triggered",
            decision="start training",
            rationale={"pending": 600},
            state_snapshot={"version": "v7"},
        )
        row_id = await logger.log(event)

    assert row_id == 42
    assert mock_repo_instance.insert.called

    # Verify HMAC
    signer = HMACSigner()
    call_args = mock_repo_instance.insert.call_args[0][0]
    payload = {
        "event_type": call_args["event_type"],
        "decision": call_args["decision"],
        "rationale": call_args["rationale"],
        "state_snapshot": call_args["state_snapshot"],
        "operator": call_args["operator"],
    }
    assert signer.verify(payload, call_args["hmac_sha256"])
