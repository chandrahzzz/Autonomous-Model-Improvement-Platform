"""
Integration test: verify HMAC audit chain integrity.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.audit.hmac_signer import HMACSigner


def test_hmac_sign_and_verify():
    signer = HMACSigner()
    payload = {
        "event_type": "model_promoted",
        "decision": "promote v8 to production",
        "rationale": {"faithfulness": 0.85},
        "state_snapshot": {"version": "v8"},
        "operator": "autonomous_pipeline",
    }
    sig = signer.sign(payload)
    assert signer.verify(payload, sig) is True


def test_hmac_detects_tampering():
    signer = HMACSigner()
    payload = {
        "event_type": "model_promoted",
        "decision": "promote v8 to production",
        "rationale": {},
        "state_snapshot": {},
        "operator": "autonomous_pipeline",
    }
    sig = signer.sign(payload)

    # Tamper with the payload
    payload["decision"] = "rollback v8"
    assert signer.verify(payload, sig) is False


def test_hmac_key_determinism():
    signer = HMACSigner()
    payload = {"key": "value", "num": 42}
    sig1 = signer.sign(payload)
    sig2 = signer.sign(payload)
    assert sig1 == sig2


def test_hmac_different_payloads_different_sigs():
    signer = HMACSigner()
    sig1 = signer.sign({"a": 1})
    sig2 = signer.sign({"a": 2})
    assert sig1 != sig2


def test_hmac_key_order_irrelevant():
    signer = HMACSigner()
    p1 = {"b": 2, "a": 1}
    p2 = {"a": 1, "b": 2}
    assert signer.sign(p1) == signer.sign(p2)
