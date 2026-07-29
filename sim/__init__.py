"""
Synthetic traffic simulator for the continuous fine-tuning pipeline.

Additive, test-only tooling: it fabricates realistic `llm_logs` traffic so the
detect → curate → train → eval → promote loop has something to observe. It never
runs in production and never touches pipeline logic — it only writes events
through the same interfaces a real app would (DB rows / Kafka events / the HTTP
interceptor), then lets the real detectors decide what is a failure.

Public surface:
    from sim import SimConfig, build_call_plan, generate_call, emit_calls
"""

from sim.config import SimConfig, FAILURE_TYPES, DOMAINS
from sim.answers import Call, generate_call
from sim.scenarios import build_call_plan, CallSpec
from sim.emit import emit_calls

__all__ = [
    "SimConfig",
    "FAILURE_TYPES",
    "DOMAINS",
    "Call",
    "generate_call",
    "build_call_plan",
    "CallSpec",
    "emit_calls",
]
