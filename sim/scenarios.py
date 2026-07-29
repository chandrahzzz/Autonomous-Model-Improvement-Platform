"""
Scenario shaping: decide, per call, whether it is healthy or a specific failure
and which domain it belongs to. The per-type RATES come from SimConfig; the
SCENARIO decides how those rates are applied across the run (steady, ramping,
bursting, etc.).

`build_call_plan(cfg, n)` returns a deterministic list[CallSpec] (seeded), so a
run is reproducible and unit-testable without touching any external service.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sim.config import SimConfig, FAILURE_TYPES

# Domain mix for healthy / refusal / drift calls (hallucination + format_break
# force their own domain in the injectors, so this only shapes the rest).
_DOMAIN_WEIGHTS = {
    "business_policy": 0.35,
    "factual": 0.30,
    "reasoning": 0.20,
    "json_format": 0.15,
}
_DOMAINS = list(_DOMAIN_WEIGHTS)
_WEIGHTS = list(_DOMAIN_WEIGHTS.values())


@dataclass
class CallSpec:
    failure_type: str | None   # None = healthy
    domain: str


def _pick_domain(rng: random.Random) -> str:
    return rng.choices(_DOMAINS, weights=_WEIGHTS, k=1)[0]


def _sample_failure(rng: random.Random, rates: dict[str, float]) -> str | None:
    """Independent Bernoulli per failure type, in severity order; first hit wins.
    Returns None (healthy) if none fire."""
    for ftype in FAILURE_TYPES:
        if rng.random() < rates.get(ftype, 0.0):
            return ftype
    return None


def _peak_rates(cfg: SimConfig) -> dict[str, float]:
    return {f: cfg.peak_rate_for(f) for f in FAILURE_TYPES}


def _dominant_failure(cfg: SimConfig) -> str:
    rates = _peak_rates(cfg)
    best = max(rates, key=lambda k: rates[k])
    return best if rates[best] > 0 else "refusal"


def build_call_plan(cfg: SimConfig, n: int) -> list[CallSpec]:
    rng = random.Random(cfg.seed)
    scenario = cfg.scenario
    peak = _peak_rates(cfg)
    plan: list[CallSpec] = []

    for i in range(n):
        frac = i / max(1, n)

        if scenario == "healthy":
            failure = None

        elif scenario == "mixed":
            failure = _sample_failure(rng, peak)

        elif scenario == "degrade":
            # First 30% healthy warm-up (also feeds a clean drift/format baseline),
            # then ramp every configured rate 0 -> peak over the remaining 70%.
            if frac < 0.30:
                failure = None
            else:
                ramp = (frac - 0.30) / 0.70
                scaled = {f: r * ramp for f, r in peak.items()}
                failure = _sample_failure(rng, scaled)

        elif scenario == "burst":
            # Healthy except a spike of one dominant failure type in the middle.
            if 0.40 <= frac < 0.60:
                dominant = _dominant_failure(cfg)
                failure = dominant if rng.random() < 0.90 else None
            else:
                failure = None

        elif scenario == "rag_heavy":
            # Mostly grounded business_policy traffic; a fraction hallucinate.
            hall_rate = peak["hallucination"] or 0.20
            failure = "hallucination" if rng.random() < hall_rate else None
            domain = "business_policy"
            plan.append(CallSpec(failure_type=failure, domain=domain))
            continue

        else:
            raise ValueError(f"unknown scenario: {scenario!r}")

        # Domain: forced for hallucination/format_break (injector overrides it),
        # weighted otherwise.
        if failure == "hallucination":
            domain = "business_policy"
        elif failure == "format_break":
            domain = "json_format"
        else:
            domain = _pick_domain(rng)

        plan.append(CallSpec(failure_type=failure, domain=domain))

    return plan


SCENARIOS = ("healthy", "degrade", "mixed", "burst", "rag_heavy")
