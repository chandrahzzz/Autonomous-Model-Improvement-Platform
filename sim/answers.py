"""
Call generation: turn a (domain, failure_type) decision into a concrete
prompt/completion pair engineered to trip the REAL detector for that failure.

Each injector is written against the actual detector logic (verified in
src/detection/*):

  refusal        → completion contains a phrase matching refusal.REFUSAL_PATTERN
                   (e.g. "I cannot", "As an AI", "I apologize, but", "I must decline").
  hallucination  → a RAG call whose retrieved_context is an Acme fact and whose
                   completion asserts the OPPOSITE, so NLI entailment collapses
                   (hallucination_score = 1 - entailment > 0.50).
  drift          → completion drawn from a distribution far from the healthy
                   baseline text, pushing Mahalanobis distance up. (Needs the
                   drift baseline seeded + >= drift_min_window samples to fire.)
  format_break   → for a JSON-expected prompt, emit prose / broken JSON; also
                   length-shift hard so the length-KL signal moves once a format
                   baseline exists. (KL fires only after a length baseline is set,
                   which the pipeline does post-promotion — documented caveat.)

A `Call` is provider-agnostic; sim/emit.py maps it onto a DB row, a Kafka
LLMEvent, or an HTTP request.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from sim.prompts import ACME_FACTS, BANKS


@dataclass
class Call:
    prompt: str
    completion: str
    domain: str
    failure_type: str | None = None      # None = healthy; else the INTENDED failure
    retrieved_context: str = ""
    is_rag: bool = False
    finish_reason: str = "stop"
    expect_json: bool = False
    metadata: dict = field(default_factory=dict)


# Refusal phrases — each contains a substring that refusal.REFUSAL_PATTERN matches.
REFUSAL_COMPLETIONS = [
    "I cannot help with that request.",
    "I'm not able to provide that information.",
    "As an AI, I can't assist with this.",
    "I apologize, but I must decline this request.",
    "I won't be able to do that, unfortunately.",
    "I'm unable to answer that question.",
    "That's not something I can help with.",
    "I don't have the ability to complete this task.",
]

# Off-distribution text for drift: dense, jargon-salad sentences unlike the
# short factual/policy answers that form a healthy baseline. Deliberately varied
# so they don't collapse into one cluster.
DRIFT_COMPLETIONS = [
    "Quantum flux capacitors reticulate the hyperbolic manifold while entropic "
    "gradients cascade through the toroidal substrate of the eigenstate lattice.",
    "The chthonic resonance of post-structural bricolage subsumes the liminal "
    "praxis of deterritorialized semiotic assemblages in recursive abundance.",
    "Interstellar plasma harmonics oscillate across the baryonic acoustic "
    "horizon as dark-sector wavefunctions decohere into fractal supersymmetry.",
    "Mycelial blockchain oracles synthesize non-Euclidean tensor foams beneath "
    "the stochastic drift of an anisotropic phlogiston reservoir.",
    "Palimpsestic hydrothermal vent metabolics entrain the circadian dynamo of "
    "abyssal chemolithoautotrophs across a Riemannian curvature field.",
    "Gyroscopic meta-linguistic turbulence deconstructs the phenomenological "
    "residue of holographic thermodynamic bifurcations ad infinitum.",
]


def _fill(template: dict, rng: random.Random) -> dict[str, str]:
    if not template["keys"] or not template["opts"]:
        return {}
    opt = rng.choice(template["opts"])
    return dict(zip(template["keys"], opt))


def _render(template: dict, rng: random.Random) -> tuple[str, str, dict]:
    fill = _fill(template, rng)
    prompt = template["q"].format(**fill)
    answer = template["a"].format(**fill)
    return prompt, answer, fill


def _pick_template(domain: str, rng: random.Random) -> dict:
    return rng.choice(BANKS[domain])


# ── Healthy ──────────────────────────────────────────────────────────────────
def _healthy(domain: str, rng: random.Random) -> Call:
    tmpl = _pick_template(domain, rng)
    prompt, answer, _ = _render(tmpl, rng)
    ctx = ACME_FACTS.get(tmpl.get("context_source") or "", "")
    return Call(
        prompt=prompt, completion=answer, domain=domain, failure_type=None,
        retrieved_context=ctx, is_rag=bool(ctx), finish_reason="stop",
        expect_json=tmpl.get("expect_json", False),
    )


# ── Refusal ──────────────────────────────────────────────────────────────────
def _refusal(domain: str, rng: random.Random) -> Call:
    tmpl = _pick_template(domain, rng)
    prompt, _, _ = _render(tmpl, rng)
    ctx = ACME_FACTS.get(tmpl.get("context_source") or "", "")
    return Call(
        prompt=prompt, completion=rng.choice(REFUSAL_COMPLETIONS), domain=domain,
        failure_type="refusal", retrieved_context=ctx, is_rag=bool(ctx),
        finish_reason="stop", expect_json=tmpl.get("expect_json", False),
    )


# ── Hallucination (RAG contradiction) ────────────────────────────────────────
# Wrong answers that directly contradict the Acme fact for a given source id.
_HALLUCINATION_BY_SOURCE = {
    "policy-refunds": "You can return any item to Acme for a full refund at any "
                      "time, even years later, and even if it has been used and "
                      "the packaging thrown away. Cash refunds are instant.",
    "policy-shipping": "Acme does not ship domestic orders at all; everything "
                       "takes at least eight weeks and there is no expedited "
                       "option under any circumstances.",
    "policy-support-hours": "Acme support is available 24 hours a day, seven days "
                            "a week, including all public holidays, by phone only.",
    "policy-plans": "Acme offers a single flat plan with no tiers, and annual "
                    "billing actually costs 50 percent more than monthly.",
    "policy-subscription-changes": "At Acme, downgrades take effect immediately "
                                   "with a refund, and upgrades are queued until "
                                   "the following year.",
    "policy-payments": "Acme only accepts payment in cryptocurrency and cash on "
                       "delivery; credit cards and PayPal are never accepted.",
    "policy-data-retention": "Acme keeps all deleted account data forever and "
                             "never erases backups.",
    "policy-returns-process": "To return an item, mail it to Acme headquarters at "
                              "your own expense; refunds take at least six months.",
}


def _hallucination(domain: str, rng: random.Random) -> Call:
    # Force a business_policy template so we have a real fact to contradict.
    tmpl = _pick_template("business_policy", rng)
    src = tmpl.get("context_source")
    prompt, _, _ = _render(tmpl, rng)
    ctx = ACME_FACTS.get(src or "", "")
    wrong = _HALLUCINATION_BY_SOURCE.get(
        src, "That is completely false; the real policy is the exact opposite in "
             "every respect and none of the stated conditions apply."
    )
    return Call(
        prompt=prompt, completion=wrong, domain="business_policy",
        failure_type="hallucination", retrieved_context=ctx, is_rag=True,
        finish_reason="stop",
    )


# ── Drift ────────────────────────────────────────────────────────────────────
def _drift(domain: str, rng: random.Random) -> Call:
    tmpl = _pick_template(domain, rng)
    prompt, _, _ = _render(tmpl, rng)
    return Call(
        prompt=prompt, completion=rng.choice(DRIFT_COMPLETIONS), domain=domain,
        failure_type="drift", finish_reason="stop",
    )


# ── Format break ─────────────────────────────────────────────────────────────
# Prose where JSON was requested + a hard length shift (very long) so the
# length-distribution KL signal moves once a format baseline exists.
_FORMAT_BREAK_COMPLETIONS = [
    "Sure! Here is the information you asked for, written out in plain prose "
    "instead of the structured JSON you requested, going on at considerable "
    "length with many extra clauses and asides and caveats and elaborations "
    "that keep expanding well beyond any reasonable response size, drifting far "
    "from the compact machine-readable object that was expected, rambling onward "
    "and onward without ever closing a single brace or bracket as required.",
    "Well, it depends on how you look at it, and there are many considerations "
    "to weigh before giving a structured answer, so instead of JSON here is a "
    "long meandering paragraph that never actually returns the object you wanted "
    "and instead keeps talking and talking about tangential matters at great "
    "length far exceeding the usual concise structured reply.",
    '{"name": "Alice", "age": 30',      # truncated / invalid JSON (unbalanced)
    '{name: Alice, age: thirty}',        # not valid JSON (unquoted / word number)
]


def _format_break(domain: str, rng: random.Random) -> Call:
    # Prefer a JSON template so the "expected JSON, got prose" signal applies.
    tmpl = _pick_template("json_format", rng)
    prompt, _, _ = _render(tmpl, rng)
    return Call(
        prompt=prompt, completion=rng.choice(_FORMAT_BREAK_COMPLETIONS),
        domain="json_format", failure_type="format_break",
        finish_reason="stop", expect_json=True,
    )


_INJECTORS = {
    None: _healthy,
    "healthy": _healthy,
    "refusal": _refusal,
    "hallucination": _hallucination,
    "drift": _drift,
    "format_break": _format_break,
}


def generate_call(domain: str, failure_type: str | None, rng: random.Random) -> Call:
    """Produce one Call for the given domain + intended failure (None = healthy)."""
    injector = _INJECTORS.get(failure_type, _healthy)
    return injector(domain, rng)
