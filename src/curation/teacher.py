"""
Teacher model correction generation (RAG-grounded), $0 stack.

Uses the Groq free tier (permanent, ~30 req/min, no card required) instead of
GPT-4o. Self-consistency votes come from THREE DIFFERENT free models
(Llama-3-70B / Mixtral / Gemma-9B by default) run concurrently — cross-model
agreement is a stronger consensus signal than temperature-resampling a single
model, and it costs the same: $0. Consensus is scored semantically (mean
pairwise MiniLM cosine) and the centroid-nearest vote is kept.

When the original production answer used retrieved context
(`llm_logs.retrieved_context`), the teacher is constrained to answer ONLY from
that context and the correction is verified against it with the existing NLI
detector. Corrections that aren't entailed by their context are dropped, so
domain-specific (policy/pricing/internal) failures can't be silently
"corrected" with the teacher's outside (and possibly wrong) knowledge.
"""

import asyncio
import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import structlog

from src.config.settings import settings
from src.detection.failure_classifier import FailureEvent
from src.monitoring.metrics import (
    teacher_corrections_rejected_grounding_total,
    teacher_rate_limit_retries_total,
    teacher_dropped_rate_limited_total,
)

log = structlog.get_logger()

# Transient Groq errors worth retrying with backoff (vs. dropping the example).
# The groq SDK mirrors the openai SDK's exception hierarchy.
try:
    from groq import (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )
    _RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
        RateLimitError, APITimeoutError, APIConnectionError, InternalServerError,
    )
except Exception:  # pragma: no cover - groq installed in this project
    _RETRYABLE_ERRORS = ()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _shared_encoder():
    """One process-wide MiniLM encoder for the semantic consistency vote.
    Lazy + cached so importing teacher.py doesn't load the model and we don't
    keep a 4th copy beyond drift/refusal/clusterer."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _groq_client():
    """One process-wide AsyncGroq client (connection pool reuse)."""
    from groq import AsyncGroq
    return AsyncGroq(api_key=settings.groq_api_key)


class _RatePacer:
    """Client-side pacing for Groq's free-tier request/min ceiling. Serializes
    call STARTS so concurrent teacher calls (asyncio.gather across a batch)
    never burst past the limit; the calls themselves still overlap."""

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / max(1, requests_per_minute)
        self._lock = asyncio.Lock()
        self._last_start = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._last_start + self._interval - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_start = time.monotonic()


@lru_cache(maxsize=1)
def _shared_pacer() -> _RatePacer:
    return _RatePacer(settings.groq_requests_per_minute)


class _GroqChat:
    """Thin ChatOpenAI-shaped adapter over AsyncGroq: `ainvoke(messages)` takes
    a list of {"role", "content"} dicts and returns an object with `.content`.
    Keeping this interface means the retry loop (and its tests) don't care which
    provider is behind it."""

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    async def ainvoke(self, messages: list[dict]) -> SimpleNamespace:
        await _shared_pacer().wait()
        resp = await _groq_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=1024,
        )
        return SimpleNamespace(content=resp.choices[0].message.content or "")


def _mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity across rows (1.0 = identical meaning)."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    unit = embeddings / norms
    sims = unit @ unit.T
    iu = np.triu_indices(n, k=1)
    return float(np.clip(sims[iu].mean(), 0.0, 1.0))

SYSTEM_PROMPT = """You are an expert LLM evaluator and corrector.
You will be given an AI assistant's output that has a specific quality failure.
Provide the IDEAL corrected response: accurate, factually grounded, and DIRECT.

STRICT OUTPUT RULES:
- Output ONLY the corrected answer text — nothing else.
- Be terse. State the facts directly. Prefer 1-3 sentences.
- NO preamble ("Sure", "Based on the context", "Here is"), NO sign-off
  ("Let me know", "I hope this helps"), NO meta-commentary, NO apologies.
- Every sentence must be a factual claim supported by the source; do not add
  filler, opinions, or conversational remarks."""

# Minimum NLI entailment of a correction by its context to accept it (module
# fallback; the live value is settings.teacher_grounding_threshold).
GROUNDING_THRESHOLD = 0.50
# Sentinel the teacher emits when the context can't answer the question.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

# Source-ID extraction patterns for retrieved_context.
_SOURCE_PATTERNS = [
    re.compile(r"\[doc_id:\s*([^\]]+)\]", re.IGNORECASE),
    re.compile(r"\bchunk_id:\s*([^\s,\]]+)", re.IGNORECASE),
    re.compile(r"\bsource:\s*([^\s,\]]+)", re.IGNORECASE),
    re.compile(r"\bDocument\s+(\d+)\s*:", re.IGNORECASE),
]
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


@dataclass
class GroundingResult:
    correction: str
    confidence: float                 # cross-model semantic self-consistency
    grounding_score: float | None     # NLI entailment vs context; None if no context
    grounding_sources: list[str] = field(default_factory=list)
    is_grounded: bool = True


class TeacherModel:
    def __init__(self) -> None:
        # One deterministic client per configured Groq model. Cross-model votes
        # replace the old temperature-resampling scheme: three independent model
        # families agreeing on meaning is a stronger signal than one model
        # agreeing with its own samples.
        models = settings.teacher_models or [settings.teacher_model]
        self._llms: list[_GroqChat] = [_GroqChat(m, temperature=0.0) for m in models]
        self._llm = self._llms[0]
        # Sampler on the primary model pads the vote count when fewer than
        # _consistency_n distinct models are configured.
        self._sampler_llm = _GroqChat(
            models[0], temperature=settings.teacher_consistency_temperature
        )
        self._consistency_n = 3
        # Bound simultaneous Groq calls across the whole curation batch (the
        # _RatePacer additionally spaces call starts to the req/min ceiling).
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_teacher_calls)
        self._cost_limit_usd = settings.curation_cost_budget_usd
        self._total_cost_usd = 0.0
        # Reused NLI detector (the singleton already loaded in the failure
        # detector node). Resolved lazily so importing teacher.py doesn't pull in
        # the graph layer / load detection models eagerly. Overridable in tests.
        self._hall_detector = None

    def reset_cost(self) -> None:
        """Reset the per-run spend counter. Call once at the start of each
        curation run (the TeacherModel instance is a long-lived singleton)."""
        self._total_cost_usd = 0.0

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    def _get_hall_detector(self):
        if self._hall_detector is None:
            from src.graph.nodes.failure_detector import _hall
            self._hall_detector = _hall
        return self._hall_detector

    async def generate_correction(
        self, failure: FailureEvent
    ) -> GroundingResult | None:
        """
        Returns a GroundingResult, or None if the example should be dropped
        (low self-consistency, exhausted budget, ungrounded, or the teacher
        signalled INSUFFICIENT_CONTEXT).
        """
        if self._total_cost_usd >= self._cost_limit_usd:
            log.warning(
                "teacher_budget_exceeded",
                spent=round(self._total_cost_usd, 4),
                limit=self._cost_limit_usd,
            )
            return None

        try:
            # 1. Resolve grounding context.
            context = await self._resolve_context(failure)
            grounding_required = bool(context and len(context.strip()) > 20)

            # 2. Build the teacher prompt.
            if grounding_required:
                teacher_prompt = self._build_grounded_prompt(
                    failure.prompt, failure.completion, context
                )
            else:
                teacher_prompt = self._build_general_prompt(
                    failure.prompt, failure.completion
                )

            # 3. Generate corrections — one vote per configured Groq model,
            # padded with sampled votes from the primary if needed.
            tasks = [
                self._single_correction(teacher_prompt, llm=llm)
                for llm in self._llms[: self._consistency_n]
            ]
            while len(tasks) < self._consistency_n:
                tasks.append(self._single_correction(teacher_prompt, use_sampler=True))
            corrections = await asyncio.gather(*tasks)
            valid = [c for c in corrections if c]
            if len(valid) < 2:
                return None  # not enough votes to judge consistency

            # 4. Self-consistency confidence — SEMANTIC (MiniLM cosine), so
            # paraphrases of the same answer count as agreement and divergent
            # meanings don't. Embed the votes once and reuse for picking the best.
            embeddings = self._embed(valid)
            confidence = self._compute_consistency_score(valid, embeddings)
            if confidence < settings.teacher_semantic_consistency_threshold:
                log.debug(
                    "teacher_confidence_below_threshold",
                    failure_type=failure.failure_type,
                    confidence=confidence,
                )
                return None

            # 5. Pick the most representative (most semantically central) correction.
            best = self._pick_best(valid, embeddings)

            # Teacher signalled the context can't answer the question.
            if grounding_required and INSUFFICIENT in best.upper():
                log.warning(
                    "teacher_returned_insufficient_context",
                    log_id=failure.llm_log_id,
                )
                return None

            # 6. Grounding verification.
            if grounding_required:
                grounding_score = await self._verify_grounding(best, context)
                if grounding_score < settings.teacher_grounding_threshold:
                    log.warning(
                        "teacher_correction_rejected_grounding",
                        grounding_score=round(grounding_score, 3),
                        log_id=failure.llm_log_id,
                    )
                    teacher_corrections_rejected_grounding_total.inc()
                    return None
                sources = self._extract_source_ids(context)
                final_confidence = confidence * grounding_score  # penalize weak grounding
                return GroundingResult(
                    correction=best,
                    confidence=final_confidence,
                    grounding_score=grounding_score,
                    grounding_sources=sources,
                    is_grounded=True,
                )

            # 7. No grounding required (general-knowledge failure).
            return GroundingResult(
                correction=best,
                confidence=confidence,
                grounding_score=None,
                grounding_sources=[],
                is_grounded=True,
            )

        except Exception:
            log.exception("teacher_correction_failed", failure_type=failure.failure_type)
            return None

    async def _resolve_context(self, failure: FailureEvent) -> str | None:
        """Resolve grounding context for a failure, in priority order:
        1. context attached to the failure / its metadata,
        2. `llm_logs.retrieved_context` (what the production answer actually used),
        3. retrieved from the domain knowledge base (the RAG fallback) so the
           teacher can ground even when upstream didn't attach any context.
        Returns None only when no relevant context can be found anywhere."""
        ctx = getattr(failure, "retrieved_context", None)
        if not ctx and isinstance(getattr(failure, "metadata", None), dict):
            ctx = failure.metadata.get("retrieved_context")
        if ctx:
            return ctx

        log_id = failure.llm_log_id
        if log_id and log_id != "unknown":
            try:
                from src.db.connection import get_db
                from src.db.repositories.llm_logs import LLMLogRepository
                async with get_db() as db:
                    record = await LLMLogRepository(db).get_by_id(log_id)
                ctx = getattr(record, "retrieved_context", None) if record else None
                if ctx:
                    return ctx
            except Exception:
                log.warning("teacher_context_fetch_failed", log_id=log_id)

        # RAG fallback: retrieve domain context for the prompt.
        if settings.retrieval_enabled:
            ctx = await self._retrieve_context(failure.prompt)
            if ctx:
                return ctx
        return None

    async def _retrieve_context(self, prompt: str) -> str | None:
        try:
            from src.db.connection import get_db
            from src.retrieval.retriever import get_retriever
            async with get_db() as db:
                result = await get_retriever().retrieve(prompt, db)
            return result[0] if result else None
        except Exception:
            log.warning("teacher_retrieval_failed")
            return None

    async def _single_correction(
        self,
        teacher_prompt: str,
        use_sampler: bool = False,
        llm: _GroqChat | None = None,
    ) -> str | None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": teacher_prompt},
        ]
        if llm is None:
            llm = self._sampler_llm if use_sampler else self._llm
        async with self._semaphore:
            for attempt in range(settings.teacher_max_retries + 1):
                try:
                    response = await llm.ainvoke(messages)
                    text = str(response.content).strip()
                    self._total_cost_usd += self._estimate_cost(teacher_prompt, text)
                    return text
                except _RETRYABLE_ERRORS as exc:
                    if attempt >= settings.teacher_max_retries:
                        teacher_dropped_rate_limited_total.inc()
                        log.warning(
                            "teacher_dropped_rate_limited",
                            attempts=attempt + 1, error=type(exc).__name__,
                        )
                        return None
                    teacher_rate_limit_retries_total.inc()
                    delay = min(
                        settings.teacher_retry_base_delay_seconds * (2 ** attempt),
                        settings.teacher_retry_max_delay_seconds,
                    )
                    delay += random.uniform(0, settings.teacher_retry_base_delay_seconds)
                    log.info(
                        "teacher_rate_limit_retry",
                        attempt=attempt + 1, delay=round(delay, 2),
                        error=type(exc).__name__,
                    )
                    await asyncio.sleep(delay)
                except Exception:
                    log.exception("teacher_single_correction_failed")
                    return None
        return None

    def _embed(self, texts: list[str]) -> np.ndarray | None:
        """Embed correction candidates with MiniLM for the semantic consistency
        vote. Returns None on failure so callers fall back to ROUGE-L. Overridable
        in tests to avoid loading the model."""
        try:
            return np.asarray(_shared_encoder().encode(texts, show_progress_bar=False))
        except Exception:
            log.warning("teacher_embed_failed_falling_back_to_rouge")
            return None

    async def _verify_grounding(self, correction: str, context: str) -> float:
        """NLI entailment of the correction by the context, via the existing
        hallucination detector. score_batch returns hallucination probability
        (high = bad), so entailment = 1 - that."""
        detector = self._get_hall_detector()
        scores = await detector.score_batch([(context, correction)])
        hallucination_prob = scores[0] if scores else 1.0
        return 1.0 - hallucination_prob

    def _extract_source_ids(self, context: str) -> list[str]:
        """Best-effort extraction of document/chunk IDs from the context. Always
        returns at least a content-hash fallback so every grounded example is
        traceable."""
        found: list[str] = []
        for pat in _SOURCE_PATTERNS:
            found.extend(m.strip() for m in pat.findall(context))
        found.extend(_UUID_RE.findall(context[:500]))
        # De-dup, preserve order.
        seen: set[str] = set()
        ordered = [s for s in found if not (s in seen or seen.add(s))]
        if ordered:
            return ordered
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()[:12]
        return [f"context_hash:{digest}"]

    def _pick_best(self, corrections: list[str], embeddings: np.ndarray | None = None) -> str:
        """Return the most representative correction: the one closest (cosine) to
        the centroid of all votes. Falls back to mean-pairwise-ROUGE-L when
        embeddings aren't available."""
        if len(corrections) == 1:
            return corrections[0]
        if embeddings is not None and len(embeddings) == len(corrections):
            try:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12
                unit = embeddings / norms
                centroid = unit.mean(axis=0)
                sims = unit @ centroid
                return corrections[int(np.argmax(sims))]
            except Exception:
                pass
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            best, best_mean = corrections[0], -1.0
            for i, a in enumerate(corrections):
                others = [b for j, b in enumerate(corrections) if j != i]
                mean = sum(
                    scorer.score(a, b)["rougeL"].fmeasure for b in others
                ) / len(others)
                if mean > best_mean:
                    best, best_mean = a, mean
            return best
        except Exception:
            return corrections[0]

    def _estimate_cost(self, prompt_text: str, output: str) -> float:
        """Groq free tier: per-call cost is $0. The budget circuit breaker stays
        wired so a paid provider can be swapped back in without touching the
        curation flow."""
        return 0.0

    def _compute_consistency_score(
        self, corrections: list[str], embeddings: np.ndarray | None = None
    ) -> float:
        """Semantic self-consistency: mean pairwise MiniLM cosine across the
        votes. Paraphrases of one answer score high; divergent meanings score
        low. ROUGE-L is the secondary fallback only when embeddings are
        unavailable (it measures surface overlap, not meaning)."""
        if len(corrections) == 1:
            return 1.0
        if embeddings is not None and len(embeddings) == len(corrections):
            return _mean_pairwise_cosine(embeddings)
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            scores = []
            for i, a in enumerate(corrections):
                for b in corrections[i + 1:]:
                    result = scorer.score(a, b)
                    scores.append(result["rougeL"].fmeasure)
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

    def _build_grounded_prompt(self, prompt: str, bad_answer: str, context: str) -> str:
        return f"""Correct the wrong answer using ONLY the context below.
Do not use outside knowledge. If the context lacks the information, respond with
exactly: {INSUFFICIENT}

Answer in 1-3 short factual sentences drawn only from the context. Output ONLY
the corrected answer — no preamble, no sign-off, no commentary.

Context:
---
{context[:6000]}
---

Question: {prompt[:2000]}

Wrong answer: {bad_answer[:2000]}

Corrected answer:"""

    def _build_general_prompt(self, prompt: str, bad_answer: str) -> str:
        return f"""Correct the low-quality answer below.

Answer in 1-3 short factual sentences. Output ONLY the corrected answer — no
preamble, no sign-off, no commentary.

Question: {prompt[:2000]}

Wrong answer: {bad_answer[:2000]}

Corrected answer:"""
