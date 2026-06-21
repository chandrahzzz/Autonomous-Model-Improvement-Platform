"""
Teacher model correction generation (RAG-grounded).

Uses GPT-4o to generate the ideal corrected completion for each failure.
Confidence is a self-consistency score (pairwise ROUGE-L across 3 corrections).

When the original production answer used retrieved context (`llm_logs.retrieved_context`),
the teacher is constrained to answer ONLY from that context and the correction is
verified against it with the existing NLI detector. Corrections that aren't entailed
by their context are dropped, so domain-specific (policy/pricing/internal) failures
can't be silently "corrected" with GPT-4o's outside (and possibly wrong) knowledge.
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import settings
from src.detection.failure_classifier import FailureEvent
from src.monitoring.metrics import teacher_corrections_rejected_grounding_total

log = structlog.get_logger()

SYSTEM_PROMPT = """You are an expert LLM evaluator and corrector.
You will be given an AI assistant's output that has a specific quality failure.
Your task is to provide the IDEAL corrected response — accurate, helpful,
well-formatted, non-refusatory, and factually grounded.
Only output the corrected response text. No explanations."""

# Minimum NLI entailment of a correction by its context to accept it.
GROUNDING_THRESHOLD = 0.50
# Sentinel the teacher emits when the context can't answer the question.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

# GPT-4o pricing (USD per token), used for the per-run cost circuit breaker.
_INPUT_COST_PER_TOKEN = 5e-6
_OUTPUT_COST_PER_TOKEN = 15e-6
_CHARS_PER_TOKEN = 4.0

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
    confidence: float                 # self-consistency (ROUGE-L across 3 votes)
    grounding_score: float | None     # NLI entailment vs context; None if no context
    grounding_sources: list[str] = field(default_factory=list)
    is_grounded: bool = True


class TeacherModel:
    def __init__(self) -> None:
        # temperature=0.0 base model used for the deterministic answer.
        self._llm = ChatOpenAI(
            model=settings.teacher_model,
            temperature=0.0,
            api_key=settings.openai_api_key,
        )
        # Separate sampler at >0 temperature for the self-consistency votes —
        # at temp 0 the extra samples are identical and the ROUGE consistency
        # score is meaninglessly ~1.0 while still costing N× the API calls.
        self._sampler_llm = ChatOpenAI(
            model=settings.teacher_model,
            temperature=settings.teacher_consistency_temperature,
            api_key=settings.openai_api_key,
        )
        self._consistency_n = 3
        # Bound simultaneous OpenAI calls across the whole curation batch.
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

            # 3. Generate corrections (first deterministic, rest sampled).
            tasks = [self._single_correction(teacher_prompt, use_sampler=False)]
            tasks += [
                self._single_correction(teacher_prompt, use_sampler=True)
                for _ in range(self._consistency_n - 1)
            ]
            corrections = await asyncio.gather(*tasks)
            valid = [c for c in corrections if c]
            if len(valid) < 2:
                return None  # not enough votes to judge consistency

            # 4. Self-consistency confidence (unchanged logic).
            confidence = self._compute_consistency_score(valid)
            if confidence < settings.teacher_confidence_threshold:
                log.debug(
                    "teacher_confidence_below_threshold",
                    failure_type=failure.failure_type,
                    confidence=confidence,
                )
                return None

            # 5. Pick the most representative correction.
            best = self._pick_best(valid)

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
                if grounding_score < GROUNDING_THRESHOLD:
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
        self, teacher_prompt: str, use_sampler: bool = False
    ) -> str | None:
        async with self._semaphore:
            try:
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=teacher_prompt),
                ]
                llm = self._sampler_llm if use_sampler else self._llm
                response = await llm.ainvoke(messages)
                text = str(response.content).strip()
                self._total_cost_usd += self._estimate_cost(teacher_prompt, text)
                return text
            except Exception:
                log.exception("teacher_single_correction_failed")
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

    def _pick_best(self, corrections: list[str]) -> str:
        """Return the correction with the highest mean pairwise ROUGE-L vs the
        others (the most representative / least outlier answer)."""
        if len(corrections) == 1:
            return corrections[0]
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
        """Rough per-call cost estimate from character counts."""
        input_tokens = (len(prompt_text) + len(SYSTEM_PROMPT)) / _CHARS_PER_TOKEN
        output_tokens = len(output) / _CHARS_PER_TOKEN
        return (input_tokens * _INPUT_COST_PER_TOKEN
                + output_tokens * _OUTPUT_COST_PER_TOKEN)

    def _compute_consistency_score(self, corrections: list[str]) -> float:
        """ROUGE-L pairwise overlap as consistency proxy. (Unchanged.)"""
        if len(corrections) == 1:
            return 1.0
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
        return f"""You are correcting a factually wrong answer.

CRITICAL RULE: Your correction MUST be based ONLY on the context documents provided below.
Do not use any outside knowledge. If the context does not contain enough information
to answer the question correctly, respond with exactly: {INSUFFICIENT}

Context documents:
---
{context[:6000]}
---

Original question: {prompt[:2000]}

Wrong answer that was given: {bad_answer[:2000]}

Correct answer (based strictly on the context above):"""

    def _build_general_prompt(self, prompt: str, bad_answer: str) -> str:
        return f"""You are correcting a low-quality answer.

Original question: {prompt[:2000]}

Wrong answer that was given: {bad_answer[:2000]}

Provide the ideal corrected response:"""
