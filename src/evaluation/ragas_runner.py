"""
RAGAS-style evaluation on the held-out eval set — $0 local metrics.

The upstream RAGAS library calls a paid LLM per metric. This runner computes
the same three metrics with models already resident in the pipeline, entirely
on CPU, with zero API calls:

  - faithfulness:     NLI entailment of the answer by the context, via the
                      existing HallucinationDetector singleton (premise=context,
                      hypothesis=answer). Same model the detection layer uses.
  - answer_relevancy: MiniLM cosine similarity between question and answer
                      embeddings (all-MiniLM-L6-v2, shared encoder).
  - context_recall:   ROUGE-L recall of the ground truth against the context —
                      "what fraction of the ground truth does the context cover".

Score semantics and the returned dict keys are unchanged, so the orchestrator,
gauges, and promotion gates need no edits.
"""

import asyncio

import numpy as np
import structlog
from typing import Any

from src.monitoring.metrics import (
    eval_faithfulness_gauge,
    eval_relevancy_gauge,
    eval_recall_gauge,
)

log = structlog.get_logger()


class RAGASRunner:
    def __init__(self) -> None:
        self._hall_detector = None  # lazy; overridable in tests

    def _get_hall_detector(self):
        if self._hall_detector is None:
            from src.graph.nodes.failure_detector import _hall
            self._hall_detector = _hall
        return self._hall_detector

    async def run(
        self,
        model_invoke_fn: Any,
        eval_set: list[dict],
    ) -> dict[str, float]:
        """
        Run the local evaluation.

        eval_set items must have: question, context, ground_truth.
        model_invoke_fn: async callable (prompt: str) -> str

        Returns dict with faithfulness, answer_relevancy, context_recall.
        """
        log.info("local_ragas_eval_starting", n_examples=len(eval_set))

        # Generate model answers. The model must see the same context that
        # faithfulness is scored against — otherwise the metric grades the
        # answer against context the model never received.
        answers = []
        for item in eval_set:
            context = item.get("context", "")
            prompt = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\nQuestion: {item['question']}"
                if context
                else item["question"]
            )
            try:
                answer = await model_invoke_fn(prompt)
                answers.append(str(answer))
            except Exception:
                log.exception("ragas_model_invoke_failed")
                answers.append("")

        faithfulness = await self._faithfulness(eval_set, answers)
        relevancy = await self._answer_relevancy(eval_set, answers)
        recall = self._context_recall(eval_set)

        scores = {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_recall": recall,
        }

        eval_faithfulness_gauge.set(scores["faithfulness"])
        eval_relevancy_gauge.set(scores["answer_relevancy"])
        eval_recall_gauge.set(scores["context_recall"])

        log.info("local_ragas_eval_complete", **scores)
        return scores

    async def _faithfulness(self, eval_set: list[dict], answers: list[str]) -> float:
        """Mean NLI entailment of each answer by its context. The detector
        returns hallucination probability (1 - entailment), scored per sentence
        internally, so faithfulness = 1 - mean(hallucination). Empty answers
        (model errors) count as fully unfaithful."""
        pairs = []
        empty = 0
        for item, answer in zip(eval_set, answers):
            if not answer.strip():
                empty += 1
                continue
            premise = item.get("context") or item["question"]
            pairs.append((premise, answer))
        if not pairs and empty == 0:
            return 0.0
        try:
            hall_scores = await self._get_hall_detector().score_batch(pairs) if pairs else []
        except Exception:
            log.exception("faithfulness_nli_failed")
            return 0.0
        entailments = [1.0 - s for s in hall_scores] + [0.0] * empty
        return float(np.mean(entailments)) if entailments else 0.0

    async def _answer_relevancy(self, eval_set: list[dict], answers: list[str]) -> float:
        """Mean cosine similarity between question and answer embeddings.
        Empty answers score 0."""
        questions = [item["question"] for item in eval_set]

        def _encode_and_score() -> float:
            from src.curation.teacher import _shared_encoder
            encoder = _shared_encoder()
            q_emb = np.asarray(encoder.encode(questions, show_progress_bar=False))
            a_texts = [a if a.strip() else " " for a in answers]
            a_emb = np.asarray(encoder.encode(a_texts, show_progress_bar=False))
            q_norm = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
            a_norm = a_emb / np.clip(np.linalg.norm(a_emb, axis=1, keepdims=True), 1e-12, None)
            sims = np.sum(q_norm * a_norm, axis=1)
            sims = np.where([bool(a.strip()) for a in answers], sims, 0.0)
            return float(np.clip(sims, 0.0, 1.0).mean())

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _encode_and_score)
        except Exception:
            log.exception("answer_relevancy_failed")
            return 0.0

    def _context_recall(self, eval_set: list[dict]) -> float:
        """ROUGE-L recall of the ground truth against the context: high when the
        context actually contains the material needed to produce the ground
        truth. Items without both fields are skipped."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        except Exception:
            log.exception("context_recall_rouge_unavailable")
            return 0.0
        recalls = []
        for item in eval_set:
            gt = item.get("ground_truth", "")
            ctx = item.get("context", "")
            if not gt or not ctx:
                continue
            recalls.append(scorer.score(gt, ctx)["rougeL"].recall)
        return float(np.mean(recalls)) if recalls else 0.0
