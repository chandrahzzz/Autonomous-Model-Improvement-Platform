"""
RAGAS evaluation on the held-out eval set.

Metrics computed:
  - faithfulness: does the answer stick to the context?
  - answer_relevancy: does the answer address the question?
  - context_recall: does the context contain the ground truth?

We run these against a held-out set seeded via scripts/seed_eval_set.py.
"""

import structlog
from typing import Any

from src.config.settings import settings
from src.monitoring.metrics import (
    eval_faithfulness_gauge,
    eval_relevancy_gauge,
    eval_recall_gauge,
)

log = structlog.get_logger()


class RAGASRunner:
    def __init__(self) -> None:
        # Lazy import to avoid loading at startup
        self._dataset: list[dict] | None = None

    async def run(
        self,
        model_invoke_fn: Any,
        eval_set: list[dict],
    ) -> dict[str, float]:
        """
        Run RAGAS evaluation.

        eval_set items must have: question, context, ground_truth.
        model_invoke_fn: async callable (prompt: str) -> str

        Returns dict with faithfulness, answer_relevancy, context_recall.
        """
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_recall
        from datasets import Dataset

        log.info("ragas_eval_starting", n_examples=len(eval_set))

        # Generate model answers. The model must see the same context that
        # faithfulness/context_recall are scored against — otherwise those
        # metrics grade the answer against context the model never received,
        # which makes context_recall in particular meaningless.
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

        # Build RAGAS Dataset
        ragas_data = {
            "question": [item["question"] for item in eval_set],
            "answer": answers,
            "contexts": [[item["context"]] for item in eval_set],
            "ground_truth": [item["ground_truth"] for item in eval_set],
        }
        dataset = Dataset.from_dict(ragas_data)

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall],
        )

        scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_recall": float(result["context_recall"]),
        }

        eval_faithfulness_gauge.set(scores["faithfulness"])
        eval_relevancy_gauge.set(scores["answer_relevancy"])
        eval_recall_gauge.set(scores["context_recall"])

        log.info("ragas_eval_complete", **scores)
        return scores
