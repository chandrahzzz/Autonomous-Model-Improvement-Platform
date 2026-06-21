"""
Continuous eval factory (RFC-002).

Generates new eval examples from recent production traffic: cluster recent
prompts, pick a representative (medoid) per cluster, ask GPT-4o for an ideal
ground-truth answer, confidence-gate via self-consistency, dedup against the
existing eval set, and insert into `eval_set` with source='factory'. LRU-evicts
the oldest factory examples when over the size cap. Seed examples are never
evicted. Purely additive — RAGAS picks the new rows up automatically.
"""

import numpy as np
import structlog

from src.config.settings import settings
from src.db.repositories.eval_set import EvalSetRepository
from src.monitoring.metrics import (
    eval_factory_runs_total,
    eval_factory_examples_added_total,
    eval_factory_examples_evicted_total,
    eval_factory_examples_skipped_total,
)

log = structlog.get_logger()

# Reused MiniLM encoder (lazy — construction stays cheap so unit tests that mock
# clustering never load the model). The codebase already loads this model name
# in drift/refusal/clustering; there is no module-level singleton to import, so
# the factory keeps one shared instance here.
_shared_encoder = None


def _get_encoder():
    global _shared_encoder
    if _shared_encoder is None:
        from sentence_transformers import SentenceTransformer
        from src.curation.clustering import EMBEDDING_MODEL
        _shared_encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _shared_encoder


def _rouge_l(a: str, b: str) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(a, b)["rougeL"].fmeasure


class PromptClusterer:
    """Adapts raw prompt strings into HDBSCAN clusters. Does NOT modify
    FailureClusterer; reuses the same MiniLM model + hdbscan import."""

    def __init__(self, min_cluster_size: int = 3) -> None:
        self._min_cluster_size = min_cluster_size

    def cluster(self, prompts: list[str]) -> dict[int, list[str]]:
        """{cluster_id: [prompt, ...]} for non-noise clusters (label != -1)."""
        if len(prompts) < self._min_cluster_size:
            return {}
        import hdbscan
        embeddings = _get_encoder().encode(prompts, batch_size=32, show_progress_bar=False)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)
        groups: dict[int, list[str]] = {}
        for prompt, label in zip(prompts, labels):
            label = int(label)
            if label == -1:
                continue
            groups.setdefault(label, []).append(prompt)
        return groups

    def pick_medoid(self, prompts: list[str]) -> tuple[str, list[float]]:
        """Return (medoid_prompt, medoid_embedding) — the prompt most similar to
        all others on average."""
        embeddings = np.asarray(
            _get_encoder().encode(prompts, batch_size=32, show_progress_bar=False),
            dtype=float,
        )
        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms == 0] = 1.0
        normed = embeddings / norms[:, None]
        sims = normed @ normed.T
        idx = int(np.argmax(sims.sum(axis=1)))
        return prompts[idx], embeddings[idx].tolist()


class EvalFactory:
    def __init__(self, eval_repo: EvalSetRepository, openai_client, redis_client) -> None:
        self._eval_repo = eval_repo
        self._openai = openai_client
        self._redis = redis_client
        self._clusterer = PromptClusterer(min_cluster_size=3)

    async def should_run(self) -> bool:
        counter = await self._redis.get(settings.eval_factory_request_counter_key)
        if counter is None:
            return False
        return int(counter) >= settings.eval_factory_trigger_every_n_requests

    async def increment_counter(self, by: int) -> None:
        await self._redis.incrby(settings.eval_factory_request_counter_key, by)

    async def reset_counter(self) -> None:
        await self._redis.set(settings.eval_factory_request_counter_key, 0)

    async def run(self, recent_prompts: list[str], db) -> int:
        clusters = self._clusterer.cluster(recent_prompts)
        if not clusters:
            log.info("eval_factory_no_clusters_found")
            return 0

        sorted_clusters = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
        candidates = sorted_clusters[: settings.eval_factory_max_examples_per_run]

        added = 0
        for cluster_id, cluster_prompts in candidates:
            if await self._try_generate_example(cluster_id, cluster_prompts, db):
                added += 1

        if added > 0:
            try:
                from src.db.connection import get_db
                from src.audit.logger import AuditLogger
                from src.audit.schemas import AuditEvent
                async with get_db() as audit_db:
                    await AuditLogger(audit_db).log(AuditEvent(
                        event_type="eval_set_updated",
                        decision=f"Eval factory added {added} examples from {len(candidates)} clusters",
                        rationale={
                            "clusters_processed": len(candidates),
                            "examples_added": added,
                            "total_prompts_clustered": len(recent_prompts),
                        },
                        state_snapshot={"examples_added": added},
                        operator="autonomous_pipeline",
                    ))
            except Exception:
                log.exception("eval_factory_audit_failed")

        await self.reset_counter()
        eval_factory_runs_total.inc()
        eval_factory_examples_added_total.inc(added)
        return added

    async def _try_generate_example(self, cluster_id: int, cluster_prompts: list[str], db) -> bool:
        prompt, embedding = self._clusterer.pick_medoid(cluster_prompts)

        if await self._eval_repo.exists_similar(
            embedding=embedding, threshold=settings.eval_factory_dedup_cosine_threshold
        ):
            eval_factory_examples_skipped_total.labels(reason="duplicate").inc()
            return False

        answers = []
        for _ in range(3):
            answer = await self._generate_answer(prompt)
            if answer is not None:
                answers.append(answer)
        if len(answers) < 2:
            eval_factory_examples_skipped_total.labels(reason="generation_failed").inc()
            return False

        confidence = self._compute_consistency(answers)
        if confidence < settings.eval_factory_min_confidence:
            eval_factory_examples_skipped_total.labels(reason="low_confidence").inc()
            return False

        best_answer = self._pick_best(answers)

        counts = await self._eval_repo.count_active_by_source()
        if sum(counts.values()) >= settings.eval_factory_max_eval_set_size:
            evicted = await self._eval_repo.evict_oldest(count=1)
            eval_factory_examples_evicted_total.inc(evicted)

        await self._eval_repo.insert_factory_example({
            "question": prompt,
            "context": "",
            "ground_truth": best_answer,
            "source": "factory",
            "cluster_id": cluster_id,
            "cluster_label": f"cluster_{cluster_id}",
            "factory_confidence": confidence,
            "embedding": embedding,
        })
        return True

    async def _generate_answer(self, prompt: str) -> str | None:
        system_prompt = (
            "You are an expert assistant. Answer the following question as accurately "
            "and completely as possible. Be concise. Do not explain your reasoning — "
            "just give the ideal answer a knowledgeable person would give."
        )
        try:
            response = await self._openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
                timeout=30.0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            log.exception("eval_factory_generation_error", prompt_preview=prompt[:80])
            return None

    def _compute_consistency(self, answers: list[str]) -> float:
        pairs = [(a, b) for i, a in enumerate(answers) for b in answers[i + 1:]]
        if not pairs:
            return 0.0
        scores = [_rouge_l(a, b) for a, b in pairs]
        return float(sum(scores) / len(scores))

    def _pick_best(self, answers: list[str]) -> str:
        if len(answers) == 1:
            return answers[0]
        best, best_mean = answers[0], -1.0
        for i, a in enumerate(answers):
            others = [b for j, b in enumerate(answers) if j != i]
            mean = sum(_rouge_l(a, b) for b in others) / len(others)
            if mean > best_mean:
                best, best_mean = a, mean
        return best


_factory_instance: EvalFactory | None = None


def get_eval_factory(eval_repo, openai_client, redis_client) -> EvalFactory:
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = EvalFactory(eval_repo, openai_client, redis_client)
    return _factory_instance
