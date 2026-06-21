"""
MinHash LSH deduplication at scale.

Uses datasketch MinHash with Jaccard threshold 0.85 so near-duplicate
examples (e.g., same failure with minor wording change) are filtered
before they enter the training set.
"""

import hashlib

import structlog
from datasketch import MinHash, MinHashLSH

from src.config.settings import settings

log = structlog.get_logger()

NUM_PERM = 128


def _text_to_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    for word in text.lower().split():
        m.update(word.encode("utf-8"))
    return m


def _dedup_hash(prompt: str, completion: str) -> str:
    """Deterministic SHA-256 hash for exact-match dedup (DB unique index)."""
    combined = f"{prompt}\x00{completion}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class Deduplicator:
    def __init__(self) -> None:
        self._lsh = MinHashLSH(
            threshold=settings.dedup_jaccard_threshold,
            num_perm=NUM_PERM,
        )
        self._seen: set[str] = set()

    def is_duplicate(self, prompt: str, completion: str) -> bool:
        """
        Returns True if this (prompt, completion) pair is a near-duplicate
        of something already seen. Uses MinHash LSH for approximate matching.
        """
        text = f"{prompt} {completion}"
        minhash = _text_to_minhash(text)

        exact_hash = _dedup_hash(prompt, completion)
        if exact_hash in self._seen:
            return True

        duplicates = self._lsh.query(minhash)
        if duplicates:
            log.debug("near_duplicate_found", n_matches=len(duplicates))
            return True

        # Register in index
        key = exact_hash
        self._lsh.insert(key, minhash)
        self._seen.add(exact_hash)
        return False

    def compute_hash(self, prompt: str, completion: str) -> str:
        return _dedup_hash(prompt, completion)
