"""
Quality filter for curated training examples.

Three checks (all must pass):
1. ROUGE-L ≥ threshold — corrected completion is meaningfully different from bad one
2. Teacher confidence ≥ threshold (already checked in teacher.py)
3. Poison detection — corrected completion doesn't contain known attack patterns
"""

import re
import structlog
from rouge_score import rouge_scorer

from src.config.settings import settings

log = structlog.get_logger()

POISON_PATTERNS = [
    r"ignore previous instructions",
    r"disregard (all|your) (previous|prior|system)",
    r"you are now (an?|a) (different|new|evil|uncensored)",
    r"jailbreak",
    r"DAN mode",
    r"pretend you (have no|don'?t have) (restrictions|limitations|guidelines)",
]

POISON_RE = re.compile("|".join(POISON_PATTERNS), re.IGNORECASE)

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


class QualityFilter:
    def passes(
        self,
        prompt: str,
        bad_completion: str,
        corrected_completion: str,
        teacher_confidence: float,
    ) -> tuple[bool, float, str]:
        """
        Returns (passes, rouge_score, rejection_reason).
        rejection_reason is empty string if it passes.
        """
        # Poison check (fail closed)
        if POISON_RE.search(corrected_completion):
            log.warning("poison_detected_in_correction")
            return False, 0.0, "poison_detected"

        if POISON_RE.search(prompt):
            log.warning("poison_detected_in_prompt")
            return False, 0.0, "poison_in_prompt"

        # Teacher confidence
        if teacher_confidence < settings.teacher_confidence_threshold:
            return False, 0.0, "low_teacher_confidence"

        # ROUGE-L: correction must be meaningfully different from the bad output
        rouge = _scorer.score(bad_completion, corrected_completion)
        rouge_score = rouge["rougeL"].fmeasure

        # We want the correction to be DIFFERENT (not a copy of bad completion)
        # and also SUBSTANTIVE (not empty or trivially short)
        if rouge_score > 0.95:
            return False, rouge_score, "correction_too_similar_to_bad_output"

        if len(corrected_completion.split()) < 5:
            return False, rouge_score, "correction_too_short"

        # Quality score: blend of confidence and distinctiveness
        quality_score = teacher_confidence * (1.0 - rouge_score)

        if quality_score < settings.quality_rouge_threshold:
            return False, quality_score, "low_quality_score"

        return True, quality_score, ""
