"""
PII scrubbing via Microsoft Presidio.

Fail-closed: if scrubbing raises any exception, the example is DROPPED
(not passed through), controlled by settings.pii_fail_closed.

Supported entity types: PERSON, EMAIL_ADDRESS, PHONE_NUMBER,
CREDIT_CARD, US_SSN, IP_ADDRESS, LOCATION, DATE_TIME.
"""

import structlog
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.config.settings import settings

log = structlog.get_logger()

ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "US_SSN", "IP_ADDRESS",
    "LOCATION", "DATE_TIME", "NRP",
]


class PIIScrubber:
    def __init__(self) -> None:
        self._analyzer = AnalyzerEngine()
        self._anonymizer = AnonymizerEngine()
        self._operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in ENTITIES
        }

    def scrub(self, text: str) -> tuple[str, bool]:
        """
        Returns (scrubbed_text, was_modified).
        On failure: returns ("", False) and logs. Caller must drop the example
        if pii_fail_closed is True.
        """
        try:
            results = self._analyzer.analyze(text=text, entities=ENTITIES, language="en")
            if not results:
                return text, False
            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=self._operators,
            )
            return anonymized.text, True
        except Exception:
            log.exception("pii_scrub_failed")
            return "", False

    def scrub_example(self, prompt: str, completion: str) -> tuple[str, str, bool]:
        """
        Scrub both prompt and completion.
        Returns (prompt, completion, success). On failure returns ("","",False).
        """
        try:
            scrubbed_prompt, _ = self.scrub(prompt)
            scrubbed_completion, _ = self.scrub(completion)
            if not scrubbed_prompt or not scrubbed_completion:
                if settings.pii_fail_closed:
                    return "", "", False
            return scrubbed_prompt, scrubbed_completion, True
        except Exception:
            log.exception("pii_scrub_example_failed")
            if settings.pii_fail_closed:
                return "", "", False
            return prompt, completion, True
