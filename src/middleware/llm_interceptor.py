"""
FastAPI middleware that intercepts every outgoing LLM call.
Captures prompt, completion, latency, tokens, cost, and model version.
Emits a structured Kafka event asynchronously — never blocks the response.
Target overhead: <5ms p99.
"""

import asyncio
import json
import time
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.kafka.producer import get_producer
from src.kafka.schemas.llm_event import LLMEvent
from src.config.settings import settings
from src.monitoring.metrics import (
    llm_calls_total,
    llm_latency_histogram,
    llm_tokens_counter,
    llm_cost_counter,
)

log = structlog.get_logger()

# Cost per token by model (USD)
TOKEN_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 5e-6, "output": 15e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.6e-6},
    "gpt-4-turbo": {"input": 10e-6, "output": 30e-6},
    "llama-3-8b": {"input": 0.05e-6, "output": 0.05e-6},
    "default": {"input": 0.05e-6, "output": 0.05e-6},
}


class LLMInterceptorMiddleware(BaseHTTPMiddleware):
    """
    Intercepts LLM completion endpoints and emits Kafka events.
    Only activates for requests tagged with X-LLM-Call: 1 or
    paths matching /chat or /completions.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._producer = get_producer()

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = int((time.monotonic() - start) * 1000)

        llm_latency_histogram.observe(latency_ms / 1000)

        if self._is_llm_endpoint(request):
            asyncio.create_task(self._emit_event(request, response, latency_ms))

        return response

    def _is_llm_endpoint(self, request: Request) -> bool:
        path = request.url.path
        return (
            request.headers.get("X-LLM-Call") == "1"
            or "/completions" in path
            or path.endswith("/chat")
        )

    async def _emit_event(
        self, request: Request, response: Response, latency_ms: int
    ) -> None:
        """Fire-and-forget. Logs on error but never raises."""
        try:
            body = self._parse_llm_metadata(request)
            model = body.get("model", "default")
            prompt_tokens = int(body.get("prompt_tokens", 0))
            completion_tokens = int(body.get("completion_tokens", 0))
            cost = self._compute_cost(model, prompt_tokens, completion_tokens)
            model_version = request.headers.get("X-Model-Version", "v7")

            event = LLMEvent(
                session_id=request.headers.get("X-Session-ID", "unknown"),
                user_cohort=request.headers.get("X-User-Cohort", "default"),
                model_version=model_version,
                prompt=request.headers.get("X-Prompt-Hash", ""),
                completion=body.get("content", ""),
                retrieved_context=body.get("retrieved_context", ""),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                finish_reason=body.get("finish_reason", "unknown"),
                cost_usd=cost,
                metadata={"model": model, "status_code": response.status_code},
            )

            await self._producer.produce(
                topic=settings.kafka_topic_llm_events,
                value=event.model_dump(),
                key=event.session_id,
            )

            llm_calls_total.labels(
                model_version=model_version,
                finish_reason=event.finish_reason,
            ).inc()
            llm_tokens_counter.labels(token_type="prompt").inc(prompt_tokens)
            llm_tokens_counter.labels(token_type="completion").inc(completion_tokens)
            llm_cost_counter.inc(cost)

            # Feed the shadow A/B window. Already off the request path (this runs
            # in a create_task), and no-ops after one Redis GET when no challenger
            # is in shadow, so serving latency is unaffected.
            from src.shadow.service import observe_production_call

            await observe_production_call(event.prompt, event.completion)

        except Exception:
            log.exception("llm_interceptor_emit_failed")

    def _parse_llm_metadata(self, request: Request) -> dict:
        """
        Extract LLM call metadata from request headers.
        In production you'd buffer the response body or use a custom header.
        We use headers here to keep middleware latency under 5ms.
        """
        raw = request.headers.get("X-LLM-Meta", "{}")
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}

    def _compute_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        costs = TOKEN_COSTS.get(model, TOKEN_COSTS["default"])
        return prompt_tokens * costs["input"] + completion_tokens * costs["output"]
