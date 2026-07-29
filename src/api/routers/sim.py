"""
No-op endpoint for the synthetic traffic simulator's `--mode http`.

The real work is done by LLMInterceptorMiddleware, which captures any request
tagged `X-LLM-Call: 1` and emits an LLMEvent from the request headers. This
handler just needs to exist and return 200 so the middleware has a response to
observe. Mounted only when `settings.sim_traffic_enabled` is True — it is a
test/demo affordance and must never be enabled in production.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/sim/llm-call")
async def sim_llm_call() -> dict:
    # The interceptor reads everything it needs from the request headers
    # (X-LLM-Meta, X-Prompt-Hash, X-Session-ID, ...). The body is ignored.
    return {"status": "ok", "note": "synthetic sim traffic captured by middleware"}
