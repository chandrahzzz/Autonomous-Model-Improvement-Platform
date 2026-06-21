"""
Pipeline runner: starts the perpetual LangGraph loop.

The graph never stops. Each iteration:
  1. Checks for new LLM logs
  2. Detects failures
  3. Curates training data
  4. Triggers training when thresholds met
  5. Evaluates challenger
  6. Runs shadow A/B test
  7. Promotes or rolls back
  8. Sleeps 60 seconds
  9. Repeats forever

Run with: python -m src.graph.runner
"""

import asyncio
import json
import signal
import sys
import time
import structlog
import redis.asyncio as aioredis

from src.config.logging import configure_logging
from src.config.settings import settings
from src.graph.graph import compile_graph
from src.monitoring.metrics import pipeline_cycle_duration, pipeline_errors_total

configure_logging()
log = structlog.get_logger()

CYCLE_SLEEP_SECONDS = 60
THREAD_ID = "pipeline-main"
# Phase-aware cycle intervals. The detection loop needs to be fast, but during
# a 48h shadow A/B window polling every 60s is ~2,880 pointless wake-ups, so we
# back off substantially while waiting on slow external work.
CYCLE_INTERVALS = {
    "monitoring": 60,     # fast detection loop
    "training": 300,      # 5 min — just checking if the Modal job finished
    "evaluating": 120,    # 2 min — eval is quick, check fairly often
    "ab_testing": 1800,   # 30 min — the A/B window spans ~48h
    "canary": 300,        # 5 min — the canary window spans ~1h
    "promoting": 60,      # promotion should be near-immediate
}
# Durable snapshot of the full pipeline state. Unlike `pipeline:state` (a
# short-TTL summary the API reads), this persists across restarts so the runner
# resumes an in-flight training job / shadow test instead of resetting to the
# hardcoded defaults below.
FULL_STATE_KEY = "pipeline:full_state"


class PipelineRunner:
    def __init__(self) -> None:
        self._graph = compile_graph()
        self._running = False
        self._state: dict = {
            "cycles_completed": 0,
            "paused": False,
            "production_version": "v7",
            "incumbent_scores": {
                "faithfulness": 0.70,
                "answer_relevancy": 0.72,
                "context_recall": 0.68,
            },
        }

    async def _rehydrate_state(self) -> None:
        """Restore the last persisted pipeline state on startup.

        Without this, a restart re-initialises `self._state` to hardcoded
        defaults (production_version="v7", no version_tag), orphaning any Modal
        job or shadow test that was running when the process died.
        """
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            raw = await r.get(FULL_STATE_KEY)
            await r.aclose()
        except Exception:
            log.warning("pipeline_state_rehydrate_failed")
            return

        if not raw:
            log.info("pipeline_state_no_snapshot", note="starting from defaults")
            return
        try:
            saved = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("pipeline_state_rehydrate_parse_failed")
            return
        self._state = {**self._state, **saved}
        log.info(
            "pipeline_state_rehydrated",
            cycle=self._state.get("cycles_completed", 0),
            version=self._state.get("production_version"),
            training_status=self._state.get("training_status"),
            shadow_active=self._state.get("shadow_active", False),
        )

    async def _persist_full_state(self) -> None:
        """Persist the full state snapshot (no TTL) for restart recovery."""
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            await r.set(FULL_STATE_KEY, json.dumps(self._state, default=str))
            await r.aclose()
        except Exception:
            log.warning("pipeline_full_state_persist_failed")

    async def run_forever(self) -> None:
        self._running = True
        log.info("pipeline_runner_starting", environment=settings.environment)
        await self._rehydrate_state()

        # Register graceful shutdown handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown)

        while self._running:
            cycle_start = time.monotonic()
            try:
                await self._run_cycle()
            except Exception:
                log.exception("pipeline_cycle_error")
                pipeline_errors_total.labels(node="runner").inc()

            elapsed = time.monotonic() - cycle_start
            pipeline_cycle_duration.observe(elapsed)

            if self._running:
                interval = CYCLE_INTERVALS.get(self._current_phase(), CYCLE_SLEEP_SECONDS)
                sleep_for = max(0, interval - elapsed)
                log.debug("pipeline_cycle_sleeping", seconds=sleep_for, phase=self._current_phase())
                await asyncio.sleep(sleep_for)

        log.info("pipeline_runner_stopped")

    def _current_phase(self) -> str:
        """Derive the pipeline phase from the current state so the runner can
        pace itself. Mirrors the graph's entry-point routing rather than relying
        on every node to set a phase field."""
        s = self._state
        if s.get("modal_job_id") and s.get("training_status") in ("submitted", "running"):
            return "training"
        if s.get("shadow_active") and not s.get("shadow_ready_for_decision"):
            return "ab_testing"
        if s.get("canary_active") and s.get("canary_decision") == "pending":
            return "canary"
        if s.get("shadow_ready_for_decision") or s.get("promotion_decision") is not None:
            return "promoting"
        if s.get("training_status") == "completed":
            return "evaluating"
        return "monitoring"

    async def _run_cycle(self) -> None:
        log.info(
            "pipeline_cycle_starting",
            cycle=self._state.get("cycles_completed", 0),
        )
        # Cost circuit breaker: if the monthly budget is blown, skip the entire
        # cycle so no curation (teacher) or training (GPU) spend occurs. Fail-safe
        # is to stop spending; it resumes automatically when the month resets.
        from src.monitoring.cost_tracker import CostTracker
        within_budget, budget_msg = await CostTracker().check_budget()
        if not within_budget:
            self._state["paused"] = True
            self._state["paused_reason"] = "budget"
            log.error("pipeline_cycle_skipped_over_budget", detail=budget_msg)
            await self._persist_full_state()
            await self._publish_state()
            return
        if self._state.get("paused_reason") == "budget":
            self._state["paused"] = False
            self._state["paused_reason"] = None

        result = await self._graph.ainvoke(
            self._state,
            config={"configurable": {"thread_id": THREAD_ID}},
        )
        # Merge result back into persistent state
        self._state = {**self._state, **result}
        log.info(
            "pipeline_cycle_complete",
            cycle=self._state.get("cycles_completed", 0),
            version=self._state.get("production_version"),
        )
        await self._persist_full_state()
        await self._publish_state()
        await self._maybe_calibrate()

    async def _maybe_calibrate(self) -> None:
        """Periodically suggest detection-threshold adjustments from observed data."""
        interval = settings.calibration_interval_cycles
        cycle = self._state.get("cycles_completed", 0)
        if interval <= 0 or cycle == 0 or cycle % interval != 0:
            return
        try:
            from src.db.connection import get_db
            from src.detection.calibrator import ThresholdCalibrator
            async with get_db() as db:
                await ThresholdCalibrator().run(db)
        except Exception:
            log.warning("threshold_calibration_failed")

    async def _publish_state(self) -> None:
        """Write a summary of the current pipeline state to Redis for the API to read."""
        summary = {
            "cycles_completed": self._state.get("cycles_completed", 0),
            "production_version": self._state.get("production_version"),
            "pipeline_phase": self._current_phase(),
            "last_cycle_at": self._state.get("last_cycle_at"),
            "pending_examples": self._state.get("pending_examples", 0),
            "drift_score": self._state.get("drift_score"),
            "drift_slope": self._state.get("drift_slope"),
            "drift_predicted_trigger_hours": self._state.get("drift_predicted_trigger_hours"),
            "drift_is_alarming": self._state.get("drift_is_alarming"),
            "drift_trend_direction": self._state.get("drift_trend_direction"),
            "has_failures": self._state.get("has_failures", False),
            "failure_count": self._state.get("failure_count", 0),
            "training_triggered": self._state.get("training_triggered", False),
            "training_status": self._state.get("training_status"),
            "eval_passed": self._state.get("eval_passed"),
            "shadow_active": self._state.get("shadow_active", False),
            "shadow_ready_for_decision": self._state.get("shadow_ready_for_decision", False),
            "promotion_decision": self._state.get("promotion_decision"),
            "incumbent_scores": self._state.get("incumbent_scores", {}),
            "eval_set_size": self._state.get("eval_set_size"),
            "eval_factory_last_run_at": self._state.get("eval_factory_last_run_at"),
            "eval_factory_examples_added_last_run": self._state.get("eval_factory_examples_added_last_run"),
            "attribution_count_this_cycle": self._state.get("attribution_count_this_cycle"),
            "last_attribution_run_at": self._state.get("last_attribution_run_at"),
            "error": self._state.get("error"),
        }
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            await r.set("pipeline:state", json.dumps(summary, default=str), ex=300)
            await r.aclose()
        except Exception:
            log.warning("pipeline_state_redis_publish_failed")

    def _shutdown(self) -> None:
        log.info("pipeline_runner_shutdown_signal_received")
        self._running = False


async def main() -> None:
    runner = PipelineRunner()
    await runner.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
