"""A/B test node: manages shadow traffic window."""
import structlog

from src.graph.state import PipelineState
from src.db.connection import AsyncSessionLocal
from src.shadow.ab_collector import ABCollector

log = structlog.get_logger()


async def ab_test_node(state: PipelineState) -> PipelineState:
    version_tag = state.get("version_tag", "unknown")

    # Register the challenger as the active shadow target. Nothing did this
    # before, so the router never had a version to shadow against and
    # shadow_logs stayed empty — the window could never fill. set_challenger is
    # idempotent and stamps the window start only once.
    started_at = None
    try:
        from src.shadow.service import _get_router

        router = _get_router()
        if await router.get_challenger_version() != version_tag:
            await router.set_challenger(version_tag)
        started_at = await router.get_started_at()
    except Exception:
        log.warning("shadow_challenger_registration_failed", version=version_tag)

    async with AsyncSessionLocal() as db:
        collector = ABCollector(db)
        ab_data = await collector.collect_window(version_tag, started_at=started_at)

    # Force a decision once the window times out, otherwise a challenger that
    # never reaches ab_min_requests keeps this node cycling to END forever and
    # the pipeline can neither promote nor roll back. The promotion gate rejects
    # a timed-out window on its own (ready is False), so this only unblocks the
    # graph — it never turns a starved window into a promotion.
    ready_for_decision = bool(ab_data["ready"] or ab_data["timed_out"])
    if ab_data["timed_out"]:
        log.warning(
            "shadow_window_timed_out",
            version=version_tag,
            n_requests=ab_data["n_requests"],
            elapsed_hours=round(ab_data["elapsed_hours"], 2),
        )

    log.info(
        "ab_test_node_complete",
        version=version_tag,
        ready=ab_data["ready"],
        timed_out=ab_data["timed_out"],
        n=ab_data["n_requests"],
        elapsed_hours=round(ab_data["elapsed_hours"], 2),
    )
    return {
        **state,
        "shadow_active": True,
        "shadow_ready_for_decision": ready_for_decision,
        "ab_data": ab_data,
    }
