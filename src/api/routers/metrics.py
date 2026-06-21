from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

router = APIRouter()


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics/cost")
async def cost_summary() -> dict:
    """Current-month spend vs. budget."""
    from src.monitoring.cost_tracker import CostTracker
    return await CostTracker().get_monthly_summary()
