from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import structlog

from src.config.settings import settings
from src.config.logging import configure_logging
from src.db.connection import engine, check_database_health
from src.kafka.producer import get_producer
from src.middleware.llm_interceptor import LLMInterceptorMiddleware
from src.api.routers import health, metrics, pipeline, models, audit, shadow, training, drift, eval, attribution, knowledge

configure_logging()
log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("pipeline_api_starting", environment=settings.environment)

    if not await check_database_health():
        raise RuntimeError("Database unreachable at startup — aborting")

    log.info("pipeline_api_ready")
    yield

    log.info("pipeline_api_shutting_down")
    producer = get_producer()
    await producer.flush()
    await engine.dispose()
    log.info("pipeline_api_shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Continuous Fine-Tuning Pipeline API",
        version="1.0.0",
        description="Autonomous LLM fine-tuning with zero human intervention",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else [],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.add_middleware(LLMInterceptorMiddleware)

    app.include_router(health.router, tags=["Health"])
    app.include_router(metrics.router, tags=["Metrics"])
    app.include_router(pipeline.router, prefix="/pipeline", tags=["Pipeline"])
    app.include_router(models.router, prefix="/models", tags=["Models"])
    app.include_router(audit.router, prefix="/audit", tags=["Audit"])
    app.include_router(shadow.router, prefix="/shadow", tags=["Shadow"])
    app.include_router(training.router, prefix="/training", tags=["Training"])
    app.include_router(drift.router, prefix="/drift", tags=["Drift"])
    app.include_router(eval.router, prefix="/eval", tags=["Eval"])
    app.include_router(attribution.router, prefix="/attribution", tags=["Attribution"])
    app.include_router(knowledge.router, prefix="/knowledge", tags=["Knowledge"])

    return app


app = create_app()
