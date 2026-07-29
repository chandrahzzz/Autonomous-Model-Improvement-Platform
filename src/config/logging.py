import structlog
import logging
import sys

from src.config.settings import settings


def configure_logging() -> None:
    log_level = logging.DEBUG if settings.debug else logging.INFO

    # Windows consoles default to a legacy code page (cp1252) that can't encode
    # non-Latin-1 characters. Log lines routinely contain them (e.g. "Brasília",
    # em dashes, model output), and the console renderer would raise
    # UnicodeEncodeError mid-log and crash the whole runner. Force stdout/stderr
    # to UTF-8 so logging can never take the process down.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to go through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
