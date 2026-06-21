"""
Kafka topic definitions. All topic names, partition counts, and retention
config are defined here as the single source of truth.
"""

from dataclasses import dataclass
from src.config.settings import settings


@dataclass(frozen=True)
class TopicConfig:
    name: str
    num_partitions: int
    replication_factor: int
    retention_ms: int   # milliseconds


TOPIC_LLM_EVENTS = TopicConfig(
    name=settings.kafka_topic_llm_events,
    num_partitions=12,
    replication_factor=1,
    retention_ms=7 * 24 * 3600 * 1000,  # 7 days
)

TOPIC_TRAINING_EVENTS = TopicConfig(
    name=settings.kafka_topic_training_events,
    num_partitions=3,
    replication_factor=1,
    retention_ms=30 * 24 * 3600 * 1000,  # 30 days
)

TOPIC_DLQ = TopicConfig(
    name=settings.kafka_topic_dlq,
    num_partitions=3,
    replication_factor=1,
    retention_ms=30 * 24 * 3600 * 1000,  # 30 days
)

ALL_TOPICS = [TOPIC_LLM_EVENTS, TOPIC_TRAINING_EVENTS, TOPIC_DLQ]


def create_topics(bootstrap_servers: str) -> None:
    """Create all required Kafka topics. Safe to call multiple times."""
    from confluent_kafka.admin import AdminClient, NewTopic
    import structlog

    log = structlog.get_logger()
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})

    new_topics = [
        NewTopic(
            t.name,
            num_partitions=t.num_partitions,
            replication_factor=t.replication_factor,
            config={"retention.ms": str(t.retention_ms)},
        )
        for t in ALL_TOPICS
    ]

    futures = admin.create_topics(new_topics)
    for topic_name, future in futures.items():
        try:
            future.result()
            log.info("kafka_topic_created", topic=topic_name)
        except Exception as e:
            if "already exists" in str(e).lower() or "topic_already_exists" in str(type(e).__name__).lower():
                log.debug("kafka_topic_already_exists", topic=topic_name)
            else:
                log.error("kafka_topic_creation_failed", topic=topic_name, error=str(e))
                raise
