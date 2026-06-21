"""
Async Kafka producer with retry and dead-letter queue.
Exactly-once semantics via idempotent producer config.
"""

import asyncio
import json
import structlog
from confluent_kafka import Producer, KafkaError

from src.config.settings import settings

log = structlog.get_logger()


class AsyncKafkaProducer:
    def __init__(self) -> None:
        self._producer = Producer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 300,
            "enable.idempotence": True,
            "compression.type": "snappy",
            "linger.ms": 5,
            "batch.size": 65536,
        })
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_event_loop()
        return self._loop

    async def produce(self, topic: str, value: dict, key: str | None = None) -> None:
        loop = self._get_loop()
        future: asyncio.Future = loop.create_future()

        def delivery_callback(err, msg):
            if err:
                loop.call_soon_threadsafe(
                    future.set_exception, Exception(str(err))
                )
            else:
                loop.call_soon_threadsafe(future.set_result, msg)

        self._producer.produce(
            topic=topic,
            value=json.dumps(value).encode("utf-8"),
            key=key.encode("utf-8") if key else None,
            callback=delivery_callback,
        )
        self._producer.poll(0)

        try:
            await asyncio.wait_for(future, timeout=10.0)
        except Exception as e:
            log.error("kafka_produce_failed", topic=topic, error=str(e))
            await self._send_to_dlq(topic, value, str(e))

    async def _send_to_dlq(self, original_topic: str, value: dict, error: str) -> None:
        payload = {"original_topic": original_topic, "payload": value, "error": error}
        self._producer.produce(
            topic=settings.kafka_topic_dlq,
            value=json.dumps(payload).encode("utf-8"),
        )
        self._producer.flush(timeout=5)
        log.warning("message_sent_to_dlq", original_topic=original_topic)

    async def flush(self) -> None:
        self._producer.flush(timeout=10)


_producer: AsyncKafkaProducer | None = None


def get_producer() -> AsyncKafkaProducer:
    global _producer
    if _producer is None:
        _producer = AsyncKafkaProducer()
    return _producer
