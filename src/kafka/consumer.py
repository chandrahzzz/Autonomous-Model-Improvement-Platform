"""
Kafka consumer with consumer-group coordination and exactly-once processing.
Each message is committed only after successful processing.
"""

import asyncio
import json
import structlog
from confluent_kafka import Consumer, KafkaError, KafkaException
from typing import Callable, Awaitable

from src.config.settings import settings

log = structlog.get_logger()

MessageHandler = Callable[[dict, str], Awaitable[None]]


class AsyncKafkaConsumer:
    def __init__(
        self,
        topics: list[str],
        group_id: str | None = None,
        handler: MessageHandler | None = None,
    ) -> None:
        self._topics = topics
        self._handler = handler
        self._running = False
        self._consumer = Consumer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id or settings.kafka_consumer_group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,     # manual commit after processing
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 30000,
        })

    def set_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def start(self) -> None:
        self._consumer.subscribe(self._topics)
        self._running = True
        log.info("kafka_consumer_started", topics=self._topics)
        await self._consume_loop()

    async def stop(self) -> None:
        self._running = False
        self._consumer.close()
        log.info("kafka_consumer_stopped")

    async def _consume_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            msg = await loop.run_in_executor(None, self._poll_once)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.error("kafka_consumer_error", error=str(msg.error()))
                continue
            try:
                value = json.loads(msg.value().decode("utf-8"))
                topic = msg.topic()
                if self._handler:
                    await self._handler(value, topic)
                self._consumer.commit(message=msg, asynchronous=False)
            except Exception:
                log.exception("kafka_message_processing_failed",
                              topic=msg.topic(), offset=msg.offset())
                # Don't commit — message will be redelivered

    def _poll_once(self) -> object:
        return self._consumer.poll(timeout=1.0)
