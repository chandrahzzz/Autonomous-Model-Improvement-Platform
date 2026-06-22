"""
Dead-letter queue replayer (#I1).

Events that failed delivery after retries land in `pipeline.dlq`. Without a
consumer they accumulate forever and the pipeline silently loses LLM calls. This
replayer drains the DLQ in bounded batches, replays each message to its original
topic, and drops a message only after it exhausts `dlq_replay_max_attempts`
(tracked in the wrapped payload). It also publishes a `dlq_depth` gauge from the
consumer's offset lag so an accumulating DLQ is visible / alertable.
"""

import asyncio
import json

import structlog
from confluent_kafka import Consumer, TopicPartition

from src.config.settings import settings
from src.monitoring.metrics import (
    dlq_depth, dlq_replayed_total, dlq_replay_failed_total,
)

log = structlog.get_logger()


class DLQReplayer:
    def __init__(self, producer=None) -> None:
        self._producer = producer  # injectable for tests
        self._consumer: Consumer | None = None

    def _get_consumer(self) -> Consumer:
        if self._consumer is None:
            self._consumer = Consumer({
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": f"{settings.kafka_consumer_group}-dlq-replayer",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            })
            self._consumer.subscribe([settings.kafka_topic_dlq])
        return self._consumer

    async def _get_producer(self):
        if self._producer is None:
            from src.kafka.producer import get_producer
            self._producer = get_producer()
        return self._producer

    @staticmethod
    def _next_payload(raw: dict) -> tuple[str, dict, int]:
        """Unwrap a DLQ envelope → (original_topic, payload, attempts)."""
        original_topic = raw.get("original_topic", settings.kafka_topic_llm_events)
        payload = raw.get("payload", raw)
        attempts = int(raw.get("attempts", 0))
        return original_topic, payload, attempts

    async def replay_batch(self, max_messages: int = 100) -> tuple[int, int]:
        """Replay up to max_messages from the DLQ. Returns (replayed, dropped).
        A message that has already failed `dlq_replay_max_attempts` times is
        dropped (counted) rather than looped forever."""
        consumer = self._get_consumer()
        producer = await self._get_producer()
        loop = asyncio.get_running_loop()

        replayed = dropped = 0
        for _ in range(max_messages):
            msg = await loop.run_in_executor(None, lambda: consumer.poll(timeout=1.0))
            if msg is None:
                break
            if msg.error():
                continue
            try:
                raw = json.loads(msg.value().decode("utf-8"))
                original_topic, payload, attempts = self._next_payload(raw)
                if attempts >= settings.dlq_replay_max_attempts:
                    dlq_replay_failed_total.inc()
                    dropped += 1
                    log.error("dlq_message_dropped_max_attempts", topic=original_topic, attempts=attempts)
                else:
                    try:
                        await producer.produce(original_topic, payload)
                        dlq_replayed_total.inc()
                        replayed += 1
                    except Exception:
                        # Re-enqueue with an incremented attempt count for backoff.
                        await producer.produce(
                            settings.kafka_topic_dlq,
                            {"original_topic": original_topic, "payload": payload,
                             "attempts": attempts + 1},
                        )
                consumer.commit(message=msg, asynchronous=False)
            except Exception:
                log.exception("dlq_replay_message_failed")
        return replayed, dropped

    async def update_depth_gauge(self) -> int:
        """Publish dlq_depth from the consumer's offset lag (committed vs high
        watermark). Best-effort; returns the depth (or -1 on error)."""
        try:
            consumer = self._get_consumer()
            loop = asyncio.get_running_loop()

            def _lag() -> int:
                total = 0
                partitions = consumer.list_topics(settings.kafka_topic_dlq).topics[
                    settings.kafka_topic_dlq
                ].partitions
                for pid in partitions:
                    tp = TopicPartition(settings.kafka_topic_dlq, pid)
                    lo, hi = consumer.get_watermark_offsets(tp, timeout=5.0)
                    committed = consumer.committed([tp], timeout=5.0)[0].offset
                    pos = committed if committed and committed >= 0 else lo
                    total += max(0, hi - pos)
                return total

            depth = await loop.run_in_executor(None, _lag)
            dlq_depth.set(depth)
            return depth
        except Exception:
            log.warning("dlq_depth_probe_failed")
            return -1
