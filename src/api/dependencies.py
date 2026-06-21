"""
FastAPI dependency injection — DB session, Redis, Kafka producer.
"""

from typing import AsyncIterator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from src.db.connection import AsyncSessionLocal
from src.kafka.producer import get_producer, AsyncKafkaProducer
from src.config.settings import settings

_redis_pool: aioredis.Redis | None = None


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            str(settings.redis_url),
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
    return _redis_pool


def get_kafka_producer() -> AsyncKafkaProducer:
    return get_producer()
