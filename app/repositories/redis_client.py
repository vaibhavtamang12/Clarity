"""Redis client factory. Central place where the Redis connection is built
so timeouts and credentials stay configuration-driven (Rule 3)."""

from __future__ import annotations

from redis.asyncio import Redis

from app.core.config import RedisSettings


def build_redis_client(settings: RedisSettings) -> Redis:
    return Redis.from_url(
        settings.url,
        socket_timeout=settings.timeout_seconds,
        socket_connect_timeout=settings.timeout_seconds,
        decode_responses=False,
    )