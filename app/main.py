"""Application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import MetricsMiddleware, RequestIDMiddleware
from app.api.v1.router import api_v1_router
from app.container import build_platform
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.repositories.database import Database
from app.repositories.vector.qdrant_client import build_qdrant_client

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    platform_factory: Callable | None = None,   # DI seam for tests (D-083)
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(level=settings.app.log_level, json_output=settings.app.log_json)

    app = FastAPI(
        title="RAG Knowledge Intelligence Platform",
        version=settings.app.version,
        debug=settings.app.debug,
    )
    app.state.settings = settings
    app.state.metrics = {
        "requests_total": 0,
        "client_errors_total": 0,
        "server_errors_total": 0,
    }

    factory = platform_factory or (
        lambda s, db, qdrant_client=None: build_platform(s, db, qdrant_client=qdrant_client)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = Database(settings.database)
        await database.initialize()
        app.state.database = database

        qdrant_client = build_qdrant_client(settings.qdrant)
        app.state.qdrant_client = qdrant_client

        app.state.platform = factory(settings, database, qdrant_client)

        logger.info(
            "application_started",
            service=settings.app.name,
            version=settings.app.version,
            environment=settings.app.environment.value,
        )
        yield

        await qdrant_client.close()
        await database.dispose()
        logger.info("application_stopped")

    app.router.lifespan_context = lifespan

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MetricsMiddleware)
    if settings.app.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.app.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(api_v1_router, prefix=settings.app.api_v1_prefix)
    register_exception_handlers(app)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()