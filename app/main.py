"""Application entrypoint.

``create_app`` is a factory so tests can build isolated instances with
overridden settings. The module-level ``app`` is what uvicorn loads.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestIDMiddleware
from app.api.v1.router import api_v1_router
from app.core.config import Settings, get_settings
from app.core.logging import get_logger, setup_logging
from app.repositories.database import Database

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    database = Database(settings.database)
    await database.initialize()
    app.state.database = database

    # Lazy-connect Qdrant client: construction opens no sockets, so startup
    # succeeds even while Qdrant is still booting (same posture as Database).
    qdrant_client = build_qdrant_client(settings.qdrant)
    app.state.qdrant_client = qdrant_client

    logger.info(
        "application_started",
        service=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment.value,
        database_host=settings.database.host,
        qdrant_host=settings.qdrant.host,
    )

    yield

    await qdrant_client.close()
    await database.dispose()
    logger.info("application_stopped")
    
def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(level=settings.app.log_level, json_output=settings.app.log_json)

    app = FastAPI(
        title="RAG Knowledge Intelligence Platform",
        version=settings.app.version,
        debug=settings.app.debug,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(RequestIDMiddleware)
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
        """Liveness probe for orchestrators — no dependencies touched."""
        return {"status": "ok"}

    return app


app = create_app()