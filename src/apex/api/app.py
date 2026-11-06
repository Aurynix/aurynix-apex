"""FastAPI application: startup, protection, /health, and router registration.

    make run        # dev, auto-reload → http://localhost:8000/docs
    make run-prod   # production

At startup the app loads the saved model once (`make train` must have run);
it never trains.
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from apex.api.database import log_rejection, purge_visitor_data
from apex.api.dependencies import Service, create_service, monitor_for
from apex.api.routers import model, monitoring, outcomes, predict
from apex.api.schemas import Health
from apex.api.security import (
    PROTECTED_PATHS,
    RateLimiter,
    Settings,
    client_address,
    has_valid_key,
    is_rate_limited_path,
)
from apex.api.service import ScoringService
from apex.monitoring.monitor import Monitor
from apex.monitoring.seed import refresh_on_schedule, seed_on_startup

logger = logging.getLogger("uvicorn.error")  # shown in the server log at INFO

HOUSEKEEPING_INTERVAL_SECONDS = 300


def housekeeping(app: FastAPI) -> None:
    """Remove old or excess visitor rows and keep demo rows fresh. Never fails a request."""
    state, settings = app.state, app.state.settings
    try:
        if settings.retention_days or settings.max_visitor_rows:
            removed = purge_visitor_data(
                state.service.db, settings.retention_days, settings.max_visitor_rows
            )
            if removed:
                logger.info("Removed %s old visitor predictions.", removed)
        refresh_on_schedule(state.service, state.monitor)
    except Exception:
        logger.exception("Housekeeping failed; the API continues.")


def create_app(service: ScoringService | None = None, monitor: Monitor | None = None) -> FastAPI:
    """Build the app. Tests pass their own service and monitor; otherwise they are
    loaded from the saved artifacts at startup."""
    settings = Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or create_service()
        app.state.monitor = monitor or monitor_for(app.state.service)
        seed_on_startup(app.state.service, app.state.monitor)
        housekeeping(app)
        app.state.next_housekeeping = time.monotonic() + HOUSEKEEPING_INTERVAL_SECONDS
        yield
        app.state.service.db.close()

    app = FastAPI(
        title="Aurynix Apex",
        description="Lead scoring: probability, priority segment, and reasons for each new lead.",
        version=version("aurynix-apex"),  # software version, from pyproject.toml
        lifespan=lifespan,
    )
    app.state.settings = settings
    limiter = (
        RateLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute else None
    )

    @app.middleware("http")
    async def protect(request: Request, call_next):
        """Limit, then check the key, before the body is read or anything is logged."""
        authorized = has_valid_key(request, settings)
        if limiter and not authorized and is_rate_limited_path(request):
            client = client_address(request, settings)
            if not limiter.allow(client):
                return JSONResponse(
                    {"detail": "Too many requests. Please try again in a minute."},
                    status_code=429,
                    headers={"Retry-After": str(limiter.retry_after(client))},
                )
        if settings.key_required and not authorized and request.url.path in PROTECTED_PATHS:
            if not settings.api_key:
                return JSONResponse(
                    {"detail": "API_KEY is not configured on the server."}, status_code=503
                )
            return JSONResponse({"detail": "Missing or invalid X-API-Key."}, status_code=401)

        response = await call_next(request)

        if request.method == "POST" and time.monotonic() >= request.app.state.next_housekeeping:
            request.app.state.next_housekeeping = time.monotonic() + HOUSEKEEPING_INTERVAL_SECONDS
            await run_in_threadpool(housekeeping, request.app)
        return response

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", response_model=Health, tags=["health"])
    def health(service: Service) -> dict:
        """Liveness only. It must not query the database: the host calls it constantly,
        which would keep a serverless database awake."""
        return {"status": "ok", "model_version": service.version}

    @app.get("/health/db", tags=["health"])
    def health_db(service: Service) -> JSONResponse:
        """Check the database connection. Call it by hand, not from an uptime monitor."""
        backend = service.db.backend
        try:
            service.db.ping()
        except Exception:
            logger.exception("Database check failed.")
            return JSONResponse({"status": "unavailable", "backend": backend}, status_code=503)
        return JSONResponse({"status": "ok", "backend": backend})

    @app.exception_handler(RequestValidationError)
    async def log_invalid_request(request: Request, exc: RequestValidationError):
        """Log rejected requests (data quality signal), then answer with the usual 422."""
        log_rejection(request.app.state.service.db, request.url.path, list(exc.errors()))
        return await request_validation_exception_handler(request, exc)

    app.include_router(model.router)
    app.include_router(predict.router)
    app.include_router(outcomes.router)
    app.include_router(monitoring.router)
    return app


app = create_app()
