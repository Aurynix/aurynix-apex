"""FastAPI application: startup, /health, and router registration.

    make run        # dev, auto-reload → http://localhost:8000/docs
    make run-prod   # production

At startup the app loads the saved model once (`make train` must have run);
it never trains.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from apex.api.database import log_rejection
from apex.api.dependencies import Service, create_service, monitor_for
from apex.api.routers import model, monitoring, outcomes, predict
from apex.api.schemas import Health
from apex.api.service import ScoringService
from apex.monitoring.monitor import Monitor


def create_app(service: ScoringService | None = None, monitor: Monitor | None = None) -> FastAPI:
    """Build the app. Tests pass their own service and monitor; otherwise they are
    loaded from the saved artifacts at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or create_service()
        app.state.monitor = monitor or monitor_for(app.state.service)
        yield
        app.state.service.db.close()

    app = FastAPI(
        title="Aurynix Apex",
        description="Lead scoring: probability, priority segment, and reasons for each new lead.",
        version=version("aurynix-apex"),  # software version, from pyproject.toml
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health, tags=["health"])
    def health(service: Service) -> dict:
        return {"status": "ok", "model_version": service.version}

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
