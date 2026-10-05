"""FastAPI application: startup, /health, and router registration.

    make run        # dev, auto-reload → http://localhost:8000/docs
    make run-prod   # production

At startup the app loads the saved model once (`make train` must have run);
it never trains.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from apex.api.dependencies import Service, create_service
from apex.api.routers import model, predict
from apex.api.schemas import Health
from apex.api.service import ScoringService


def create_app(service: ScoringService | None = None) -> FastAPI:
    """Build the app. Tests pass their own service; otherwise it is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or create_service()
        yield
        app.state.service.db.close()

    app = FastAPI(
        title="Aurynix Apex",
        description="Lead scoring: probability, priority segment, and reasons for each new lead.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=Health, tags=["health"])
    def health(service: Service) -> dict:
        return {"status": "ok", "model_version": service.version}

    app.include_router(model.router)
    app.include_router(predict.router)
    return app


app = create_app()
