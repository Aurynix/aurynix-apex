"""Create the ScoringService once at startup and hand it to the routers.

Startup only loads artifacts (models/model.pkl, models/model_meta.json) and
builds the explainer. It never trains: run `make train` first.
"""

from typing import Annotated

from fastapi import Depends, Request

from apex.api.database import connect
from apex.api.service import ScoringService
from apex.config import path
from apex.models.predict import load_model


def create_service() -> ScoringService:
    """Load the saved model and metadata, open the database, build the service."""
    model, meta = load_model()
    return ScoringService(model, meta, connect(path("database")))


def get_service(request: Request) -> ScoringService:
    """FastAPI dependency: the service created at startup."""
    return request.app.state.service


Service = Annotated[ScoringService, Depends(get_service)]
