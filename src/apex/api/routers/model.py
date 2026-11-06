"""GET /model/info: version, training date, thresholds, and test metrics of the loaded model."""

from fastapi import APIRouter, Request

from apex.api.dependencies import Service
from apex.api.schemas import ModelInfo

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfo)
def model_info(service: Service, request: Request) -> dict:
    return {**service.info(), "max_batch_size": request.app.state.settings.max_batch_size}
