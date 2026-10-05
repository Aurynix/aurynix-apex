"""GET /model/info: version, training date, thresholds, and test metrics of the loaded model."""

from fastapi import APIRouter

from apex.api.dependencies import Service
from apex.api.schemas import ModelInfo

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info", response_model=ModelInfo)
def model_info(service: Service) -> dict:
    return service.info()
