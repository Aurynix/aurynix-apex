"""POST /predict/single, POST /predict/batch: validate → service → response."""

from fastapi import APIRouter

from apex.api.dependencies import Service
from apex.api.schemas import BatchPrediction, Lead, LeadBatch, Prediction

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/single", response_model=Prediction)
def predict_single(lead: Lead, service: Service) -> dict:
    """Score one lead: probability, High / Medium / Low segment, and the top reasons."""
    return service.score([lead.to_row()])[0]


@router.post("/batch", response_model=BatchPrediction)
def predict_batch(batch: LeadBatch, service: Service) -> dict:
    """Score a list of leads (up to 10,000); results keep the input order."""
    predictions = service.score([lead.to_row() for lead in batch.leads])
    return {"count": len(predictions), "predictions": predictions}
