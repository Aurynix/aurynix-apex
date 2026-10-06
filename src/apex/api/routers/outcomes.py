"""POST /outcomes: real results of scored leads, for performance monitoring."""

from fastapi import APIRouter, HTTPException

from apex.api.dependencies import Service
from apex.api.schemas import OutcomeBatch, OutcomesSaved

router = APIRouter(prefix="/outcomes", tags=["outcomes"])


@router.post("", response_model=OutcomesSaved)
def record_outcomes(batch: OutcomeBatch, service: Service) -> dict:
    """Record whether scored leads converted (by `prediction_id`).

    Sending an outcome again for the same prediction replaces it. If any id is
    unknown, nothing is saved and the unknown ids are returned (404).
    """
    outcomes = {o.prediction_id: o.converted for o in batch.outcomes}
    unknown = service.record_outcomes(outcomes)
    if unknown:
        raise HTTPException(404, {"message": "Unknown prediction_id", "unknown": unknown})
    return {"saved": len(outcomes)}
