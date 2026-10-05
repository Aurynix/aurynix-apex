"""Drift monitoring endpoints: validate → monitor → response."""

from fastapi import APIRouter, HTTPException, Query

from apex.api.dependencies import MonitorDep

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/run")
def run_monitoring(
    monitor: MonitorDep, window_days: int | None = Query(None, ge=1, le=365)
) -> dict:
    """Check recent predictions for feature drift, prediction drift, and data quality."""
    return monitor.run(window_days)


@router.get("/latest")
def latest(monitor: MonitorDep) -> dict:
    """The most recent monitoring report."""
    runs = monitor.history(limit=1)
    if not runs:
        raise HTTPException(404, "No monitoring run yet. POST /monitoring/run first.")
    return runs[0]


@router.get("/history")
def history(monitor: MonitorDep, limit: int = Query(30, ge=1, le=365)) -> list[dict]:
    """Status and key numbers of past runs, newest first."""
    return [
        {
            "id": run["id"],
            "created_at": run["created_at"],
            "status": run["status"],
            "n_samples": run["n_samples"],
            "score_psi": run.get("prediction", {}).get("score_psi"),
            "segment_shares": run.get("prediction", {}).get("segment_shares", {}).get("current"),
        }
        for run in monitor.history(limit)
    ]
