"""Pydantic request and response models (validation only).

A lead is sent with simple snake_case fields; `to_row()` maps it to the raw
column names the model was trained on (`config.json → serving.input_fields`).
Missing optional fields are fine: cleaning fills them the same way as in training.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Lead(BaseModel):
    """One new lead, as known when it enters the queue (before any sales contact)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "lead_origin": "Landing Page Submission",
                "lead_source": "Google",
                "do_not_email": False,
                "total_visits": 3,
                "time_on_website": 1200,
                "specialization": "Finance Management",
                "occupation": "Working Professional",
            }
        },
    )

    lead_origin: str = Field(description="e.g. API, Landing Page Submission, Lead Add Form")
    lead_source: str | None = Field(None, description="e.g. Google, Direct Traffic, Reference")
    do_not_email: bool = Field(False, description="The lead opted out of emails")
    total_visits: int | None = Field(None, ge=0, description="Website visits")
    time_on_website: int = Field(0, ge=0, description="Total time on the website, in seconds")
    specialization: str | None = Field(None, description="Industry the lead worked in")
    occupation: str | None = Field(None, description="e.g. Working Professional, Student")

    def to_row(self) -> dict:
        """The lead with the raw column names the model expects.

        No time on the website means no visits: visits are only unknown (and filled
        by cleaning) when the lead did spend time on the site, as in the training data.
        """
        visits = self.total_visits
        if visits is None and self.time_on_website == 0:
            visits = 0
        return {
            "Lead Origin": self.lead_origin,
            "Lead Source": self.lead_source,
            "Do Not Email": "Yes" if self.do_not_email else "No",
            "TotalVisits": visits,
            "Total Time Spent on Website": self.time_on_website,
            "Specialization": self.specialization,
            "What is your current occupation": self.occupation,
        }


class LeadBatch(BaseModel):
    leads: list[Lead] = Field(min_length=1, max_length=10_000)


class Reasons(BaseModel):
    up: list[str] = Field(description="Strongest reasons that raise the score")
    down: list[str] = Field(description="Strongest reasons that lower the score")


class Prediction(BaseModel):
    prediction_id: int = Field(description="Send this id with the real result to POST /outcomes")
    score: float = Field(ge=0, le=1, description="Probability that the lead converts")
    segment: Literal["high", "medium", "low"]
    reasons: Reasons
    model_version: str


class BatchPrediction(BaseModel):
    count: int
    predictions: list[Prediction]


class Outcome(BaseModel):
    prediction_id: int = Field(ge=1)
    converted: bool = Field(description="Did the lead become a customer?")


class OutcomeBatch(BaseModel):
    outcomes: list[Outcome] = Field(min_length=1, max_length=10_000)


class OutcomesSaved(BaseModel):
    saved: int


class ModelInfo(BaseModel):
    model_version: str
    trained_at: str
    training_rows: int
    input_fields: list[str]
    segment_thresholds: dict[str, float]
    test_metrics: dict[str, float]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
