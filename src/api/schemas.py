"""Pydantic response models for the Autiz API. Field meanings: PROJECT_STATUS.md section 6a."""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class AcousticObservations(BaseModel):
    pitch_variation_stddev_norm: Optional[float] = None
    mean_unvoiced_segment_sec: Optional[float] = None
    voiced_segments_per_sec: Optional[float] = None


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class Explanation(BaseModel):
    method: Literal["shap", "proxy"]
    explains: Literal["prosody_only_baseline_score"]
    top_features: list[FeatureContribution]
    note: str


class Session(BaseModel):
    speaker_id: str
    system_stage: str
    candidate_disclosure_required: bool


# content_score is clipped to [0, 1] by the pipeline, but content_score_raw is not, so no ge/le here.
class AnalyzeResponse(BaseModel):
    content_score: float
    content_score_raw: float
    content_score_source: Literal["trained_head", "proxy"]
    content_scorer_stamp: Optional[str] = None
    content_label_source: Optional[str] = None
    intent_label: None = None
    intent_status: Literal["not_trained"]
    prosody_only_baseline_score: float
    prosody_only_baseline_label: str
    transcript: str
    acoustic_observations: AcousticObservations
    delivery_pattern: list[str]
    explanation: Explanation
    interpretation: str
    interpretation_source: Literal["template"]
    mode: str
    mode_effect: str
    session: Session
    smoke_artifacts: bool


class ScoreResponse(BaseModel):
    content_score: float
    content_score_raw: float
    content_score_source: Literal["trained_head", "proxy"]
    content_scorer_stamp: Optional[str] = None
    mode: str
    mode_effect: str
    system_stage: str
    candidate_disclosure_required: bool
    smoke_artifacts: bool


class AboutResponse(BaseModel):
    name: str
    system_stage: str
    stores_nothing: bool
    content_label_source: Optional[str] = None
    content_label_note: str
    results_available: bool
    independence_results: Optional[dict[str, Any]] = None
    results_note: Optional[str] = None
    limitations: list[str]
    smoke_artifacts: bool


class HealthResponse(BaseModel):
    status: str
    system_stage: str
    smoke_artifacts: bool
