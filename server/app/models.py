from __future__ import annotations
from typing import Literal, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import uuid


MediaType = Literal["image", "video", "audio"]
CaseStatus = Literal["processing", "authentic", "manipulated", "inconclusive", "failed"]


class AnalysisResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    id: str
    case_id: str
    model_name: str
    confidence: float
    label: str
    details: Optional[dict] = None
    created_at: datetime


class CaseBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    media_type: MediaType
    user_id: Optional[str] = None
    notes: Optional[str] = None


class CaseCreate(CaseBase):
    pass


class CaseUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[CaseStatus] = None
    risk_score: Optional[float] = Field(None, ge=0, le=100)
    synthetic_likelihood: Optional[float] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


class JobState(BaseModel):
    state: Literal["queued", "running", "retrying", "succeeded", "failed", "cached"]
    case_id: Optional[str] = Field(None, alias="caseId")
    media_type: Optional[str] = Field(None, alias="mediaType")
    position: Optional[int] = None
    attempt: int = 0
    max_attempts: int = Field(0, alias="maxAttempts")
    worker: Optional[str] = None
    queued_at: Optional[float] = Field(None, alias="queuedAt")
    started_at: Optional[float] = Field(None, alias="startedAt")
    finished_at: Optional[float] = Field(None, alias="finishedAt")
    retry_at: Optional[float] = Field(None, alias="retryAt")
    error: Optional[str] = None
    cache_hit: bool = Field(False, alias="cacheHit")
    content_hash: Optional[str] = Field(None, alias="contentHash")
    source_url: Optional[str] = Field(None, alias="sourceUrl")

    model_config = {"populate_by_name": True}


class CaseResponse(BaseModel):
    id: str
    case_id: str = Field(alias="caseId")
    title: str
    media_type: MediaType = Field(alias="mediaType")
    status: CaseStatus
    risk_score: float = Field(alias="riskScore")
    synthetic_likelihood: float = Field(alias="syntheticLikelihood")
    file_name: Optional[str] = Field(None, alias="fileName")
    file_url: Optional[str] = Field(None, alias="fileUrl")
    file_size: Optional[int] = Field(None, alias="fileSize")
    user_id: Optional[str] = Field(None, alias="userId")
    notes: Optional[str] = None
    created_at: Optional[datetime] = Field(None, alias="createdAt")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")
    analysis_results: List[AnalysisResult] = Field(default_factory=list, alias="analysisResults")
    job: Optional[JobState] = None

    model_config = {"populate_by_name": True, "from_attributes": True}


class CaseListResponse(BaseModel):
    cases: List[CaseResponse]
    total: int
    page: int
    page_size: int


class DashboardStats(BaseModel):
    total_cases: int = Field(alias="totalCases")
    authentic: int
    manipulated: int
    processing: int
    inconclusive: int = 0
    failed: int = 0
    by_media_type: dict[str, int] = Field(default_factory=dict, alias="byMediaType")
    avg_risk_score: float = Field(alias="avgRiskScore")

    model_config = {"populate_by_name": True}


class HealthResponse(BaseModel):
    status: str
    version: str
    db: str
    redis: str = "unknown"
    workers: int = 0
