"""
Pydantic schemas — request bodies & API responses.
"""

from __future__ import annotations
from typing import Literal, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import uuid


MediaType = Literal["image", "video", "audio"]
CaseStatus = Literal["processing", "authentic", "manipulated", "inconclusive", "failed"]


# ── Shared ────────────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    # `model_name` collides with pydantic's protected "model_" namespace and
    # warns on every import; the field name is part of the API contract, so
    # opt out of the check instead of renaming it.
    model_config = {"protected_namespaces": ()}

    id: str
    case_id: str
    model_name: str
    confidence: float
    label: str
    details: Optional[dict] = None
    created_at: datetime


# ── Case ──────────────────────────────────────────────────────────────────────

class CaseBase(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    media_type: MediaType
    user_id: Optional[str] = None
    notes: Optional[str] = None


class CaseCreate(CaseBase):
    """Body for POST /cases (form fields; file sent as multipart)."""
    pass


class CaseUpdate(BaseModel):
    """Body for PATCH /cases/{id}."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[CaseStatus] = None
    risk_score: Optional[float] = Field(None, ge=0, le=100)
    synthetic_likelihood: Optional[float] = Field(None, ge=0, le=100)
    notes: Optional[str] = None


class CaseResponse(BaseModel):
    """Shape the frontend expects (camelCase via alias)."""
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

    model_config = {"populate_by_name": True, "from_attributes": True}


class CaseListResponse(BaseModel):
    cases: List[CaseResponse]
    total: int
    page: int
    page_size: int


# ── Stats ─────────────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_cases: int = Field(alias="totalCases")
    authentic: int
    manipulated: int
    processing: int
    avg_risk_score: float = Field(alias="avgRiskScore")

    model_config = {"populate_by_name": True}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    db: str
