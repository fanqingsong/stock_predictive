"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    ticker: str = Field(..., min_length=1)


class OkResponse(BaseModel):
    ok: bool = True


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str


class TrainCreateResponse(BaseModel):
    ok: bool = True
    job_id: int
    status: str
    reused: bool = False


class TrainStatusResponse(BaseModel):
    ok: bool = True
    job_id: int
    ticker: str
    market: str
    status: str
    progress: int
    message: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class ModelCard(BaseModel):
    ticker: str
    name: str = ""
    market: str
    market_label: str
    trained_at: Optional[str] = None
    feature_version: str = ""
    task_type: str = "classification"
    metrics: dict[str, Any] = Field(default_factory=dict)
    val_direction_acc: Optional[float] = None
    val_week_direction_acc: Optional[float] = None
    val_month_direction_acc: Optional[float] = None
    val_precision_up: Optional[float] = None
    val_recall_up: Optional[float] = None
    val_f1_up: Optional[float] = None
    val_mae: Optional[float] = None
    val_rmse: Optional[float] = None
    val_mape: Optional[float] = None
    train_direction_acc: Optional[float] = None
    predict_url: Optional[str] = None


class ModelsListResponse(BaseModel):
    ok: bool = True
    count: int
    items: list[ModelCard]


class ModelStatusResponse(BaseModel):
    ok: bool = True
    ticker: str
    market: str
    market_label: str
    ready: bool
    on_disk: bool
    trained_at: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    feature_version: Optional[str] = None


class SuggestItem(BaseModel):
    symbol: str = ""
    name: str = ""
    market: str = ""
    display: str = ""

    class Config:
        extra = "allow"


class SuggestResponse(BaseModel):
    query: str
    items: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str = "ok"
