"""ORM models for train jobs and model artifacts."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .database import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")


class StockModelArtifact(Base):
    __tablename__ = "stock_model_artifacts"
    __table_args__ = (UniqueConstraint("ticker", "market", name="uq_artifact_ticker_market"),)

    STATUS_READY = "ready"
    STATUS_STALE = "stale"
    STATUS_MISSING = "missing"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    market: Mapped[str] = mapped_column(String(32), index=True)
    display: Mapped[str] = mapped_column(String(64), default="", server_default="")
    artifact_dir: Mapped[str] = mapped_column(String(512), default="", server_default="")
    status: Mapped[str] = mapped_column(String(16), default=STATUS_MISSING, server_default=STATUS_MISSING)
    metrics_json: Mapped[dict] = mapped_column(JSONType, default=dict)
    feature_version: Mapped[str] = mapped_column(String(32), default="", server_default="")
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TrainJob(Base):
    __tablename__ = "train_jobs"

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_SUCCEEDED = "succeeded"
    STATUS_FAILED = "failed"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    market: Mapped[str] = mapped_column(String(32), index=True)
    display: Mapped[str] = mapped_column(String(64), default="", server_default="")
    tencent_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="", server_default="")
    market_label: Mapped[str] = mapped_column(String(32), default="", server_default="")
    raw_input: Mapped[str] = mapped_column(String(128), default="", server_default="")

    status: Mapped[str] = mapped_column(
        String(16), default=STATUS_PENDING, server_default=STATUS_PENDING, index=True
    )
    progress: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    message: Mapped[str] = mapped_column(String(512), default="", server_default="")
    metrics_json: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[str] = mapped_column(Text, default="", server_default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
