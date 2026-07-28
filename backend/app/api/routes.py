"""API route modules."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import unquote

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..ml import max_history_bars
from ..ml.predictor import predict_horizons
from ..ml.storage import any_artifact_exists, load_meta
from ..models import StockModelArtifact, TrainJob
from ..schemas import (
    HealthResponse,
    ModelStatusResponse,
    ModelsListResponse,
    SuggestResponse,
    TrainCreateResponse,
    TrainRequest,
    TrainStatusResponse,
)
from ..services.models_service import delete_ready_model, list_ready_models, market_label
from ..stock_data import (
    fetch_benchmark_history,
    fetch_history,
    fetch_multi_quotes,
    fetch_quote_info,
    fetch_tencent_daily,
    is_chinese_market,
    resolve_ticker,
    suggest_stocks,
)

router = APIRouter()

CN_HOME_CODES = (
    "sh600519",
    "sz000001",
    "sh601318",
    "sz300750",
    "sh600036",
    "hk00700",
)


@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@router.get("/api/home")
def home_dashboard() -> dict[str, Any]:
    recent_stocks: list[dict] = []
    series: list[dict] = []
    try:
        quotes = fetch_multi_quotes(CN_HOME_CODES)
        if not quotes.empty:
            recent_stocks = quotes.reset_index().to_dict(orient="records")

        labels = {
            "sh600519": "贵州茅台",
            "sz000001": "平安银行",
            "sh601318": "中国平安",
            "sz300750": "宁德时代",
        }
        for code in CN_HOME_CODES[:4]:
            hist = fetch_tencent_daily(code, bars=30)
            if hist.empty:
                continue
            points = [
                {"date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx), "close": float(row["Close"])}
                for idx, row in hist.iterrows()
                if pd.notna(row.get("Close"))
            ]
            series.append({"code": code, "name": labels.get(code, code), "points": points})
    except Exception as exc:
        print(f"home load failed: {exc}")

    return {"ok": True, "recent_stocks": recent_stocks, "series": series}


@router.get("/api/suggest", response_model=SuggestResponse)
def suggest(q: str = Query("", alias="q")):
    query = (q or "").strip()
    items = suggest_stocks(query, limit=8) if query else []
    return {"query": query, "items": items}


@router.get("/api/models", response_model=ModelsListResponse)
def api_list_models(db: Session = Depends(get_db)):
    try:
        models = list_ready_models(db)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "count": len(models), "items": models}


@router.get("/api/model/{ticker_value:path}", response_model=ModelStatusResponse)
def api_model_status(ticker_value: str, db: Session = Depends(get_db)):
    raw = unquote(ticker_value or "").strip()
    resolved = resolve_ticker(raw)
    if resolved is None:
        raise HTTPException(status_code=400, detail="invalid ticker")

    artifact = (
        db.query(StockModelArtifact)
        .filter(
            StockModelArtifact.ticker == resolved.display,
            StockModelArtifact.market == resolved.market,
        )
        .order_by(StockModelArtifact.trained_at.desc())
        .first()
    )
    on_disk = any_artifact_exists(resolved.market, resolved.display)
    meta = load_meta(resolved.market, resolved.display, "day") if on_disk else None
    metrics = {}
    if artifact and artifact.metrics_json:
        metrics = artifact.metrics_json
    elif meta:
        metrics = meta.get("metrics") or {}

    return {
        "ok": True,
        "ticker": resolved.display,
        "market": resolved.market,
        "market_label": resolved.market_label,
        "ready": bool(
            on_disk and (not artifact or artifact.status == StockModelArtifact.STATUS_READY)
        ),
        "on_disk": on_disk,
        "trained_at": (
            artifact.trained_at.isoformat()
            if artifact and artifact.trained_at
            else (meta or {}).get("trained_at")
        ),
        "metrics": metrics,
        "feature_version": (
            artifact.feature_version if artifact else (meta or {}).get("feature_version")
        ),
    }


@router.delete("/api/model/{ticker_value:path}")
def api_delete_model(
    ticker_value: str,
    market: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    raw = unquote(ticker_value or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="ticker is required")

    resolved = resolve_ticker(raw)
    if resolved is None:
        raise HTTPException(status_code=400, detail="invalid ticker")

    target_market = (market or "").strip() or resolved.market
    target_ticker = resolved.display

    active = (
        db.query(TrainJob)
        .filter(
            TrainJob.ticker == target_ticker,
            TrainJob.market == target_market,
            TrainJob.status.in_([TrainJob.STATUS_PENDING, TrainJob.STATUS_RUNNING]),
        )
        .first()
    )
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"训练任务进行中（#{active.id}），无法删除",
        )

    on_disk = any_artifact_exists(target_market, target_ticker)
    artifact = (
        db.query(StockModelArtifact)
        .filter(
            StockModelArtifact.ticker == target_ticker,
            StockModelArtifact.market == target_market,
        )
        .first()
    )
    if not on_disk and not artifact:
        raise HTTPException(status_code=404, detail="model not found")

    result = delete_ready_model(db, target_market, target_ticker)
    return {"ok": True, **result}


@router.post("/api/train", response_model=TrainCreateResponse)
def api_train(payload: TrainRequest, db: Session = Depends(get_db)):
    raw = (payload.ticker or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="ticker is required")

    resolved = resolve_ticker(raw)
    if resolved is None:
        raise HTTPException(status_code=400, detail="invalid ticker")

    active = (
        db.query(TrainJob)
        .filter(
            TrainJob.ticker == resolved.display,
            TrainJob.market == resolved.market,
            TrainJob.status.in_([TrainJob.STATUS_PENDING, TrainJob.STATUS_RUNNING]),
        )
        .first()
    )
    if active:
        return {
            "ok": True,
            "job_id": active.id,
            "status": active.status,
            "reused": True,
        }

    job = TrainJob(
        ticker=resolved.display,
        market=resolved.market,
        display=resolved.display,
        tencent_code=resolved.tencent_code,
        currency=resolved.currency,
        market_label=resolved.market_label,
        raw_input=raw,
        status=TrainJob.STATUS_PENDING,
        progress=0,
        message="Queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {
        "ok": True,
        "job_id": job.id,
        "status": job.status,
        "reused": False,
    }


@router.get("/api/train/{job_id}", response_model=TrainStatusResponse)
def api_train_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(TrainJob).filter(TrainJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "ok": True,
        "job_id": job.id,
        "ticker": job.display or job.ticker,
        "market": job.market,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "metrics": job.metrics_json or {},
        "error": job.error or "",
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@router.get("/api/tickers")
def api_tickers() -> dict[str, Any]:
    cn_path = settings.DATA_DIR / "cn_tickers.csv"
    us_path = settings.DATA_DIR / "new_tickers.csv"
    cn_df = pd.read_csv(cn_path, dtype={"Symbol": str})
    us_df = pd.read_csv(us_path, dtype={"Symbol": str})

    if "Market" not in us_df.columns:
        us_df = us_df[["Symbol", "Name"]].copy()
        us_df["Market"] = "US"
    cn_view = cn_df[["Symbol", "Name", "Market"]].copy()
    us_view = us_df[["Symbol", "Name", "Market"]].copy()
    combined = pd.concat([cn_view, us_view], ignore_index=True)
    items = combined.to_dict(orient="records")
    return {"ok": True, "count": len(items), "items": items}


@router.get("/api/predict/{ticker_value:path}")
def api_predict(
    ticker_value: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    raw = unquote(ticker_value or "").strip()
    resolved = resolve_ticker(raw)
    if resolved is None:
        raise HTTPException(status_code=400, detail="invalid ticker")

    if not any_artifact_exists(resolved.market, resolved.display):
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {resolved.display} ({resolved.market_label})",
        )

    try:
        df = fetch_history(resolved, bars=max_history_bars())
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="invalid ticker or empty history")
        _, benchmark_df = fetch_benchmark_history(resolved.market, bars=max_history_bars())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"market data unavailable: {exc}") from exc

    try:
        horizons, meta = predict_horizons(
            df, resolved.market, resolved.display, benchmark_df=benchmark_df
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"forecast failed: {exc}") from exc

    metrics = (meta or {}).get("metrics") or {}
    train_metrics = metrics.get("train") or {}
    val_metrics = metrics.get("val") or {}
    by_timeframe = metrics.get("by_timeframe") or {}

    history = []
    for idx, row in df.tail(120).iterrows():
        history.append(
            {
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0) or 0),
            }
        )

    try:
        info = fetch_quote_info(resolved)
    except Exception:
        info = {
            "Symbol": resolved.display,
            "Name": resolved.display,
            "Last_Sale": "-",
            "Net_Change": "-",
            "Percent_Change": "-",
            "Market_Cap": "-",
            "Country": "China" if is_chinese_market(resolved.market) else "United States",
            "IPO_Year": "-",
            "Volume": "-",
            "Sector": "-",
            "Industry": "-",
        }

    currency_label = {
        "CNY": "CNY / 元",
        "HKD": "HKD / 港币",
        "USD": "USD / 美元",
    }.get(resolved.currency, resolved.currency)

    return {
        "ok": True,
        "ticker": resolved.display,
        "market": resolved.market,
        "market_label": resolved.market_label or market_label(resolved.market),
        "currency_label": currency_label,
        "history": history,
        "horizons": horizons,
        "trained_at": (meta or {}).get("trained_at", "-"),
        "feature_version": (meta or {}).get("feature_version", "-"),
        "task_type": (meta or {}).get("task_type", "classification"),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "by_timeframe": by_timeframe,
        },
        "quote": info,
    }
