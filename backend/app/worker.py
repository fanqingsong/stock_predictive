"""Train worker: poll TrainJob queue and run PyTorch LSTM training."""

from __future__ import annotations

import argparse
import time
import traceback
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.ml import max_history_bars
from app.ml.storage import _json_safe, ticker_dir
from app.ml.trainer import train_lstm_on_history
from app.models import StockModelArtifact, TrainJob
from app.stock_data import ResolvedTicker, fetch_benchmark_history, fetch_history


def claim_next_job(db: Session) -> Optional[TrainJob]:
    dialect = db.bind.dialect.name if db.bind is not None else ""
    q = (
        db.query(TrainJob)
        .filter(TrainJob.status == TrainJob.STATUS_PENDING)
        .order_by(TrainJob.created_at.asc())
    )
    if dialect == "postgresql":
        job = q.with_for_update(skip_locked=True).first()
    else:
        job = q.with_for_update().first()
    if not job:
        return None
    job.status = TrainJob.STATUS_RUNNING
    job.started_at = datetime.utcnow()
    job.progress = max(job.progress or 0, 1)
    job.message = job.message or "Starting"
    db.commit()
    db.refresh(job)
    return job


def run_job(db: Session, job: TrainJob) -> None:
    resolved = ResolvedTicker(
        display=job.display or job.ticker,
        market=job.market,
        tencent_code=job.tencent_code or None,
        currency=job.currency or "USD",
        market_label=job.market_label or job.market,
    )

    def progress_cb(pct: int, message: str) -> None:
        db.execute(
            text(
                "UPDATE train_jobs SET progress = :p, message = :m WHERE id = :id"
            ),
            {"p": max(0, min(100, int(pct))), "m": (message or "")[:512], "id": job.id},
        )
        db.commit()

    bars = max_history_bars()
    progress_cb(3, "Downloading history")
    history = fetch_history(resolved, bars=bars)
    if history is None or history.empty:
        raise ValueError(f"No history for {resolved.display}")

    progress_cb(4, "Downloading benchmark")
    benchmark, benchmark_history = fetch_benchmark_history(resolved.market, bars=bars)
    benchmark_meta = {
        "symbol": benchmark.display,
        "name": benchmark.name or benchmark.display,
        "market": benchmark.market,
        "tencent_code": benchmark.tencent_code,
    }

    meta, paths = train_lstm_on_history(
        history,
        market=resolved.market,
        ticker=resolved.display,
        benchmark_df=benchmark_history,
        benchmark_meta=benchmark_meta,
        job_id=str(job.id),
        progress_cb=progress_cb,
    )

    metrics = _json_safe(meta.get("metrics") or {})
    # Persist timeframe summary alongside day metrics for UI cards.
    if meta.get("by_timeframe"):
        metrics = _json_safe(
            {
                **(meta.get("metrics") or {}),
                "by_timeframe": meta.get("by_timeframe"),
            }
        )
    existing = (
        db.query(StockModelArtifact)
        .filter(
            StockModelArtifact.ticker == resolved.display,
            StockModelArtifact.market == resolved.market,
        )
        .first()
    )
    if existing:
        existing.display = resolved.display
        existing.artifact_dir = paths.get("dir") or str(
            ticker_dir(resolved.market, resolved.display)
        )
        existing.status = StockModelArtifact.STATUS_READY
        existing.metrics_json = metrics
        existing.feature_version = meta.get("feature_version", "")
        existing.trained_at = datetime.utcnow()
    else:
        db.add(
            StockModelArtifact(
                ticker=resolved.display,
                market=resolved.market,
                display=resolved.display,
                artifact_dir=paths.get("dir")
                or str(ticker_dir(resolved.market, resolved.display)),
                status=StockModelArtifact.STATUS_READY,
                metrics_json=metrics,
                feature_version=meta.get("feature_version", ""),
                trained_at=datetime.utcnow(),
            )
        )

    job.status = TrainJob.STATUS_SUCCEEDED
    job.progress = 100
    job.metrics_json = metrics
    job.message = "Training completed"
    job.finished_at = datetime.utcnow()
    job.error = ""
    db.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock LSTM train worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()

    init_db()
    interval = max(0.5, float(args.poll_interval))
    print("Train worker started", flush=True)

    while True:
        job = None
        db = SessionLocal()
        try:
            job = claim_next_job(db)
            if job is None:
                db.close()
                if args.once:
                    print("No pending jobs.")
                    return
                time.sleep(interval)
                continue

            print(f"Running job {job.id}: {job.display or job.ticker} ({job.market})", flush=True)
            run_job(db, job)
            print(f"Job {job.id} succeeded", flush=True)
        except Exception as exc:
            print(f"Job failed: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            if job is not None:
                try:
                    db.rollback()
                    failed = db.query(TrainJob).filter(TrainJob.id == job.id).first()
                    if failed:
                        failed.status = TrainJob.STATUS_FAILED
                        failed.error = str(exc)[:4000]
                        failed.message = "Training failed"
                        failed.finished_at = datetime.utcnow()
                        db.commit()
                except Exception:
                    db.rollback()
        finally:
            db.close()

        if args.once:
            return


if __name__ == "__main__":
    main()
