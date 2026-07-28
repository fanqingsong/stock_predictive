"""Shared helpers for model listing and serialization."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote

from sqlalchemy.orm import Session

from ..ml.storage import (
    _json_safe,
    any_artifact_exists,
    delete_artifact,
    load_all_metas,
    load_meta,
    models_root,
)
from ..models import StockModelArtifact
from ..stock_data import lookup_stock_name

MARKET_LABELS = {
    "cn_sh": "上交所",
    "cn_sz": "深交所",
    "cn_hk": "港交所",
    "us": "美股",
}


def market_label(market: str) -> str:
    return MARKET_LABELS.get(market, market or "-")


def _tf_val_acc(block: dict) -> Optional[float]:
    if not block:
        return None
    if block.get("status") == "failed":
        return None
    if "val_direction_acc" in block and block["val_direction_acc"] is not None:
        return block["val_direction_acc"]
    return ((block.get("metrics") or {}).get("val") or {}).get("direction_acc")


def _arch_summary(meta: dict) -> dict:
    lookback = meta.get("lookback")
    input_size = meta.get("input_size")
    layers = meta.get("num_layers")
    hidden = meta.get("hidden_size")
    return {
        "input_shape": f"(B, {lookback}, {input_size})",
        "lookback": lookback,
        "input_size": input_size,
        "num_layers": layers,
        "hidden_size": hidden,
        "dropout": meta.get("dropout"),
        "diagram": (
            f"x({lookback}×{input_size}) → LSTM×{layers}(h={hidden}) → Linear(1)"
            if lookback is not None and input_size is not None
            else None
        ),
        "feature_columns": meta.get("feature_columns") or [],
        "bar_rule": meta.get("bar_rule"),
        "n_bars": meta.get("n_bars"),
        "n_feature_rows": meta.get("n_feature_rows"),
        "n_train_samples": ((meta.get("metrics") or {}).get("train") or {}).get("n_samples"),
        "n_val_samples": ((meta.get("metrics") or {}).get("val") or {}).get("n_samples"),
        "epochs_ran": meta.get("epochs_ran"),
        "best_epoch": meta.get("best_epoch"),
        "trained_at": meta.get("trained_at"),
    }


def serialize_model_card(
    artifact: Optional[StockModelArtifact] = None,
    meta: Optional[dict] = None,
    ticker: str = "",
    market: str = "",
    all_metas: Optional[dict] = None,
) -> dict[str, Any]:
    metrics: dict = {}
    trained_at = None
    feature_version = ""
    ticker_value = ticker
    market_value = market
    name = ""

    if artifact:
        ticker_value = artifact.ticker or ticker
        market_value = artifact.market
        metrics = artifact.metrics_json or {}
        trained_at = artifact.trained_at.isoformat() if artifact.trained_at else None
        feature_version = artifact.feature_version or ""
        if artifact.display and artifact.display != artifact.ticker:
            name = artifact.display

    if meta:
        metrics = metrics or meta.get("metrics") or {}
        trained_at = trained_at or meta.get("trained_at")
        feature_version = feature_version or meta.get("feature_version") or ""
        ticker_value = ticker_value or meta.get("ticker") or ticker
        market_value = market_value or meta.get("market") or market
        meta_name = (meta.get("name") or "").strip()
        if not name and meta_name and meta_name != ticker_value:
            name = meta_name

    if not name:
        name = lookup_stock_name(ticker_value, market_value)

    metrics = _json_safe(metrics) or {}
    by_tf = (metrics or {}).get("by_timeframe") or {}
    if all_metas:
        for tf, m in all_metas.items():
            if tf not in by_tf or by_tf[tf].get("status") != "ready":
                by_tf[tf] = {
                    "status": "ready",
                    "metrics": m.get("metrics"),
                    "val_direction_acc": (m.get("metrics") or {})
                    .get("val", {})
                    .get("direction_acc"),
                }
            by_tf[tf] = {
                **(by_tf.get(tf) or {}),
                "model": _arch_summary(m),
            }
            trained_at = trained_at or m.get("trained_at")
            feature_version = feature_version or m.get("feature_version") or ""

    val = (metrics or {}).get("val") or {}
    if not val and by_tf.get("day"):
        val = (by_tf["day"].get("metrics") or {}).get("val") or {}
    train = (metrics or {}).get("train") or {}
    if not train and by_tf.get("day"):
        train = (by_tf["day"].get("metrics") or {}).get("train") or {}

    return {
        "ticker": ticker_value,
        "name": name,
        "market": market_value,
        "market_label": market_label(market_value),
        "trained_at": trained_at,
        "feature_version": feature_version,
        "task_type": "classification",
        "metrics": {**(metrics or {}), "by_timeframe": by_tf},
        "val_direction_acc": val.get("direction_acc") or _tf_val_acc(by_tf.get("day") or {}),
        "val_week_direction_acc": _tf_val_acc(by_tf.get("week") or {}),
        "val_month_direction_acc": _tf_val_acc(by_tf.get("month") or {}),
        "val_precision_up": val.get("precision_up"),
        "val_recall_up": val.get("recall_up"),
        "val_f1_up": val.get("f1_up"),
        "train_direction_acc": train.get("direction_acc"),
        "predict_url": f"/predict/{quote(str(ticker_value), safe='')}",
    }


def list_ready_models(db: Session) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    artifacts = (
        db.query(StockModelArtifact)
        .filter(StockModelArtifact.status == StockModelArtifact.STATUS_READY)
        .order_by(StockModelArtifact.trained_at.desc())
        .all()
    )
    for artifact in artifacts:
        if not any_artifact_exists(artifact.market, artifact.ticker):
            continue
        key = (artifact.market, artifact.ticker)
        if key in seen:
            continue
        seen.add(key)
        all_metas = load_all_metas(artifact.market, artifact.ticker)
        day_meta = all_metas.get("day") or load_meta(artifact.market, artifact.ticker, "day")
        cards.append(
            serialize_model_card(
                artifact=artifact,
                meta=day_meta,
                all_metas=all_metas,
            )
        )

    root = models_root()
    if root.is_dir():
        for market_dir in sorted(root.iterdir()):
            if not market_dir.is_dir():
                continue
            for tdir in sorted(market_dir.iterdir()):
                if not tdir.is_dir():
                    continue
                market = market_dir.name
                ticker = tdir.name
                key = (market, ticker)
                if key in seen:
                    continue
                if not any_artifact_exists(market, ticker):
                    continue
                all_metas = load_all_metas(market, ticker)
                day_meta = all_metas.get("day") or {}
                cards.append(
                    serialize_model_card(
                        meta=day_meta,
                        ticker=ticker,
                        market=market,
                        all_metas=all_metas,
                    )
                )
                seen.add(key)

    return cards


def delete_ready_model(db: Session, market: str, ticker: str) -> dict[str, Any]:
    """Delete DB artifact rows and on-disk files for one market/ticker."""
    removed_disk = delete_artifact(market, ticker)
    deleted_rows = (
        db.query(StockModelArtifact)
        .filter(
            StockModelArtifact.ticker == ticker,
            StockModelArtifact.market == market,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {
        "ticker": ticker,
        "market": market,
        "removed_disk": removed_disk,
        "deleted_rows": int(deleted_rows),
    }
