"""Load scale-native models and predict next day/week/month direction."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .features import build_features, feature_column_names, prepare_bars_with_benchmark
from .resample import bars_for_timeframe
from .storage import artifact_exists, load_artifact
from .timeframes import TIMEFRAME_KEYS, timeframe_config


def _device() -> torch.device:
    return torch.device("cpu")


def _feature_dim(cfg: Dict[str, Any]) -> int:
    return len(feature_column_names(cfg["feature_windows"]))


def _series_from_bars(bars: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    if bars is None or bars.empty:
        return []
    out = []
    for idx, row in bars.tail(limit).iterrows():
        out.append(
            {
                "date": idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row.get("Adj Close", row["Close"])),
                "volume": float(row.get("Volume", 0) or 0),
            }
        )
    return out


def _model_detail(meta: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    metrics = meta.get("metrics") or {}
    train_m = metrics.get("train") or {}
    val_m = metrics.get("val") or {}
    lookback = int(meta.get("lookback", cfg["lookback"]))
    input_size = int(meta.get("input_size", len(meta.get("feature_columns") or [])))
    hidden = int(meta.get("hidden_size", cfg["hidden_size"]))
    layers = int(meta.get("num_layers", cfg["num_layers"]))
    return {
        "architecture": {
            "network": "StockLSTM",
            "input_shape": f"(B, {lookback}, {input_size})",
            "lookback": lookback,
            "input_size": input_size,
            "hidden_size": hidden,
            "num_layers": layers,
            "dropout": meta.get("dropout", cfg.get("dropout")),
            "num_outputs": meta.get("num_outputs", 1),
            "head": "Linear → logit → sigmoid P(up)",
            "feature_columns": meta.get("feature_columns")
            or feature_column_names(cfg["feature_windows"]),
            "feature_windows": meta.get("feature_windows") or cfg.get("feature_windows"),
            "diagram": (
                f"x({lookback}×{input_size}) → LSTM×{layers}(h={hidden}) "
                f"→ last step → Dropout → Linear(1) → logit"
            ),
        },
        "training": {
            "bar_rule": meta.get("bar_rule") or cfg.get("bar_rule"),
            "n_bars": meta.get("n_bars"),
            "n_feature_rows": meta.get("n_feature_rows"),
            "train_ratio": meta.get("train_ratio"),
            "split_idx": meta.get("split_idx"),
            "n_train_samples": train_m.get("n_samples"),
            "n_val_samples": val_m.get("n_samples"),
            "epochs": meta.get("epochs"),
            "epochs_ran": meta.get("epochs_ran"),
            "best_epoch": meta.get("best_epoch"),
            "patience": meta.get("patience"),
            "batch_size": meta.get("batch_size"),
            "lr": meta.get("lr"),
            "weight_decay": meta.get("weight_decay"),
            "pos_weight": meta.get("pos_weight"),
            "trained_at": meta.get("trained_at"),
            "feature_version": meta.get("feature_version"),
            "train_metrics": train_m,
            "val_metrics": val_m,
        },
    }


def predict_one_timeframe(
    history_df: pd.DataFrame,
    market: str,
    ticker: str,
    timeframe: str,
    benchmark_df: pd.DataFrame,
) -> Dict[str, Any]:
    cfg = timeframe_config(timeframe)
    n_features = _feature_dim(cfg)
    series_limit = {"day": 120, "week": 80, "month": 48}.get(timeframe, 60)
    bars = bars_for_timeframe(history_df, timeframe, drop_incomplete=True)
    series = _series_from_bars(bars, series_limit)

    if not artifact_exists(market, ticker, timeframe):
        return {
            "key": timeframe,
            "label": cfg["label"],
            "description": cfg["description"],
            "bar_rule": cfg["bar_rule"],
            "status": "missing",
            "error": "model not trained",
            "series": series,
            "architecture": {
                "network": "StockLSTM",
                "input_shape": f"(B, {cfg['lookback']}, {n_features})",
                "lookback": cfg["lookback"],
                "hidden_size": cfg["hidden_size"],
                "num_layers": cfg["num_layers"],
                "dropout": cfg["dropout"],
                "diagram": (
                    f"x({cfg['lookback']}×{n_features}) → LSTM×{cfg['num_layers']}"
                    f"(h={cfg['hidden_size']}) → Linear(1)"
                ),
                "feature_columns": feature_column_names(cfg["feature_windows"]),
            },
            "training": None,
        }

    model, scaler, meta = load_artifact(market, ticker, timeframe, device=_device())
    lookback = int(meta.get("lookback", cfg["lookback"]))
    feature_columns = meta.get("feature_columns") or feature_column_names(
        cfg["feature_windows"]
    )
    detail = _model_detail(meta, cfg)

    feat_bars = prepare_bars_with_benchmark(
        history_df, benchmark_df, timeframe, drop_incomplete=True
    )
    feat_df = build_features(feat_bars, timeframe=timeframe, for_training=False)
    if len(feat_df) < lookback:
        return {
            "key": timeframe,
            "label": cfg["label"],
            "description": cfg["description"],
            "bar_rule": cfg["bar_rule"],
            "status": "error",
            "error": f"need lookback={lookback}, got {len(feat_df)} feature rows",
            "series": series,
            **detail,
        }

    window = feat_df[feature_columns].values.astype(np.float64)[-lookback:]
    window_scaled = scaler.transform(window)
    x = torch.from_numpy(window_scaled.astype(np.float32)).unsqueeze(0).to(_device())
    model.eval()
    with torch.no_grad():
        logit = float(model(x).cpu().reshape(-1)[0].item())
    prob_up = float(1.0 / (1.0 + np.exp(-logit)))
    direction = "up" if prob_up >= 0.5 else "down"

    as_of = feat_df.index[-1]
    as_of_str = as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime") else str(as_of)
    ref_close = float(feat_df["close"].iloc[-1]) if "close" in feat_df.columns else None

    return {
        "key": timeframe,
        "label": cfg["label"],
        "description": meta.get("description") or cfg["description"],
        "bar_rule": meta.get("bar_rule") or cfg["bar_rule"],
        "status": "ok",
        "probability": round(prob_up, 4),
        "direction": direction,
        "as_of": as_of_str,
        "ref_close": ref_close,
        "lookback": lookback,
        "feature_version": meta.get("feature_version"),
        "series": series,
        **detail,
    }


def predict_horizons(
    history_df: pd.DataFrame,
    market: str,
    ticker: str,
    benchmark_df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], dict]:
    """Run day/week/month predictors independently; combine metas."""
    results: List[Dict[str, Any]] = []
    metas: Dict[str, Any] = {}
    for tf in TIMEFRAME_KEYS:
        item = predict_one_timeframe(
            history_df, market, ticker, tf, benchmark_df=benchmark_df
        )
        results.append(item)
        if artifact_exists(market, ticker, tf):
            try:
                _, _, meta = load_artifact(market, ticker, tf, device=_device())
                metas[tf] = meta
            except Exception:
                pass

    combined_metrics = {
        "by_timeframe": {
            tf: (metas[tf].get("metrics") if tf in metas else {})
            for tf in TIMEFRAME_KEYS
        },
        "train": (metas.get("day") or {}).get("metrics", {}).get("train") or {},
        "val": (metas.get("day") or {}).get("metrics", {}).get("val") or {},
    }
    summary_meta = {
        "feature_version": (metas.get("day") or {}).get("feature_version")
        or next((m.get("feature_version") for m in metas.values()), "-"),
        "task_type": "classification",
        "trained_at": max(
            (m.get("trained_at") or "" for m in metas.values()),
            default="-",
        ),
        "metrics": combined_metrics,
    }
    return results, summary_meta
