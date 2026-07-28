"""Model artifact paths and persistence helpers (per-timeframe)."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch

from .lstm_model import StockLSTM
from .timeframes import TIMEFRAME_KEYS


def models_root() -> Path:
    root = os.environ.get("MODELS_STORE_DIR")
    if root:
        return Path(root)
    return Path(__file__).resolve().parents[2] / "models_store"


def _safe_segment(value: str) -> str:
    text = (value or "unknown").strip()
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text[:80] or "unknown"


def ticker_dir(market: str, ticker: str) -> Path:
    return models_root() / _safe_segment(market) / _safe_segment(ticker)


def artifact_dir(market: str, ticker: str, timeframe: str = "day") -> Path:
    return ticker_dir(market, ticker) / _safe_segment(timeframe)


def model_paths(market: str, ticker: str, timeframe: str = "day") -> Dict[str, Path]:
    base = artifact_dir(market, ticker, timeframe)
    return {
        "dir": base,
        "model": base / "model.pt",
        "scaler": base / "scaler.joblib",
        "meta": base / "meta.json",
    }


def artifact_exists(market: str, ticker: str, timeframe: str = "day") -> bool:
    paths = model_paths(market, ticker, timeframe)
    return paths["model"].is_file() and paths["scaler"].is_file() and paths["meta"].is_file()


def any_artifact_exists(market: str, ticker: str) -> bool:
    """True if any timeframe model exists (also checks legacy flat layout)."""
    if any(artifact_exists(market, ticker, tf) for tf in TIMEFRAME_KEYS):
        return True
    # Legacy v4 flat layout: models_store/m/t/model.pt
    base = ticker_dir(market, ticker)
    return (base / "model.pt").is_file() and (base / "meta.json").is_file()


def delete_artifact(market: str, ticker: str) -> bool:
    """Remove on-disk ticker directory (all timeframes). Returns True if removed."""
    base = ticker_dir(market, ticker)
    if not base.exists():
        return False
    shutil.rmtree(base)
    parent = base.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
    return True


def load_meta(
    market: str, ticker: str, timeframe: str = "day"
) -> Optional[Dict[str, Any]]:
    paths = model_paths(market, ticker, timeframe)
    if not paths["meta"].is_file():
        # Legacy flat meta
        legacy = ticker_dir(market, ticker) / "meta.json"
        if timeframe == "day" and legacy.is_file():
            with open(legacy, "r", encoding="utf-8") as fh:
                return _json_safe(json.load(fh))
        return None
    with open(paths["meta"], "r", encoding="utf-8") as fh:
        return _json_safe(json.load(fh))


def load_all_metas(market: str, ticker: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for tf in TIMEFRAME_KEYS:
        meta = load_meta(market, ticker, tf)
        if meta:
            out[tf] = meta
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            return value
    return value


def save_artifact(
    market: str,
    ticker: str,
    model: StockLSTM,
    scaler,
    meta: Dict[str, Any],
    timeframe: str = "day",
) -> Dict[str, str]:
    paths = model_paths(market, ticker, timeframe)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "input_size": meta["input_size"],
        "hidden_size": meta["hidden_size"],
        "num_layers": meta["num_layers"],
        "dropout": meta["dropout"],
        "lookback": meta["lookback"],
        "feature_columns": meta["feature_columns"],
        "num_outputs": meta.get("num_outputs", 1),
        "timeframe": timeframe,
    }
    torch.save(payload, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    safe_meta = _json_safe(meta)
    with open(paths["meta"], "w", encoding="utf-8") as fh:
        json.dump(safe_meta, fh, ensure_ascii=False, indent=2, allow_nan=False)

    return {k: str(v) for k, v in paths.items()}


def load_artifact(
    market: str,
    ticker: str,
    timeframe: str = "day",
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cpu")
    paths = model_paths(market, ticker, timeframe)
    if not artifact_exists(market, ticker, timeframe):
        raise FileNotFoundError(f"No saved model for {market}/{ticker}/{timeframe}")

    payload = torch.load(paths["model"], map_location=device)
    model = StockLSTM(
        input_size=payload["input_size"],
        hidden_size=payload["hidden_size"],
        num_layers=payload["num_layers"],
        dropout=payload.get("dropout", 0.2),
        num_outputs=int(payload.get("num_outputs", 1)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()

    scaler = joblib.load(paths["scaler"])
    with open(paths["meta"], "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return model, scaler, meta
