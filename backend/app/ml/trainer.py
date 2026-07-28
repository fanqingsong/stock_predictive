"""Train scale-native PyTorch LSTM classifiers (day / week / month)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .features import (
    build_features,
    feature_column_names,
    make_sequences,
    prepare_bars_with_benchmark,
)
from .lstm_model import StockLSTM
from .metrics import compute_metrics
from .storage import save_artifact
from .timeframes import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    FEATURE_VERSION,
    TASK_TYPE,
    TIMEFRAME_KEYS,
    TRAIN_RATIO,
    timeframe_config,
)


ProgressCallback = Optional[Callable[[int, str], None]]


def _device() -> torch.device:
    return torch.device("cpu")


def _predict_proba(model: StockLSTM, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    if not probs:
        return np.array([], dtype=np.float32)
    return np.concatenate(probs, axis=0)


def train_one_timeframe(
    history_df,
    *,
    market: str,
    ticker: str,
    timeframe: str,
    benchmark_df,
    benchmark_meta: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    patience: int = DEFAULT_PATIENCE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    progress_cb: ProgressCallback = None,
    progress_span: Tuple[int, int] = (0, 100),
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Train a single-scale classifier on day or resampled week/month bars."""

    cfg = timeframe_config(timeframe)
    lookback = int(cfg["lookback"])
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_layers"])
    dropout = float(cfg["dropout"])
    feature_windows = cfg["feature_windows"]
    feature_columns = feature_column_names(feature_windows)
    lo, hi = progress_span

    def report(local_pct: int, message: str) -> None:
        if progress_cb:
            pct = lo + int((hi - lo) * max(0, min(100, local_pct)) / 100)
            progress_cb(pct, f"[{cfg['label']}] {message}")

    report(5, "Building bars/features")
    bars = prepare_bars_with_benchmark(
        history_df, benchmark_df, timeframe, drop_incomplete=True
    )
    feat_df = build_features(bars, timeframe=timeframe, for_training=True)
    min_rows = lookback + int(cfg["min_extra_rows"])
    if len(feat_df) < min_rows:
        raise ValueError(
            f"{timeframe}: not enough bars after features: {len(feat_df)} < {min_rows}"
        )

    feature_matrix = feat_df[feature_columns].values.astype(np.float64)
    targets = feat_df["target"].values.astype(np.float64)

    split_idx = int(len(feat_df) * TRAIN_RATIO)
    split_idx = max(lookback + 3, min(split_idx, len(feat_df) - lookback - 3))

    train_feat = feature_matrix[:split_idx]
    val_feat = feature_matrix[split_idx:]
    train_tgt = targets[:split_idx]
    val_tgt = targets[split_idx:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    val_scaled = scaler.transform(val_feat)

    x_train, y_train = make_sequences(train_scaled, train_tgt, lookback)

    context = train_scaled[-lookback:] if len(train_scaled) >= lookback else train_scaled
    val_with_ctx = np.vstack([context, val_scaled])
    val_tgt_with_ctx = np.concatenate(
        [train_tgt[-lookback:] if len(train_tgt) >= lookback else train_tgt, val_tgt]
    )
    x_val, y_val = make_sequences(val_with_ctx, val_tgt_with_ctx, lookback)
    if len(x_val) > len(val_scaled):
        trim = len(x_val) - len(val_scaled)
        x_val, y_val = x_val[trim:], y_val[trim:]

    if len(x_train) < 5 or len(x_val) < 2:
        raise ValueError(
            f"{timeframe}: insufficient sequences train={len(x_train)} val={len(x_val)}"
        )

    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    device = _device()
    model = StockLSTM(
        input_size=len(feature_columns),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_outputs=1,
    ).to(device)

    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.astype(np.float32))
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    train_eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), y_train_t),
        batch_size=batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), y_val_t),
        batch_size=batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    report(15, "Training")
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    wait = 0
    epochs_ran = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        epochs_ran = epoch + 1
        val_prob = _predict_proba(model, val_loader, device)
        val_metrics_epoch = compute_metrics(y_val, val_prob)
        val_acc = float(val_metrics_epoch.get("direction_acc") or 0.0)
        avg = epoch_loss / max(n_batches, 1)
        local = 15 + int(70 * epochs_ran / epochs)
        report(local, f"Epoch {epochs_ran}/{epochs} loss={avg:.4f} val_acc={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epochs_ran
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                report(local, f"Early stop (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    report(90, "Evaluating")
    train_prob = _predict_proba(model, train_eval_loader, device)
    val_prob = _predict_proba(model, val_loader, device)
    metrics = {
        "train": compute_metrics(y_train, train_prob),
        "val": compute_metrics(y_val, val_prob),
    }

    meta: Dict[str, Any] = {
        "ticker": ticker,
        "market": market,
        "timeframe": timeframe,
        "label": cfg["label"],
        "description": cfg["description"],
        "bar_rule": cfg["bar_rule"],
        "task_type": TASK_TYPE,
        "feature_version": FEATURE_VERSION,
        "feature_columns": feature_columns,
        "feature_windows": feature_windows,
        "benchmark": benchmark_meta or {},
        "lookback": lookback,
        "input_size": len(feature_columns),
        "num_outputs": 1,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs": epochs,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "pos_weight": float(pos_weight.item()),
        "train_ratio": TRAIN_RATIO,
        "n_feature_rows": int(len(feat_df)),
        "n_bars": int(len(bars)),
        "split_idx": int(split_idx),
        "metrics": metrics,
        "job_id": job_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    report(95, "Saving")
    paths = save_artifact(market, ticker, model, scaler, meta, timeframe=timeframe)
    report(100, "Done")
    return meta, paths


def train_lstm_on_history(
    history_df,
    *,
    market: str,
    ticker: str,
    benchmark_df,
    benchmark_meta: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    timeframes: Optional[List[str]] = None,
    progress_cb: ProgressCallback = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Train day/week/month models sequentially; return combined meta + last paths."""
    keys = list(timeframes) if timeframes else list(TIMEFRAME_KEYS)
    by_tf: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    last_paths: Dict[str, str] = {}
    n = len(keys)

    for i, tf in enumerate(keys):
        lo = int(100 * i / n)
        hi = int(100 * (i + 1) / n)
        try:
            meta, paths = train_one_timeframe(
                history_df,
                market=market,
                ticker=ticker,
                timeframe=tf,
                benchmark_df=benchmark_df,
                benchmark_meta=benchmark_meta,
                job_id=job_id,
                progress_cb=progress_cb,
                progress_span=(lo, hi),
                **{k: v for k, v in kwargs.items() if k in {
                    "epochs", "batch_size", "lr", "patience", "weight_decay"
                }},
            )
            by_tf[tf] = {
                "status": "ready",
                "metrics": meta.get("metrics"),
                "lookback": meta.get("lookback"),
                "feature_version": meta.get("feature_version"),
                "trained_at": meta.get("trained_at"),
                "n_feature_rows": meta.get("n_feature_rows"),
                "val_direction_acc": (meta.get("metrics") or {}).get("val", {}).get(
                    "direction_acc"
                ),
            }
            last_paths = paths
        except Exception as exc:
            errors[tf] = str(exc)
            by_tf[tf] = {"status": "failed", "error": str(exc)}
            if progress_cb:
                progress_cb(hi, f"[{tf}] skipped: {exc}")

    if not any(v.get("status") == "ready" for v in by_tf.values()):
        raise ValueError(f"All timeframes failed: {errors}")

    # Top-level metrics stay day-centric for UI cards; full detail in by_timeframe.
    day_block = by_tf.get("day") or {}
    combined = {
        "ticker": ticker,
        "market": market,
        "feature_version": FEATURE_VERSION,
        "task_type": TASK_TYPE,
        "benchmark": benchmark_meta or {},
        "by_timeframe": by_tf,
        "errors": errors,
        "metrics": (day_block.get("metrics") if day_block.get("status") == "ready" else {}),
        "job_id": job_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    # Prefer ticker root as artifact dir for DB.
    from .storage import ticker_dir

    last_paths = dict(last_paths)
    last_paths["dir"] = str(ticker_dir(market, ticker))
    return combined, last_paths
