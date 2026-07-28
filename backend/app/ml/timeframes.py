"""Per-timeframe hyperparameters for scale-native models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

FEATURE_VERSION = "v5-scale"
TASK_TYPE = "classification"
TRAIN_RATIO = 0.8
DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 10
DEFAULT_WEIGHT_DECAY = 1e-4

# Shared relative feature recipe (windows in *bars* of that timeframe).
DAY_WEEK_FEATURE_WINDOWS: Dict[str, Any] = {
    "return_short": 1,
    "return_long": 5,
    "sma_fast": 5,
    "sma_slow": 20,
    "rsi": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "vol": 20,
    "volume_sma": 20,
}

# Shorter windows on monthly bars (avoid 14–26 month lags).
MONTH_FEATURE_WINDOWS: Dict[str, Any] = {
    "return_short": 1,
    "return_long": 3,
    "sma_fast": 3,
    "sma_slow": 6,
    "rsi": 6,
    "macd_fast": 6,
    "macd_slow": 12,
    "macd_signal": 4,
    "vol": 6,
    "volume_sma": 6,
}

TIMEFRAMES: Dict[str, Dict[str, Any]] = {
    "day": {
        "key": "day",
        "label": "日",
        "description": "下一交易日相对今日涨跌",
        "bar_rule": "trading_day",
        "lookback": 10,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "history_bars": 320,
        "feature_windows": DAY_WEEK_FEATURE_WINDOWS,
        "min_extra_rows": 40,
    },
    "week": {
        "key": "week",
        "label": "周",
        "description": "下一自然周相对本周收盘涨跌",
        "bar_rule": "natural_week",
        "lookback": 12,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "history_bars": 1300,  # ~5y trading days
        "feature_windows": DAY_WEEK_FEATURE_WINDOWS,
        "min_extra_rows": 30,
    },
    "month": {
        "key": "month",
        "label": "月",
        "description": "下一自然月相对本月收盘涨跌",
        "bar_rule": "natural_month",
        # Shorter lookback: CN day-K APIs often return ~2–3y only (~30 months).
        "lookback": 6,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.4,
        "history_bars": 2000,
        "feature_windows": MONTH_FEATURE_WINDOWS,
        "min_extra_rows": 10,
    },
}

TIMEFRAME_KEYS: Tuple[str, ...] = ("day", "week", "month")


def timeframe_config(key: str) -> Dict[str, Any]:
    if key not in TIMEFRAMES:
        raise KeyError(f"Unknown timeframe: {key}")
    return TIMEFRAMES[key]


def max_history_bars() -> int:
    # Cap for market APIs that reject oversized day-K requests (e.g. Tencent ~2000).
    return min(2000, max(int(cfg["history_bars"]) for cfg in TIMEFRAMES.values()))
