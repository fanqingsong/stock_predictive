"""Scale-native relative feature engineering for LSTM classifiers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .resample import bars_for_timeframe
from .timeframes import timeframe_config

INDEX_CLOSE_COL = "Index Close"


def feature_column_names(windows: Dict[str, Any]) -> List[str]:
    return [
        "return_1",
        "return_long",
        "index_return_1",
        "index_return_long",
        "excess_return_1",
        "excess_return_long",
        "price_sma_fast_dev",
        "price_sma_slow_dev",
        "sma_fast_slope",
        "sma_slow_slope",
        "RSI",
        "macd_hist_norm",
        "bb_pct_b",
        "vol",
        "volume_sma_ratio",
    ]


def _normalize_ts_index(index: pd.Index) -> pd.DatetimeIndex:
    ts = pd.to_datetime(index)
    if getattr(ts, "tz", None) is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def attach_benchmark_close(
    stock_bars: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """Align benchmark close onto stock bars of the same timeframe.

    Day bars match by calendar date (ffill gaps). Week/month bars match by
    natural period key so slightly different session end dates still align.
    """
    if stock_bars is None or stock_bars.empty:
        return stock_bars
    if benchmark_daily is None or benchmark_daily.empty:
        raise ValueError("benchmark history is empty")

    bench_bars = bars_for_timeframe(benchmark_daily, timeframe, drop_incomplete=True)
    if bench_bars is None or bench_bars.empty:
        raise ValueError("benchmark bars empty after resample")

    out = stock_bars.copy()
    stock_index = _normalize_ts_index(out.index)
    out.index = stock_index

    bench_close = bench_bars["Adj Close"].astype(float).copy()
    bench_close.index = _normalize_ts_index(bench_close.index)

    if timeframe == "day":
        aligned = bench_close.reindex(stock_index).ffill()
    elif timeframe == "week":
        by_period: Dict[Any, float] = {}
        for ts, val in bench_close.items():
            by_period[pd.Timestamp(ts).to_period("W-SUN")] = float(val)
        aligned = pd.Series(
            [by_period.get(ts.to_period("W-SUN"), np.nan) for ts in stock_index],
            index=stock_index,
            dtype=float,
        )
    elif timeframe == "month":
        by_period = {}
        for ts, val in bench_close.items():
            by_period[pd.Timestamp(ts).to_period("M")] = float(val)
        aligned = pd.Series(
            [by_period.get(ts.to_period("M"), np.nan) for ts in stock_index],
            index=stock_index,
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    out[INDEX_CLOSE_COL] = aligned.values
    if out[INDEX_CLOSE_COL].isna().all():
        raise ValueError("failed to align benchmark close onto stock bars")
    return out


def prepare_bars_with_benchmark(
    history_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    timeframe: str,
    *,
    as_of: Optional[pd.Timestamp] = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Resample stock bars then attach aligned benchmark close."""
    bars = bars_for_timeframe(
        history_df,
        timeframe,
        as_of=as_of,
        drop_incomplete=drop_incomplete,
    )
    return attach_benchmark_close(bars, benchmark_df, timeframe)


def build_features(
    df: pd.DataFrame,
    *,
    timeframe: str = "day",
    for_training: bool = True,
) -> pd.DataFrame:
    """Build relative features + binary next-bar ``target`` on the given bar series.

    Requires ``Index Close`` (see ``attach_benchmark_close``).
    When ``for_training`` is False, keep the last row even if ``target`` is NaN.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cfg = timeframe_config(timeframe)
    windows = cfg["feature_windows"]
    cols = feature_column_names(windows)

    work = df.copy()
    if "Adj Close" not in work.columns:
        work["Adj Close"] = work["Close"]
    if "Volume" not in work.columns:
        work["Volume"] = 0.0
    if INDEX_CLOSE_COL not in work.columns:
        raise ValueError(
            f"missing {INDEX_CLOSE_COL}; call attach_benchmark_close before build_features"
        )

    close = work["Adj Close"].astype(float)
    volume = work["Volume"].astype(float)
    index_close = work[INDEX_CLOSE_COL].astype(float)

    rs = int(windows["return_short"])
    rl = int(windows["return_long"])
    sma_f = int(windows["sma_fast"])
    sma_s = int(windows["sma_slow"])
    rsi_n = int(windows["rsi"])
    macd_f = int(windows["macd_fast"])
    macd_s = int(windows["macd_slow"])
    macd_sig = int(windows["macd_signal"])
    vol_n = int(windows["vol"])
    vol_sma_n = int(windows["volume_sma"])

    feat = pd.DataFrame(index=work.index)
    feat["return_1"] = close.pct_change(rs)
    feat["return_long"] = close.pct_change(rl)
    feat["index_return_1"] = index_close.pct_change(rs)
    feat["index_return_long"] = index_close.pct_change(rl)
    feat["excess_return_1"] = feat["return_1"] - feat["index_return_1"]
    feat["excess_return_long"] = feat["return_long"] - feat["index_return_long"]

    sma_fast = close.rolling(sma_f).mean()
    sma_slow = close.rolling(sma_s).mean()
    ema_fast = close.ewm(span=macd_f, adjust=False).mean()
    ema_slow = close.ewm(span=macd_s, adjust=False).mean()

    feat["price_sma_fast_dev"] = (close - sma_fast) / sma_fast.replace(0, np.nan)
    feat["price_sma_slow_dev"] = (close - sma_slow) / sma_slow.replace(0, np.nan)
    feat["sma_fast_slope"] = sma_fast.pct_change(1)
    feat["sma_slow_slope"] = sma_slow.pct_change(1)

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_n).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_n).mean()
    rs_ratio = gain / loss.replace(0, np.nan)
    feat["RSI"] = 100 - (100 / (1 + rs_ratio))

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=macd_sig, adjust=False).mean()
    feat["macd_hist_norm"] = (macd - macd_signal) / close.replace(0, np.nan)

    feat["vol"] = feat["return_1"].rolling(vol_n).std()

    bb_mid = sma_slow
    bb_std = close.rolling(sma_s).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    vol_sma = volume.rolling(vol_sma_n).mean()
    feat["volume_sma_ratio"] = volume / vol_sma.replace(0, np.nan)

    next_close = close.shift(-1)
    feat["target"] = (next_close > close).astype(np.float64)
    feat["close"] = close

    feat = feat.replace([np.inf, -np.inf], np.nan)
    if for_training:
        feat = feat.dropna(subset=cols + ["target"])
    else:
        feat = feat.dropna(subset=cols)
    return feat


def make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Window ending on day/bar ``i`` predicts ``targets[i]`` (next-bar direction)."""
    if len(features) < lookback:
        return np.empty((0, lookback, features.shape[1])), np.empty((0,))

    xs, ys = [], []
    for i in range(lookback - 1, len(features)):
        xs.append(features[i - lookback + 1 : i + 1])
        ys.append(targets[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
