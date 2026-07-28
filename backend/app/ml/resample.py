"""Resample daily OHLCV into natural week / natural month bars."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "Date" in work.columns:
            work = work.set_index("Date")
        work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    if "Adj Close" not in work.columns and "Close" in work.columns:
        work["Adj Close"] = work["Close"]
    if "Volume" not in work.columns:
        work["Volume"] = 0.0
    return work


def _aggregate_groups(daily: pd.DataFrame, period_index: pd.Series) -> pd.DataFrame:
    work = daily.copy()
    work["_period"] = period_index
    grouped = work.groupby("_period", sort=True)

    rows = []
    for _, g in grouped:
        g = g.sort_index()
        last_ts = g.index.max()
        rows.append(
            {
                "Date": last_ts,
                "Open": float(g["Open"].iloc[0]),
                "High": float(g["High"].max()),
                "Low": float(g["Low"].min()),
                "Close": float(g["Close"].iloc[-1]),
                "Adj Close": float(g["Adj Close"].iloc[-1]),
                "Volume": float(g["Volume"].sum()),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("Date").sort_index()
    return out


def to_natural_week(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trading days into calendar weeks (ISO week; index = last session)."""
    daily = _ensure_ohlcv(daily_df)
    if daily.empty:
        return daily
    # Week ending Sunday (pandas period); groups Mon–Sun calendar week.
    period = daily.index.to_period("W-SUN")
    return _aggregate_groups(daily, period)


def to_natural_month(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trading days into calendar months (index = last session of month)."""
    daily = _ensure_ohlcv(daily_df)
    if daily.empty:
        return daily
    period = daily.index.to_period("M")
    return _aggregate_groups(daily, period)


def drop_incomplete_bars(
    bars: pd.DataFrame,
    *,
    timeframe: str,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Drop the last bar if its natural period is still in progress as of ``as_of``."""
    if bars is None or bars.empty or timeframe == "day":
        return bars

    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.utcnow().tz_localize(None)
    if getattr(ref, "tzinfo", None) is not None:
        ref = ref.tz_localize(None)
    ref = ref.normalize()
    out = bars.sort_index()
    last = out.index[-1]
    if timeframe == "week":
        if last.to_period("W-SUN") == ref.to_period("W-SUN"):
            out = out.iloc[:-1]
    elif timeframe == "month":
        if last.to_period("M") == ref.to_period("M"):
            out = out.iloc[:-1]
    return out


def bars_for_timeframe(
    daily_df: pd.DataFrame,
    timeframe: str,
    *,
    as_of: Optional[pd.Timestamp] = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Daily passthrough, or natural week/month bars with incomplete period dropped."""
    daily = _ensure_ohlcv(daily_df)
    if timeframe == "day":
        return daily
    if timeframe == "week":
        bars = to_natural_week(daily)
    elif timeframe == "month":
        bars = to_natural_month(daily)
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    if drop_incomplete:
        bars = drop_incomplete_bars(bars, timeframe=timeframe, as_of=as_of)
    return bars
