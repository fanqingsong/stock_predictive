"""Stock market data helpers for US and Chinese equities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

from .config import settings


def _data_path(name: str) -> Path:
    return settings.DATA_DIR / name

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn",
}

TENCENT_QUOTE_URL = "https://qt.gtimg.cn/q={codes}"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
TENCENT_SUGGEST_URL = "https://smartbox.gtimg.cn/s3/"

# Common Chinese aliases → preferred symbol.
NAME_ALIASES = {
    "茅台": "600519",
    "贵州茅台": "600519",
    "宁德": "300750",
    "宁德时代": "300750",
    "比亚迪": "002594",
    "腾讯": "00700",
    "腾讯控股": "00700",
    "阿里": "09988",
    "阿里巴巴": "09988",
    "美团": "03690",
    "小米": "01810",
    "京东": "09618",
    "五粮液": "000858",
    "招商银行": "600036",
    "招行": "600036",
    "平安银行": "000001",
    "万科": "000002",
    "万科a": "000002",
    "中国平安": "601318",
    "海康": "002415",
    "海康威视": "002415",
    "美的": "000333",
    "美的集团": "000333",
    "东方财富": "300059",
    "立讯": "002475",
    "立讯精密": "002475",
    "迈瑞": "300760",
    "迈瑞医疗": "300760",
    "中国移动": "00941",
}


@dataclass
class ResolvedTicker:
    display: str          # e.g. 600519 / 00700 / AAPL
    market: str           # cn_sh / cn_sz / cn_hk / us
    tencent_code: Optional[str]  # sh600519 / sz000001 / hk00700
    currency: str
    market_label: str
    name: Optional[str] = None


def _strip_ticker(raw: str) -> str:
    return (raw or "").strip().upper().replace(" ", "")


def _from_market_code(market: str, code: str, name: Optional[str] = None) -> Optional[ResolvedTicker]:
    market = (market or "").lower()
    code = (code or "").strip()
    if not code:
        return None

    if market in {"sh", "ss"}:
        code = code.zfill(6)
        return ResolvedTicker(code, "cn_sh", f"sh{code}", "CNY", "上交所", name)
    if market == "sz":
        code = code.zfill(6)
        return ResolvedTicker(code, "cn_sz", f"sz{code}", "CNY", "深交所", name)
    if market == "hk":
        code = code.zfill(5)
        return ResolvedTicker(code, "cn_hk", f"hk{code}", "HKD", "港交所", name)
    if market in {"us", "nasdaq", "nyse"}:
        return ResolvedTicker(code.upper(), "us", None, "USD", "美股", name)
    return None


def resolve_code(raw: str) -> Optional[ResolvedTicker]:
    """Resolve pure ticker / code style input."""
    ticker = _strip_ticker(raw)
    if not ticker:
        return None

    # Explicit Yahoo-style Chinese suffixes.
    m = re.fullmatch(r"(\d{6})\.(SS|SH)", ticker)
    if m:
        return _from_market_code("sh", m.group(1))

    m = re.fullmatch(r"(\d{6})\.SZ", ticker)
    if m:
        return _from_market_code("sz", m.group(1))

    m = re.fullmatch(r"(\d{1,5})\.HK", ticker)
    if m:
        return _from_market_code("hk", m.group(1))

    # Prefixed Chinese codes: SH600519 / SZ000001 / HK00700
    m = re.fullmatch(r"(SH|SS)(\d{6})", ticker)
    if m:
        return _from_market_code("sh", m.group(2))

    m = re.fullmatch(r"SZ(\d{6})", ticker)
    if m:
        return _from_market_code("sz", m.group(1))

    m = re.fullmatch(r"HK(\d{1,5})", ticker)
    if m:
        return _from_market_code("hk", m.group(1))

    # Pure digits → Chinese A-share / HK by code rules.
    if re.fullmatch(r"\d{6}", ticker):
        if ticker.startswith(("5", "6", "9")):
            return _from_market_code("sh", ticker)
        if ticker.startswith(("0", "1", "2", "3")):
            return _from_market_code("sz", ticker)

    if re.fullmatch(r"\d{1,5}", ticker):
        return _from_market_code("hk", ticker)

    # Fallback: US / international Yahoo ticker.
    if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", ticker):
        return ResolvedTicker(ticker, "us", None, "USD", "美股")

    return None


def _load_cn_tickers() -> pd.DataFrame:
    try:
        return pd.read_csv(_data_path("cn_tickers.csv"), dtype={"Symbol": str})
    except Exception:
        return pd.DataFrame(columns=["Symbol", "Name", "Market", "Exchange"])


def lookup_stock_name(symbol: str, market: str = "") -> str:
    """Resolve a display name for a ticker from local catalogs, then live quote."""
    code = (symbol or "").strip()
    if not code:
        return ""

    df = _load_cn_tickers()
    if not df.empty and "Symbol" in df.columns and "Name" in df.columns:
        matched = df[df["Symbol"].astype(str).str.strip() == code]
        if not matched.empty:
            name = str(matched.iloc[0].get("Name", "")).strip()
            if name:
                return name

    for filename in ("Tickers.csv", "new_tickers.csv"):
        try:
            us = pd.read_csv(_data_path(filename), dtype={"Symbol": str})
            if "Symbol" not in us.columns or "Name" not in us.columns:
                continue
            matched = us[us["Symbol"].astype(str).str.strip().str.upper() == code.upper()]
            if not matched.empty:
                name = str(matched.iloc[0].get("Name", "")).strip()
                if name:
                    return name
        except Exception:
            continue

    # Fallback: Tencent quote for CN / HK codes (local CSV coverage is limited).
    if market.startswith("cn_") or re.fullmatch(r"\d{1,6}", code):
        resolved = resolve_code(code)
        if resolved and resolved.tencent_code:
            try:
                quote = fetch_tencent_quote(resolved.tencent_code)
                name = str((quote or {}).get("Name", "")).strip()
                if name:
                    return name
            except Exception:
                pass
    return ""


def _local_name_candidates(query: str) -> list:
    q = (query or "").strip()
    if not q:
        return []

    results = []
    alias_key = q.lower() if re.fullmatch(r"[A-Za-z0-9]+", q) else q
    if alias_key in NAME_ALIASES:
        resolved = resolve_code(NAME_ALIASES[alias_key])
        if resolved:
            results.append(
                {
                    "symbol": resolved.display,
                    "name": q,
                    "market": resolved.market_label,
                    "tencent_code": resolved.tencent_code,
                    "score": 100,
                }
            )

    df = _load_cn_tickers()
    if df.empty:
        return results

    for _, row in df.iterrows():
        name = str(row.get("Name", "")).strip()
        symbol = str(row.get("Symbol", "")).strip()
        if not name or not symbol:
            continue
        score = 0
        if name == q:
            score = 95
        elif q in name:
            score = 80
        elif name in q and len(name) >= 2:
            score = 70
        if score:
            results.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "market": str(row.get("Exchange", row.get("Market", "CN"))),
                    "tencent_code": None,
                    "score": score,
                }
            )

    # Deduplicate by symbol, keep highest score.
    best = {}
    for item in results:
        prev = best.get(item["symbol"])
        if prev is None or item["score"] > prev["score"]:
            best[item["symbol"]] = item
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def _decode_suggest_text(value: str) -> str:
    text = value or ""
    if "\\u" in text:
        try:
            return text.encode("utf-8").decode("unicode_escape")
        except Exception:
            return text
    return text


def _parse_tencent_suggest(text: str) -> list:
    # v_hint="sh~600519~贵州茅台~gzmt~GP-A^hk~00700~腾讯控股~txkg~GP"
    if "v_hint=" not in text:
        return []
    payload = text.split("=", 1)[1].strip().strip(";").strip('"')
    if not payload or payload == "N":
        return []

    items = []
    for chunk in payload.split("^"):
        parts = chunk.split("~")
        if len(parts) < 5:
            continue
        market, code, name, _pinyin, kind = parts[:5]
        name = _decode_suggest_text(name)
        kind = (kind or "").upper()
        # Skip indexes / funds / bonds / RMB counters when possible.
        if kind.startswith(("ZS", "JJ", "ZQ", "QH", "WH", "HG")):
            continue
        if name.endswith("r") and market.lower() == "hk":
            continue
        if not kind.startswith("GP") and kind not in {"", "BK"}:
            continue
        resolved = _from_market_code(market, code, name)
        if not resolved:
            continue
        items.append(
            {
                "symbol": resolved.display,
                "name": name,
                "market": resolved.market_label,
                "tencent_code": resolved.tencent_code,
                "kind": kind,
                "resolved": resolved,
            }
        )
    return items


def suggest_stocks(query: str, limit: int = 8) -> list:
    """Return stock suggestions for code / Chinese name / pinyin."""
    q = (query or "").strip()
    if not q:
        return []

    suggestions = []

    # Code-style input: also return itself if resolvable.
    code_resolved = resolve_code(q)
    if code_resolved:
        suggestions.append(
            {
                "symbol": code_resolved.display,
                "name": code_resolved.name or code_resolved.display,
                "market": code_resolved.market_label,
                "label": f"{code_resolved.display} {code_resolved.name or ''}".strip(),
            }
        )

    for item in _local_name_candidates(q):
        suggestions.append(
            {
                "symbol": item["symbol"],
                "name": item["name"],
                "market": item["market"],
                "label": f"{item['symbol']} {item['name']}",
            }
        )

    try:
        resp = requests.get(
            TENCENT_SUGGEST_URL,
            params={"v": "2", "q": q, "t": "all"},
            headers=HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        for item in _parse_tencent_suggest(resp.text):
            suggestions.append(
                {
                    "symbol": item["symbol"],
                    "name": item["name"],
                    "market": item["market"],
                    "label": f"{item['symbol']} {item['name']}",
                }
            )
    except Exception as exc:
        print(f"suggest api failed: {exc}")

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for item in suggestions:
        key = item["symbol"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def resolve_by_name(raw: str) -> Optional[ResolvedTicker]:
    """Resolve Chinese/English stock name to ticker."""
    q = (raw or "").strip()
    if not q:
        return None

    local = _local_name_candidates(q)
    if local:
        top = local[0]
        resolved = resolve_code(top["symbol"])
        if resolved:
            resolved.name = top["name"]
            return resolved

    try:
        resp = requests.get(
            TENCENT_SUGGEST_URL,
            params={"v": "2", "q": q, "t": "all"},
            headers=HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        items = _parse_tencent_suggest(resp.text)
    except Exception as exc:
        print(f"name lookup failed: {exc}")
        return None

    if not items:
        return None

    # Prefer exact name, then startswith, then first equity hit.
    exact = [i for i in items if i["name"] == q]
    if exact:
        return exact[0]["resolved"]

    prefix = [i for i in items if i["name"].startswith(q)]
    if prefix:
        return prefix[0]["resolved"]

    contains = [i for i in items if q in i["name"]]
    if contains:
        return contains[0]["resolved"]

    return items[0]["resolved"]


def resolve_ticker(raw: str) -> Optional[ResolvedTicker]:
    """Normalize user input (code or stock name) into a market-specific ticker."""
    text = (raw or "").strip()
    if not text:
        return None

    # Prefer code resolution for ticker-like input.
    by_code = resolve_code(text)
    if by_code and re.fullmatch(
        r"(?i)(\d{1,6}|[A-Z]{1,5}\d{0,4}|[A-Z][A-Z0-9.\-]{0,14}|\d{6}\.(SS|SH|SZ)|\d{1,5}\.HK|(SH|SS|SZ|HK)\d{1,6})",
        text.replace(" ", ""),
    ):
        return by_code

    # Chinese name / alias / pinyin style.
    if re.search(r"[\u4e00-\u9fff]", text) or text.lower() in NAME_ALIASES or len(text) >= 2:
        by_name = resolve_by_name(text)
        if by_name:
            return by_name

    # Fallback to code again (e.g. AAPL).
    return by_code


def is_chinese_market(market: str) -> bool:
    return market in {"cn_sh", "cn_sz", "cn_hk"}


def _parse_tencent_quote(raw: str) -> Optional[dict]:
    # v_sh600519="1~贵州茅台~600519~1289.50~..."
    if "=" not in raw or "~" not in raw:
        return None
    payload = raw.split("=", 1)[1].strip().strip(";").strip('"')
    parts = payload.split("~")
    if len(parts) < 45:
        return None

    try:
        price = float(parts[3]) if parts[3] else 0.0
        prev_close = float(parts[4]) if parts[4] else 0.0
        change = float(parts[31]) if parts[31] else price - prev_close
        pct = float(parts[32]) if parts[32] else (
            (change / prev_close * 100) if prev_close else 0.0
        )
        volume = float(parts[6]) if parts[6] else 0.0
        market_cap = parts[44] if len(parts) > 44 and parts[44] else "-"
    except (TypeError, ValueError):
        return None

    return {
        "Symbol": parts[2],
        "Name": parts[1],
        "Last_Sale": price,
        "Net_Change": round(change, 4),
        "Percent_Change": f"{pct:.3f}%",
        "Market_Cap": market_cap,
        "Country": "China",
        "IPO_Year": "-",
        "Volume": volume,
        "Sector": "-",
        "Industry": "-",
    }


def fetch_tencent_quote(tencent_code: str) -> Optional[dict]:
    url = TENCENT_QUOTE_URL.format(codes=tencent_code)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    resp.encoding = "gbk"
    return _parse_tencent_quote(resp.text)


def fetch_tencent_daily(tencent_code: str, bars: int = 180) -> pd.DataFrame:
    """Return OHLCV dataframe indexed by date."""
    param = f"{tencent_code},day,,,{bars},qfq"
    resp = requests.get(
        TENCENT_KLINE_URL,
        params={"param": param},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    node = (payload.get("data") or {}).get(tencent_code) or {}
    series = node.get("qfqday") or node.get("day") or []
    if not series:
        return pd.DataFrame()

    rows = []
    for item in series:
        # [date, open, close, high, low, volume]
        rows.append(
            {
                "Date": pd.to_datetime(item[0]),
                "Open": float(item[1]),
                "Close": float(item[2]),
                "High": float(item[3]),
                "Low": float(item[4]),
                "Volume": float(item[5]),
            }
        )

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    df["Adj Close"] = df["Close"]
    return df


def fetch_us_daily(symbol: str, period: str = "6mo") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(tickers=symbol, period=period, interval="1d", progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    return df


def _us_period_for_bars(bars: int) -> str:
    if bars <= 160:
        return "6mo"
    if bars <= 320:
        return "1y"
    if bars <= 800:
        return "2y"
    if bars <= 1600:
        return "5y"
    return "10y"


def fetch_history(resolved: ResolvedTicker, bars: int = 180) -> pd.DataFrame:
    """Fetch daily OHLCV; fall back to smaller windows if the API rejects large bars."""
    attempts = [bars]
    for candidate in (2000, 1300, 800, 320):
        if candidate < bars and candidate not in attempts:
            attempts.append(candidate)

    last_empty = pd.DataFrame()
    for n in attempts:
        if is_chinese_market(resolved.market):
            df = fetch_tencent_daily(resolved.tencent_code, bars=n)
        else:
            df = fetch_us_daily(resolved.display, period=_us_period_for_bars(n))
        if df is not None and not df.empty:
            return df
        last_empty = df if df is not None else last_empty
    return last_empty


def fetch_quote_info(resolved: ResolvedTicker) -> dict:
    defaults = {
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
        "Market": resolved.market_label,
        "Currency": resolved.currency,
    }

    if is_chinese_market(resolved.market):
        quote = fetch_tencent_quote(resolved.tencent_code)
        if quote:
            defaults.update(quote)
            defaults["Market"] = resolved.market_label
            defaults["Currency"] = resolved.currency
            defaults["Country"] = "China / Hong Kong" if resolved.market == "cn_hk" else "China"
        return defaults

    # Prefer local US CSV if present.
    try:
        ticker = pd.read_csv(_data_path("Tickers.csv"))
        ticker.columns = [
            "Symbol",
            "Name",
            "Last_Sale",
            "Net_Change",
            "Percent_Change",
            "Market_Cap",
            "Country",
            "IPO_Year",
            "Volume",
            "Sector",
            "Industry",
        ]
        matched = ticker[ticker["Symbol"] == resolved.display]
        if not matched.empty:
            row = matched.iloc[0].to_dict()
            defaults.update(row)
            defaults["Market"] = resolved.market_label
            defaults["Currency"] = resolved.currency
            return defaults
    except Exception:
        pass

    return defaults


def fetch_multi_quotes(tencent_codes: Tuple[str, ...]) -> pd.DataFrame:
    url = TENCENT_QUOTE_URL.format(codes=",".join(tencent_codes))
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    resp.encoding = "gbk"
    rows = []
    for chunk in resp.text.strip().split(";"):
        quote = _parse_tencent_quote(chunk.strip())
        if quote:
            rows.append(
                {
                    "Ticker": quote["Symbol"],
                    "Name": quote["Name"],
                    "Open": quote["Last_Sale"],
                    "High": quote["Last_Sale"],
                    "Low": quote["Last_Sale"],
                    "Close": quote["Last_Sale"],
                    "Adj_Close": quote["Last_Sale"],
                    "Volume": quote["Volume"],
                    "Change": quote["Net_Change"],
                    "Percent_Change": quote["Percent_Change"],
                }
            )
    return pd.DataFrame(rows)
