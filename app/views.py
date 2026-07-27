from django.shortcuts import render
from django.http import JsonResponse
from plotly.offline import plot
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json
import datetime as dt
from urllib.parse import unquote

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection

from .stock_data import (
    resolve_ticker,
    fetch_history,
    fetch_quote_info,
    fetch_multi_quotes,
    is_chinese_market,
    suggest_stocks,
    fetch_tencent_daily,
)


CN_HOME_CODES = (
    "sh600519",  # 贵州茅台
    "sz000001",  # 平安银行
    "sh601318",  # 中国平安
    "sz300750",  # 宁德时代
    "sh600036",  # 招商银行
    "hk00700",   # 腾讯控股
)


def index(request):
    plot_div_left = ""
    recent_stocks = []

    try:
        quotes = fetch_multi_quotes(CN_HOME_CODES)
        if not quotes.empty:
            recent_stocks = json.loads(quotes.reset_index().to_json(orient="records"))

        history_frames = []
        for code in CN_HOME_CODES[:4]:
            hist = fetch_tencent_daily(code, bars=30)
            if hist.empty:
                continue
            series = hist[["Close"]].rename(columns={"Close": code})
            history_frames.append(series)

        if history_frames:
            merged = pd.concat(history_frames, axis=1).dropna(how="all")
            fig_left = go.Figure()
            labels = {
                "sh600519": "贵州茅台",
                "sz000001": "平安银行",
                "sh601318": "中国平安",
                "sz300750": "宁德时代",
            }
            for col in merged.columns:
                fig_left.add_trace(
                    go.Scatter(
                        x=merged.index,
                        y=merged[col],
                        name=labels.get(col, col),
                    )
                )
            fig_left.update_layout(
                paper_bgcolor="#14151b",
                plot_bgcolor="#14151b",
                font_color="white",
                title="中国热门股票近30日收盘价",
            )
            plot_div_left = plot(fig_left, auto_open=False, output_type="div")
    except Exception as exc:
        print(f"index load failed: {exc}")

    return render(
        request,
        "index.html",
        {
            "plot_div_left": plot_div_left,
            "recent_stocks": recent_stocks,
        },
    )


def search(request):
    return render(request, "search.html", {})


def suggest(request):
    query = (request.GET.get("q") or "").strip()
    items = suggest_stocks(query, limit=8) if query else []
    return JsonResponse({"query": query, "items": items})


def ticker(request):
    cn_df = pd.read_csv("app/Data/cn_tickers.csv", dtype={"Symbol": str})
    us_df = pd.read_csv("app/Data/new_tickers.csv", dtype={"Symbol": str})

    # Keep Symbol/Name columns consistent for the table template.
    if "Market" not in us_df.columns:
        us_df = us_df[["Symbol", "Name"]].copy()
        us_df["Market"] = "US"
    cn_view = cn_df[["Symbol", "Name", "Market"]].copy()
    us_view = us_df[["Symbol", "Name", "Market"]].copy()
    combined = pd.concat([cn_view, us_view], ignore_index=True)

    ticker_list = json.loads(combined.reset_index().to_json(orient="records"))
    return render(request, "ticker.html", {"ticker_list": ticker_list})


def _run_linear_forecast(df_ml: pd.DataFrame, number_of_days: int):
    work = df_ml[["Adj Close"]].copy()
    forecast_out = int(number_of_days)
    if len(work) <= forecast_out + 5:
        raise ValueError("not enough history for forecast")

    work["Prediction"] = work[["Adj Close"]].shift(-forecast_out)
    X = np.array(work.drop(["Prediction"], axis=1))
    X = preprocessing.scale(X)
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = np.array(work["Prediction"])
    y = y[:-forecast_out]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    forecast = clf.predict(X_forecast).tolist()
    return confidence, forecast


def predict(request, ticker_value, number_of_days):
    ticker_value = unquote(ticker_value or "").strip()
    resolved = resolve_ticker(ticker_value)
    if resolved is None:
        return render(request, "Invalid_Ticker.html", {})

    try:
        number_of_days = int(number_of_days)
    except (TypeError, ValueError):
        return render(request, "Invalid_Days_Format.html", {})

    if number_of_days < 0:
        return render(request, "Negative_Days.html", {})
    if number_of_days > 365:
        return render(request, "Overflow_days.html", {})
    if number_of_days == 0:
        number_of_days = 1

    try:
        df = fetch_history(resolved, bars=max(180, number_of_days + 60))
        if df is None or df.empty:
            return render(request, "Invalid_Ticker.html", {})
        print(f"Downloaded ticker = {resolved.display} ({resolved.market}) successfully")
    except Exception as exc:
        print(f"download failed: {exc}")
        return render(request, "API_Down.html", {})

    currency_label = {
        "CNY": "CNY / 元",
        "HKD": "HKD / 港币",
        "USD": "USD / 美元",
    }.get(resolved.currency, resolved.currency)

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="market data",
        )
    )
    fig.update_layout(
        title=f"{resolved.display} 股价走势 ({resolved.market_label})",
        yaxis_title=f"Stock Price ({currency_label})",
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white",
    )
    fig.update_xaxes(rangeslider_visible=True)
    plot_div = plot(fig, auto_open=False, output_type="div")

    try:
        confidence, forecast = _run_linear_forecast(df, number_of_days)
    except Exception as exc:
        print(f"forecast failed: {exc}")
        return render(request, "API_Down.html", {})

    pred_dict = {"Date": [], "Prediction": []}
    for i in range(len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])

    pred_df = pd.DataFrame(pred_dict)
    pred_fig = go.Figure([go.Scatter(x=pred_df["Date"], y=pred_df["Prediction"])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(
        paper_bgcolor="#14151b",
        plot_bgcolor="#14151b",
        font_color="white",
        title=f"{resolved.display} 未来 {number_of_days} 日预测",
    )
    plot_div_pred = plot(pred_fig, auto_open=False, output_type="div")

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

    return render(
        request,
        "result.html",
        {
            "plot_div": plot_div,
            "confidence": confidence,
            "forecast": forecast,
            "ticker_value": resolved.display,
            "number_of_days": number_of_days,
            "plot_div_pred": plot_div_pred,
            "Symbol": info.get("Symbol", resolved.display),
            "Name": info.get("Name", resolved.display),
            "Last_Sale": info.get("Last_Sale", "-"),
            "Net_Change": info.get("Net_Change", "-"),
            "Percent_Change": info.get("Percent_Change", "-"),
            "Market_Cap": info.get("Market_Cap", "-"),
            "Country": info.get("Country", "-"),
            "IPO_Year": info.get("IPO_Year", "-"),
            "Volume": info.get("Volume", "-"),
            "Sector": info.get("Sector", info.get("Market", "-")),
            "Industry": info.get("Industry", resolved.market_label),
        },
    )
