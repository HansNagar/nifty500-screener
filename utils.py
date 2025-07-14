# utils.py

import yfinance as yf
import ta
from ta.volatility import BollingerBands
import pandas as pd
from indicators import supertrend

def get_data(ticker):
    df = yf.Ticker(ticker).history(period="6mo")
    df['SMA_20']     = ta.trend.sma_indicator(df['Close'], window=20)
    df['RSI']        = ta.momentum.rsi(df['Close'], window=14)
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support']    = df['Low'].rolling(window=20).min()
    return supertrend(df)

def scan_20d_breakout(df):
    if df.shape[0] < 21:
        return None
    prev, curr = df.iloc[-2], df.iloc[-1]
    if prev.Close <= prev.Resistance and curr.Close > prev.Resistance:
        return "Buy 🔼 20D Breakout"
    if prev.Close >= prev.Support and curr.Close < prev.Support:
        return "Sell 🔽 20D Breakdown"
    return None

def scan_golden_cross(df):
    if df.shape[0] < 201:
        return None
    df['SMA_50']  = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    prev50, prev200 = df['SMA_50'].iat[-2], df['SMA_200'].iat[-2]
    cur50,  cur200  = df['SMA_50'].iat[-1], df['SMA_200'].iat[-1]
    if prev50 <= prev200 and cur50 > cur200:
        return "Buy 🎯 Golden Cross"
    if prev50 >= prev200 and cur50 < cur200:
        return "Sell 🔻 Death Cross"
    return None

def scan_bb_breakout(df):
    if df.shape[0] < 21:
        return None
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_UP']  = bb.bollinger_hband()
    df['BB_LOW'] = bb.bollinger_lband()
    prev, curr = df.iloc[-2], df.iloc[-1]
    if prev.Close <= prev.BB_UP and curr.Close > prev.BB_UP:
        return "Buy ⚡ BB Breakout"
    if prev.Close >= prev.BB_LOW and curr.Close < prev.BB_LOW:
        return "Sell ⚡ BB Breakdown"
    return None

def scan_rsi_reversion(df):
    if df.shape[0] < 15:
        return None
    latest = df['RSI'].iat[-1]
    if latest < 30:
        return "Buy 🔄 RSI Oversold (<30)"
    if latest > 70:
        return "Sell 🔄 RSI Overbought (>70)"
    return None

def scan_macd_crossover(df):
    if df.shape[0] < 35:
        return None
    macd = ta.trend.MACD(df['Close'])
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    prev_macd, prev_sig = df['MACD'].iat[-2], df['MACD_Signal'].iat[-2]
    curr_macd, curr_sig = df['MACD'].iat[-1], df['MACD_Signal'].iat[-1]
    if prev_macd <= prev_sig and curr_macd > curr_sig:
        return "Buy 🔄 MACD Bullish Crossover"
    if prev_macd >= prev_sig and curr_macd < curr_sig:
        return "Sell 🔄 MACD Bearish Crossunder"
    return None

def auto_analysis(df):
    if df.shape[0] < 2:
        return ["Not enough data."]
    latest, prev = df.iloc[-1], df.iloc[-2]
    res = [
        f"📈 Supertrend: **{'UP' if latest.Supertrend else 'DOWN'}**",
        "⚠️ RSI Overbought" if latest.RSI > 70 else
        "✅ RSI Oversold"   if latest.RSI < 30 else
        "ℹ️ RSI Neutral",
        "✅ Price above SMA20" if latest.Close > latest.SMA_20 else "⚠️ Price below SMA20"
    ]
    if prev.Close <= prev.Resistance and latest.Close > latest.Resistance:
        res.append("🚀 Broke 20D High")
    if prev.Close >= prev.Support and latest.Close < latest.Support:
        res.append("🔻 Broke 20D Low")
    return res

def generate_chart(df):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.2, 0.3],
        subplot_titles=["Price + SMA20 + Bands", "Volume", "RSI"]
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df.Open, high=df.High,
        low=df.Low, close=df.Close, name="Candles"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.SMA_20, name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.ST_Upper, name="ST Up"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.ST_Lower, name="ST Low"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.Resistance, name="20D High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.Support, name="20D Low"), row=1, col=1)

    bb = BollingerBands(df.Close, window=20, window_dev=2)
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_hband(), name="BB Up"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb.bollinger_lband(), name="BB Low"), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', row=3, col=1)

    fig.update_layout(template="plotly_dark", height=900)
    return fig

def get_financial_data(ticker):
    tkr = yf.Ticker(ticker)
    fin = tkr.financials.iloc[::-1]
    bs  = tkr.balance_sheet.iloc[::-1]
    cf  = tkr.cashflow.iloc[::-1]

    info = tkr.info or {}
    raw = {
        "P/E (ttm)": info.get("trailingPE"),
        "P/E (fwd)": info.get("forwardPE"),
        "Div Yield": info.get("dividendYield"),
        "Mkt Cap":   info.get("marketCap")
    }
    # ... compute ratios as before ...
    ratios = {
        "ROE": raw["P/E (ttm)"] and (info.get("returnOnEquity")),
        # etc.
    }
    return fin, bs, cf, ratios
