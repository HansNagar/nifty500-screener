import yfinance as yf
import ta
from ta.volatility import BollingerBands
import pandas as pd
from indicators import supertrend
import streamlit as st

def to_crore(val):
    try:
        val = float(val)
        if abs(val) >= 1e7:
            return f"{val/1e7:.2f} Cr"
        elif abs(val) >= 1e5:
            return f"{val/1e5:.2f} L"
        else:
            return f"{val:,.0f}"
    except Exception:
        return val

def format_df_crores(df):
    df_fmt = df.copy()
    for col in df_fmt.columns:
        df_fmt[col] = df_fmt[col].apply(to_crore)
    return df_fmt

@st.cache_data(show_spinner=False)
def get_data(ticker):
    df = yf.Ticker(ticker).history(period="6mo")
    df['SMA_20']     = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50']     = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200']    = ta.trend.sma_indicator(df['Close'], window=200)
    df['RSI']        = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support']    = df['Low'].rolling(window=20).min()
    # Add Bollinger Bands
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_UP']  = bb.bollinger_hband()
    df['BB_LOW'] = bb.bollinger_lband()
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
    prev_macd, prev_sig = df['MACD'].iat[-2], df['MACD_Signal'].iat[-2]
    curr_macd, curr_sig = df['MACD'].iat[-1], df['MACD_Signal'].iat[-1]
    if prev_macd <= prev_sig and curr_macd > curr_sig:
        return "Buy 🔄 MACD Bullish Crossover"
    if prev_macd >= prev_sig and curr_macd < curr_sig:
        return "Sell 🔄 MACD Bearish Crossunder"
    return None

def scan_price_above_sma200(df):
    if df.shape[0] < 200:
        return None
    if df['Close'].iat[-1] > df['SMA_200'].iat[-1]:
        return "Buy 🟢 Price above SMA200"
    return None

def scan_price_below_sma200(df):
    if df.shape[0] < 200:
        return None
    if df['Close'].iat[-1] < df['SMA_200'].iat[-1]:
        return "Sell 🔴 Price below SMA200"
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
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_UP, name="BB Up"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_LOW, name="BB Low"), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', row=3, col=1)

    fig.update_layout(template="plotly_dark", height=900)
    return fig

def get_financial_data(ticker):
    tkr = yf.Ticker(ticker)
    fin = tkr.financials.iloc[::-1] if not tkr.financials.empty else pd.DataFrame()
    bs  = tkr.balance_sheet.iloc[::-1] if not tkr.balance_sheet.empty else pd.DataFrame()
    cf  = tkr.cashflow.iloc[::-1] if not tkr.cashflow.empty else pd.DataFrame()

    info = tkr.info or {}
    ratios = {
        "P/E (ttm)": info.get("trailingPE"),
        "P/E (fwd)": info.get("forwardPE"),
        "Div Yield": info.get("dividendYield"),
        "Market Cap": to_crore(info.get("marketCap")),
        "ROE": info.get("returnOnEquity"),
        "ROA": info.get("returnOnAssets"),
        "ROCE": info.get("returnOnCapitalEmployed"),
        "Debt/Equity": info.get("debtToEquity"),
        "Current Ratio": info.get("currentRatio"),
        "Quick Ratio": info.get("quickRatio"),
        "EPS (ttm)": to_crore(info.get("trailingEps")),
        "EPS (fwd)": to_crore(info.get("forwardEps")),
        "PB Ratio": info.get("priceToBook"),
        "PEG Ratio": info.get("pegRatio"),
        "Operating Margin": info.get("operatingMargins"),
        "Profit Margin": info.get("profitMargins"),
        "Gross Margin": info.get("grossMargins"),
        "Beta": info.get("beta"),
        "52W High": to_crore(info.get("fiftyTwoWeekHigh")),
        "52W Low": to_crore(info.get("fiftyTwoWeekLow")),
    }
    # Format financials in crores/lakhs
    if not fin.empty:
        fin = format_df_crores(fin)
    if not bs.empty:
        bs = format_df_crores(bs)
    if not cf.empty:
        cf = format_df_crores(cf)
    return fin, bs, cf, ratios
