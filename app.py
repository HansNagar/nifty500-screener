import os
import json
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import concurrent.futures
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import (
    get_data, auto_analysis, generate_chart,
    scan_20d_breakout, scan_golden_cross, scan_bb_breakout,
    scan_rsi_reversion, scan_macd_crossover, get_financial_data
)

WATCHLIST_FILE = "watchlist.json"
CSV_URL_500 = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


# =====================
# 📥 WATCHLIST HELPERS
# =====================


def load_watchlist():

    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, "r") as f:
                data = json.load(f)

                if isinstance(data, list) and data and isinstance(data[0], str):
                    return [{"Company": c, "Strategy": "", "Signal": ""} for c in data]

                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(watchlist, f, indent=2)


def load_nifty5000():
    try:
        df = pd.read_csv(CSV_URL_500)
        df.columns = [c.strip() for c in df.columns]
        return {row["Company Name"]: f"{row['Symbol']}.NS" for _, row in df.iterrows()}
    except Exception:
        st.error("Failed to load Nifty 500 list.")
        return {}


def fetch_company_meta(symbol):
    tkr = yf.Ticker(symbol)
    info = tkr.info or {}
    return {
        "Sector": info.get("sector", "N/A"),
        "Industry": info.get("industry", "N/A"),
        "MarketCap": info.get("marketCap", 0)

    }

strategies = {
    "20D Breakout/Breakdown": scan_20d_breakout,
    "Golden/Death Cross": scan_golden_cross,
    "BB Breakout/Breakdown": scan_bb_breakout,
    "RSI Mean-Reversion": scan_rsi_reversion,
    "MACD Crossover": scan_macd_crossover

}


# =====================
# ⚙️ STRATEGY EXECUTION
# =====================

def run_strategy_scan(name: str) -> pd.DataFrame:
    fn = strategies[name]
    rows = []
    total = len(all_companies)
    st.info(f"Scanning {total} stocks using strategy: {name}")
    progress = st.progress(0)
    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for comp in all_companies:
            ticker = nifty_dict[comp]
            futures[executor.submit(lambda t, f=fn: f(get_data(t)), ticker)] = comp

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            comp = futures[future]
            try:
                sig = future.result()
                progress.progress((i + 1) / total)
                if sig:
                    rows.append({'Company': comp, 'Signal': sig})
            except Exception:
                pass
    return pd.DataFrame(rows)


# =====================
# 🚀 APP CONFIG + INIT
# =====================

st.set_page_config(page_title='Nifty 500 Screener', layout='wide', page_icon='📊')
nifty_dict = load_nifty5000()
all_companies = sorted(nifty_dict.keys())
view = st.sidebar.radio("Select View", ["Dashboard", "Strategy Scanner", "Watchlist"])


# =====================
# 📊 DASHBOARD VIEW
# =====================

if view == "Dashboard":
    st.sidebar.subheader("🔍 Search Company")
    query = st.sidebar.text_input("Type name or symbol", "")
    opts = (
        [name for name in all_companies if query.lower() in name.lower()]
        if query else all_companies
    )
    if not opts:
        st.warning("No matching companies.")
        st.stop()

    company = st.sidebar.selectbox("Select Company", opts)
    symbol = nifty_dict[company]
    df = get_data(symbol)
    tkr = yf.Ticker(symbol)

    current_price = df["Close"].iat[-1]
    sma_20 = df["SMA_20"].iat[-1]
    sma_50 = df["Close"].rolling(window=50).mean().iat[-1]
    avg_vol = int(df["Volume"].rolling(window=20).mean().iat[-1])
    last_updated = df.index[-1].strftime("%d %b %Y")

    info = tkr.info or {}
    industry = info.get("industry", "N/A")
    website = info.get("website", "")
    domain = website.split("//")[-1].split("/")[0] if website else None
    logo_url = f"https://logo.clearbit.com/{domain}" if domain else None

    try:
        _, _, _, ratios = get_financial_data(symbol)
    except Exception:
        ratios = {}

    st.title(f"📦 {company}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Price", f"{current_price:.2f}")
        st.metric("RSI", f"{df['RSI'].iat[-1]:.2f}")
    with col2:
        st.metric("SMA 20", f"{sma_20:.2f}")
        st.metric("SMA 50", f"{sma_50:.2f}")
    with col3:
        st.metric("Avg Vol (20D)", f"{avg_vol}")
        st.metric("Industry", industry)
    with col4:
        pe = ratios.get("P/E (ttm)")
        mcap = info.get("marketCap")
        st.metric("P/E (ttm)", f"{pe:.2f}" if pe else "N/A")
        st.metric("Market Cap", f"{mcap/1e9:.2f}B" if mcap else "N/A")

    if logo_url:
        st.image(logo_url, width=80)
    st.caption(f"Last Updated: {last_updated}")


    if st.checkbox("📉 Show Technical Chart", value=False):
        st.plotly_chart(generate_chart(df), use_container_width=True)


    with st.expander("🧠 Technical Insights", expanded=False):

        for insight in auto_analysis(df):
            st.markdown(f"- {insight}")


    with st.expander("📊 Financial Statements & Ratios", expanded=False):
        try:
            fin, bs, cf, rts = get_financial_data(symbol)
            st.subheader("Income Statement (Annual)")
            st.dataframe(fin)
            st.subheader("Balance Sheet (Annual)")
            st.dataframe(bs)
            st.subheader("Cash Flow (Annual)")
            st.dataframe(cf)
            st.subheader("Key Financial Ratios")
            st.table(pd.DataFrame.from_dict(rts, orient='index', columns=['Value']))
        except Exception as e:
            st.error(f"Error fetching financial data: {e}")


# =====================
# 🔍 STRATEGY SCANNER
# =====================

elif view == "Strategy Scanner":
    st.title("🔍 Strategy Scanner")
    strategy = st.sidebar.selectbox("Pick a Strategy", list(strategies.keys()))
    drill_prefix = st.sidebar.text_input("Drill-down Prefix", "")


    if st.sidebar.button("▶️ Run Scan"):
        df_scan = run_strategy_scan(strategy)
        meta_rows = []
        for comp in df_scan["Company"]:
            sym = nifty_dict[comp]
            meta = fetch_company_meta(sym)
            meta["Company"] = comp
            df_tmp = get_data(sym)
            meta["AvgVol20D"] = int(df_tmp["Volume"].rolling(20).mean().iat[-1])
            meta_rows.append(meta)
        df_merged = df_scan.merge(pd.DataFrame(meta_rows), on="Company")
        st.session_state.df_merged = df_merged
        st.session_state.last_scan_signals = dict(zip(df_scan["Company"], df_scan["Signal"]))
        st.session_state.last_scan_strategy = strategy

    df_merged = st.session_state.get("df_merged")
    if df_merged is None:
        st.info("Click ▶️ Run Scan to generate signals.")
        st.stop()

    df_display = df_merged if not drill_prefix else df_merged[df_merged["Company"].str.startswith(drill_prefix)]
    if df_display.empty:
        st.info("No signals found.")
        st.stop()

# =====================
# 📋 WATCHLIST VIEW
# =====================

elif view == "Watchlist":
    entries = load_watchlist()
    df_wl = pd.DataFrame(entries) if entries else pd.DataFrame(columns=["Company", "Strategy", "Signal"])

    if not entries:
        st.info("Your watchlist is empty.")
    else:
        signals_col = df_wl["Signal"].fillna("").astype(str)
        total = len(df_wl)
        buys = signals_col.str.contains("Buy").sum()
        sells = signals_col.str.contains("Sell").sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Buy Signals", buys)
        col3.metric("Sell Signals", sells)

# =====================
# 📋 WATCHLIST DETAILS
# =====================

# =====================
# 📋 WATCHLIST ENTRIES
# =====================
if not entries:
    st.info("Your watchlist is empty.")
else:
    pass  # Placeholder to fix indentation error
for entry in entries.copy():
    comp = entry["Company"]
    sig = entry.get("Signal", "")
    sym = nifty_dict.get(comp)
    df = get_data(sym) if sym else None
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("❌ Remove", key=f"rm_{comp}"):
            entries.remove(entry)
            save_watchlist(entries)
            st.experimental_rerun()
    with col2:
        st.subheader(comp)
        st.write(f"Signal: {sig}")
        if df is not None:
            st.write(f"Price: {df['Close'].iat[-1]:.2f}")
        meta = fetch_company_meta(sym) if sym else {}
        st.write(f"Industry: {meta.get('Industry','N/A')}")
sel = st.selectbox("Select for Details", df_wl["Company"].tolist())
sym = nifty_dict.get(sel)
if sym:
    df_sel = get_data(sym)
    price = df_sel['Close'].iat[-1]
    rsi = df_sel['RSI'].iat[-1]
    sma20 = df_sel['SMA_20'].iat[-1]
    avgvol = int(df_sel['Volume'].rolling(window=20).mean().iat[-1])
    sig = st.session_state.get("last_scan_signals", {}).get(sel, "")
    st.subheader(f"Details for {sel}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Price", f"{price:.2f}")
        st.metric("RSI", f"{rsi:.2f}")
    with c2:
        st.metric("SMA 20", f"{sma20:.2f}")
        st.metric("Avg Vol", f"{avgvol}")
    with c3:
        st.metric("Signal", sig)
    with c4:
        st.write(f"Industry: {fetch_company_meta(sym).get('Industry','N/A')}")
    if st.checkbox("📈 Show Chart", key=f"wl_chart_{sel}"):
        st.plotly_chart(generate_chart(df_sel), use_container_width=True)
    if st.button("❌ Remove from Watchlist"):
        new_entries = [e for e in entries if e.get("Company") != sel]
        save_watchlist(new_entries)
        st.success(f"{sel} removed from watchlist.")
        st.experimental_rerun()

        for entry in entries.copy():
            comp = entry["Company"]
            sig = entry.get("Signal", "")
            sym = nifty_dict.get(comp)
            df = get_data(sym) if sym else None
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("❌ Remove", key=f"rm_{comp}"):
                    entries.remove(entry)
                    save_watchlist(entries)
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
            with col2:
                st.subheader(comp)
                st.write(f"Signal: {sig}")
                if df is not None:
                    st.write(f"Price: {df['Close'].iat[-1]:.2f}")
                meta = fetch_company_meta(sym) if sym else {}
                st.write(f"Industry: {meta.get('Industry','N/A')}")

        new_entries = [e for e in entries if e.get("Company") != sel]
    df_display = st.session_state.get("df_merged")
    if df_display is None or df_display.empty:
        st.info("Click ▶️ Run Scan to generate signals.")
        st.stop()
    st.table(df_display)

    df_display = st.session_state.get("df_merged")
    if df_display is None or df_display.empty:
        st.info("Click ▶️ Run Scan to generate signals.")
        st.stop()
    st.table(df_display)

    if not df_display.empty:
        st.subheader("📖 Signal Details")
        selected = st.selectbox("Select Company to view details", df_display["Company"].tolist())
        symbol_sel = nifty_dict[selected]
        df_sel = get_data(symbol_sel)

        price = df_sel["Close"].iat[-1]
        rsi = df_sel["RSI"].iat[-1]
        sma20 = df_sel["SMA_20"].iat[-1]
        avgvol = int(df_sel["Volume"].rolling(window=20).mean().iat[-1])
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price", f"{price:.2f}")
            st.metric("RSI", f"{rsi:.2f}")
        with col2:
            st.metric("SMA 20", f"{sma20:.2f}")
            st.metric("Avg Vol (20D)", f"{avgvol}")
        with col3:
            st.write(f"Industry: {fetch_company_meta(symbol_sel).get('Industry','N/A')}")
        with col4:
            sig = st.session_state.get("last_scan_signals", {}).get(selected, "")
            st.write(f"Signal: {sig}")


        if st.button("➕ Add to Watchlist", key=f"add_{selected}"):
            wl = load_watchlist()

            if not any(item["Company"] == selected for item in wl):
                wl.append({"Company": selected, "Strategy": st.session_state.get('last_scan_strategy',''), "Signal": sig})
                save_watchlist(wl)
                st.success(f"{selected} added to watchlist.")
            else:
                st.info(f"{selected} is already in your watchlist.")


    def color_signal(val):
        if "Buy" in val:
            return "color: green"
        if "Sell" in val:
            return "color: red"
        return ""


else:
    st.warning("Invalid view selected.")





































































































































































