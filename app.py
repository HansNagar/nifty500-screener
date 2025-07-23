import os
import json
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import concurrent.futures
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils import (
    get_data, auto_analysis, generate_chart, format_df_crores,
    scan_20d_breakout, scan_golden_cross, scan_bb_breakout,
    scan_rsi_reversion, scan_macd_crossover, scan_price_above_sma200, scan_price_below_sma200,
    get_financial_data, to_crore
)

WATCHLIST_FILE = "watchlist.json"
CSV_URL_500 = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# =====================
# 📥 WATCHLIST HELPERS
# =====================

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def load_nifty5000():
    try:
        df = pd.read_csv(CSV_URL_500)
        df.columns = [c.strip() for c in df.columns]
        return {row["Company Name"]: f"{row['Symbol']}.NS" for _, row in df.iterrows()}
    except Exception:
        st.error("Failed to load Nifty 500 list.")
        return {}

@st.cache_data(show_spinner=False)
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
    "MACD Crossover": scan_macd_crossover,
    "Price Above SMA200": scan_price_above_sma200,
    "Price Below SMA200": scan_price_below_sma200,
}

def color_signal(val):
    if "Buy" in val:
        return "background-color: #d4f8e8; color: green"
    if "Sell" in val:
        return "background-color: #ffeaea; color: red"
    return ""

def pretty_signal(signal):
    if "Buy" in signal:
        return "🟢 <b style='color:green;'>Buy</b>"
    if "Sell" in signal:
        return "🔴 <b style='color:red;'>Sell</b>"
    return signal

def custom_strategy_rule(df, indicator, operator, value):
    if indicator not in df.columns:
        return False
    last = df[indicator].iloc[-1]
    if operator == ">":
        return last > value
    if operator == "<":
        return last < value
    if operator == "=":
        return last == value
    return False

# =====================
# ⚙️ STRATEGY EXECUTION
# =====================

def run_strategy_scan(name: str, custom_rule=None) -> pd.DataFrame:
    if name == "Custom":
        rows = []
        total = len(all_companies)
        st.info(f"Scanning {total} stocks using custom strategy...")
        progress = st.progress(0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(get_data, nifty_dict[comp]): comp for comp in all_companies}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                comp = futures[future]
                try:
                    df = future.result()
                    if custom_rule(df):
                        rows.append({'Company': comp, 'Signal': "Custom"})
                except Exception:
                    pass
                progress.progress((i + 1) / total)
        return pd.DataFrame(rows)
    else:
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
st.markdown("""
    <style>
    .stMetric {font-size: 18px;}
    .stDataFrame th, .stDataFrame td {font-size: 15px;}
    .signal-badge {padding:2px 8px;border-radius:8px;font-weight:bold;}
    .signal-buy {background:#d4f8e8;color:green;}
    .signal-sell {background:#ffeaea;color:red;}
    </style>
""", unsafe_allow_html=True)

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
    with st.spinner("Loading data..."):
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
        fin, bs, cf, ratios = get_financial_data(symbol)
    except Exception:
        ratios = {}

    st.title(f"📦 {company}")
    if logo_url and website:
        st.markdown(f"[![logo]({logo_url})]({website})", unsafe_allow_html=True)
    elif logo_url:
        st.image(logo_url, width=80)
    st.caption(f"Last Updated: {last_updated}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🧠 Insights", "📊 Financials", "📌 Key Ratios", "📉 Analytics"
    ])
    with tab1:
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
            mcap = ratios.get("Market Cap")
            st.metric("P/E (ttm)", f"{pe:.2f}" if pe else "N/A")
            st.metric("Market Cap", mcap if mcap else "N/A")
        if st.checkbox("📉 Show Technical Chart", value=False):
            st.plotly_chart(generate_chart(df), use_container_width=True)

    with tab2:
        st.subheader("Technical Insights")
        for insight in auto_analysis(df):
            st.markdown(f"- {insight}")

    with tab3:
        try:
            st.subheader("Income Statement (Annual)")
            st.dataframe(format_df_crores(fin), use_container_width=True)
            st.subheader("Balance Sheet (Annual)")
            st.dataframe(format_df_crores(bs), use_container_width=True)
            st.subheader("Cash Flow (Annual)")
            st.dataframe(format_df_crores(cf), use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching financial data: {e}")

    with tab4:
        st.subheader("📌 Key Ratios")
        try:
            _, _, _, rts = get_financial_data(symbol)
            ratio_names = list(rts.keys())
            ratio_selected = st.selectbox("Select Ratio", ratio_names, key=f"dashboard_ratio_{symbol}")
            st.metric(ratio_selected, rts[ratio_selected])
            st.dataframe(pd.DataFrame.from_dict(rts, orient='index', columns=['Value']), use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching ratios: {e}")

    with tab5:
        st.subheader("📉 Analytics")
        import plotly.graph_objs as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
        buy_signals = df[df['RSI'] < 30]
        sell_signals = df[df['RSI'] > 70]
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='RSI Buy', marker=dict(color='green', size=10, symbol='triangle-up')))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='RSI Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Peer Comparison**")
        sector = info.get("sector", None)
        if sector:
            peers = [k for k, v in nifty_dict.items() if fetch_company_meta(v).get("Sector") == sector and k != company]
            if peers:
                peer_data = []
                for peer in peers[:5]:
                    sym_peer = nifty_dict[peer]
                    try:
                        _, _, _, peer_ratios = get_financial_data(sym_peer)
                        peer_data.append({
                            "Company": peer,
                            "P/E": peer_ratios.get("P/E (ttm)", "N/A"),
                            "ROE": peer_ratios.get("ROE", "N/A"),
                            "ROCE": peer_ratios.get("ROCE", "N/A"),
                        })
                    except Exception:
                        continue
                if peer_data:
                    st.dataframe(pd.DataFrame(peer_data), use_container_width=True)
                else:
                    st.info("No peer data available.")
            else:
                st.info("No peers found in same sector.")
        st.markdown("**RSI Strategy Backtest (Buy <30, Sell >70)**")
        rsi_buys = (df['RSI'] < 30).sum()
        rsi_sells = (df['RSI'] > 70).sum()
        st.metric("Buy Signals", rsi_buys)
        st.metric("Sell Signals", rsi_sells)

# =====================
# 🔍 STRATEGY SCANNER
# =====================

elif view == "Strategy Scanner":
    st.title("🔍 Strategy Scanner")
    strategy = st.sidebar.selectbox("Pick a Strategy", list(strategies.keys()) + ["Custom"])
    drill_prefix = st.sidebar.text_input("Drill-down Prefix", "")

    custom_rule = None
    if strategy == "Custom":
        st.sidebar.markdown("**Custom Rule Builder**")
        custom_rules = []
        num_rules = st.sidebar.number_input("Number of Conditions", min_value=1, max_value=5, value=1, step=1)
        for i in range(num_rules):
            indicator = st.sidebar.selectbox(f"Indicator {i+1}", ["Close", "RSI", "SMA_20", "SMA_50", "MACD"], key=f"ind_{i}")
            operator = st.sidebar.selectbox(f"Operator {i+1}", [">", "<", "="], key=f"op_{i}")
            value = st.sidebar.number_input(f"Value {i+1}", value=50.0, key=f"val_{i}")
            custom_rules.append((indicator, operator, value))
        logic = st.sidebar.selectbox("Combine conditions with", ["AND", "OR"])

        def rule(df):
            results = []
            for indicator, operator, value in custom_rules:
                if indicator not in df.columns:
                    results.append(False)
                    continue
                last = df[indicator].iloc[-1]
                if operator == ">":
                    results.append(last > value)
                elif operator == "<":
                    results.append(last < value)
                elif operator == "=":
                    results.append(last == value)
            if logic == "AND":
                return all(results)
            else:
                return any(results)
        custom_rule = rule

        rule_str = f" {logic} ".join([f"{ind} {op} {val}" for ind, op, val in custom_rules])
        st.sidebar.info(f"Rule: {rule_str}")

    if st.sidebar.button("▶️ Run Scan"):
        with st.spinner("Running scan..."):
            df_scan = run_strategy_scan(strategy, custom_rule)
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

    search = st.text_input("🔍 Search results", "")
    df_display = df_merged if not search else df_merged[df_merged["Company"].str.contains(search, case=False)]
    sort_col = st.selectbox("Sort by", ["Company", "Signal", "AvgVol20D"])
    df_display = df_display.sort_values(sort_col)

    if df_display.empty:
        st.info("No signals found.")
        st.stop()

    styled_df = df_display.style.applymap(color_signal, subset=["Signal"])
    st.dataframe(styled_df, use_container_width=True)
    st.download_button("Download CSV", df_display.to_csv(index=False), "scan_results.csv")

    if not df_display.empty:
        st.subheader("📖 Signal Details")
        selected = st.selectbox("Select Company to view details", df_display["Company"].tolist())
        symbol_sel = nifty_dict[selected]
        with st.spinner("Loading details..."):
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
            st.markdown(f"**Signal:** {pretty_signal(sig)}", unsafe_allow_html=True)

        if st.button("➕ Add to Watchlist", key=f"add_{selected}"):
            wl = load_watchlist()
            if not any(item["Company"] == selected for item in wl):
                wl.append({"Company": selected, "Strategy": st.session_state.get('last_scan_strategy',''), "Signal": sig})
                save_watchlist(wl)
                st.success(f"{selected} added to watchlist.")
            else:
                st.info(f"{selected} is already in your watchlist.")

# =====================
# 📋 WATCHLIST VIEW
# =====================

elif view == "Watchlist":
    entries = load_watchlist()
    df_wl = pd.DataFrame(entries) if entries else pd.DataFrame(columns=["Company", "Strategy", "Signal"])

    if not entries:
        st.info("Your watchlist is empty.")
    else:
        search = st.text_input("🔍 Search watchlist", "")
        df_wl_display = df_wl if not search else df_wl[df_wl["Company"].str.contains(search, case=False)]
        sort_col = st.selectbox("Sort by", ["Company", "Signal"], key="wl_sort")
        df_wl_display = df_wl_display.sort_values(sort_col)

        signals_col = df_wl_display["Signal"].fillna("").astype(str)
        total = len(df_wl_display)
        buys = signals_col.str.contains("Buy").sum()
        sells = signals_col.str.contains("Sell").sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", total)
        col2.metric("Buy Signals", buys)
        col3.metric("Sell Signals", sells)

        st.markdown("### Watchlist")
        styled_wl = df_wl_display.style.applymap(color_signal, subset=["Signal"])
        st.dataframe(styled_wl, use_container_width=True)
        st.download_button("Download CSV", df_wl_display.to_csv(index=False), "watchlist.csv")

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
                    st.rerun()
            with col2:
                st.subheader(comp)
                st.write(f"Signal: {sig}")
                if df is not None:
                    st.write(f"Price: {df['Close'].iat[-1]:.2f}")
                meta = fetch_company_meta(sym) if sym else {}
                st.write(f"Industry: {meta.get('Industry','N/A')}")

        st.markdown("### 📋 Watchlist Details")
        if not df_wl_display.empty:
            sel = st.selectbox("Select a company for details", df_wl_display["Company"].tolist(), key="wl_details_select")
            sym = nifty_dict.get(sel)
            if sym:
                with st.spinner("Loading company data..."):
                    df_sel = get_data(sym)
                price = df_sel['Close'].iat[-1]
                rsi = df_sel['RSI'].iat[-1]
                sma20 = df_sel['SMA_20'].iat[-1]
                avgvol = int(df_sel['Volume'].rolling(window=20).mean().iat[-1])
                sig = st.session_state.get("last_scan_signals", {}).get(sel, df_wl[df_wl["Company"] == sel]["Signal"].values[0])
                meta = fetch_company_meta(sym)
                industry = meta.get('Industry', 'N/A')
                sector = meta.get('Sector', 'N/A')

                tabs = st.tabs(["Overview", "Chart", "Financials", "Key Ratios", "Analytics"])
                with tabs[0]:
                    st.markdown(
                        f"<h4 style='margin-bottom:0'>{sel} <span style='font-size:16px;color:#888;'>({sym})</span></h4>",
                        unsafe_allow_html=True
                    )
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Price", f"{price:.2f}")
                        st.metric("RSI", f"{rsi:.2f}")
                    with c2:
                        st.metric("SMA 20", f"{sma20:.2f}")
                        st.metric("Avg Vol", f"{avgvol}")
                    with c3:
                        st.markdown(f"**Signal:** {pretty_signal(sig)}", unsafe_allow_html=True)
                    with c4:
                        st.write(f"**Industry:** {industry}")
                        st.write(f"**Sector:** {sector}")
                    st.caption("Navigate tabs for chart, financials, and ratios.")
                    st.divider()
                    remove_col, _ = st.columns([1, 5])
                    with remove_col:
                        if st.button("❌ Remove from Watchlist", key=f"remove_{sel}"):
                            new_entries = [e for e in entries if e.get("Company") != sel]
                            save_watchlist(new_entries)
                            st.success(f"{sel} removed from watchlist.")
                            st.rerun()
                with tabs[1]:
                    st.subheader("📈 Chart")
                    st.plotly_chart(generate_chart(df_sel), use_container_width=True)
                with tabs[2]:
                    st.subheader("📊 Financials")
                    try:
                        fin, bs, cf, rts = get_financial_data(sym)
                        st.markdown("**Income Statement (Annual)**")
                        st.dataframe(format_df_crores(fin), use_container_width=True)
                        st.markdown("**Balance Sheet (Annual)**")
                        st.dataframe(format_df_crores(bs), use_container_width=True)
                        st.markdown("**Cash Flow (Annual)**")
                        st.dataframe(format_df_crores(cf), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error fetching financial data: {e}")
                with tabs[3]:
                    st.subheader("📌 Key Ratios")
                    try:
                        _, _, _, rts = get_financial_data(sym)
                        ratio_names = list(rts.keys())
                        ratio_selected = st.selectbox("Select Ratio", ratio_names, key=f"ratio_{sel}")
                        st.metric(ratio_selected, rts[ratio_selected])
                        st.dataframe(pd.DataFrame.from_dict(rts, orient='index', columns=['Value']), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error fetching ratios: {e}")
                with tabs[4]:
                    st.subheader("📉 Analytics")
                    import plotly.graph_objs as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_sel.index, y=df_sel['Close'], mode='lines', name='Close'))
                    buy_signals = df_sel[df_sel['RSI'] < 30]
                    sell_signals = df_sel[df_sel['RSI'] > 70]
                    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='RSI Buy', marker=dict(color='green', size=10, symbol='triangle-up')))
                    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='RSI Sell', marker=dict(color='red', size=10, symbol='triangle-down')))
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("**RSI Strategy Backtest (Buy <30, Sell >70)**")
                    rsi_buys = (df_sel['RSI'] < 30).sum()
                    rsi_sells = (df_sel['RSI'] > 70).sum()
                    st.metric("Buy Signals", rsi_buys)
                    st.metric("Sell Signals", rsi_sells)
        else:
            st.info("Your watchlist is empty.")

else:
    st.warning("Invalid view selected.")
