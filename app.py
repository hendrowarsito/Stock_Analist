import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Stock Analyzer – Multi-Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background: #f5f7fa; }
    section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }

    /* ── Agent cards ── */
    .agent-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        color: #1e293b;
    }
    .agent-header {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid #f1f5f9;
    }

    /* ── Agent accent colors (kept vivid for contrast on white) ── */
    .bull   { color: #059669; }
    .bear   { color: #dc2626; }
    .tech   { color: #0284c7; }
    .fund   { color: #ea580c; }
    .newsc  { color: #7c3aed; }
    .sent   { color: #0891b2; }

    /* ── Decision boxes ── */
    .decision-buy  {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: #14532d;
        box-shadow: 0 2px 8px rgba(22,163,74,0.12);
    }
    .decision-sell {
        background: #fff1f2;
        border: 2px solid #dc2626;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: #7f1d1d;
        box-shadow: 0 2px 8px rgba(220,38,38,0.12);
    }
    .decision-hold {
        background: #fffbeb;
        border: 2px solid #d97706;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: #78350f;
        box-shadow: 0 2px 8px rgba(217,119,6,0.12);
    }

    /* ── Misc ── */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] { background: #f1f5f9; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"]      { border-radius: 6px; color: #475569; font-weight: 600; }
    .stTabs [aria-selected="true"]    { background: #ffffff; color: #1e293b; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stButton > button                { border-radius: 8px; font-weight: 600; }
    div[data-testid="stMetric"]       { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)


# ── Pure-pandas Technical Indicators (no external TA lib) ─────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # RSI (Wilder smoothing via EWM)
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    df["RSI"] = 100 - (100 / (1 + gain.ewm(com=13, adjust=False).mean()
                                   / loss.ewm(com=13, adjust=False).mean().replace(0, np.nan)))

    # MACD
    ema12        = close.ewm(span=12, adjust=False).mean()
    ema26        = close.ewm(span=26, adjust=False).mean()
    df["MACD"]   = ema12 - ema26
    df["MACD_S"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_H"] = df["MACD"] - df["MACD_S"]

    # Bollinger Bands (20, 2σ)
    df["SMA20"]  = close.rolling(20).mean()
    std20        = close.rolling(20).std()
    df["BB_UP"]  = df["SMA20"] + 2 * std20
    df["BB_LO"]  = df["SMA20"] - 2 * std20

    # SMAs
    df["SMA50"]  = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # ATR (Wilder)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(com=13, adjust=False).mean()

    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password",
                            help="sk-ant-... dari console.anthropic.com")

    st.markdown("---")
    page = st.radio("Navigation",
                    options=["Portfolio", "WMA Scanner"],
                    horizontal=False)


    st.markdown("---")
    st.markdown("### 💼 My Positions")

    HOLDINGS = {
        "AMD":  3.7012,
        "TSLA": 3.6887,
        "DUOL": 13.5623,
        "NVDA": 4.4887,
        "HIMS": 39.7669,
        "PLTR": 6.5000,
        "IBIT": 17.3876,
        "VOO":  0.9261,
        "MSFT": 0.4883,
    }

    @st.cache_data(ttl=300)
    def fetch_portfolio_prices(tickers):
        prices = {}
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                h = tk.history(period="2d", interval="1d")
                if not h.empty:
                    prices[t] = float(h["Close"].iloc[-1])
            except Exception:
                prices[t] = None
        return prices

    prices_live = fetch_portfolio_prices(tuple(HOLDINGS.keys()))

    total_value = sum(
        prices_live.get(t, 0) * s
        for t, s in HOLDINGS.items()
        if prices_live.get(t)
    )

    st.markdown(f"""
    <div style="background:#f0fdf4;border:1.5px solid #16a34a;border-radius:10px;
                padding:12px 14px;margin-bottom:12px;text-align:center;">
        <div style="font-size:10px;font-weight:800;color:#15803d;letter-spacing:1.2px;
                    text-transform:uppercase;margin-bottom:4px;">Total Portfolio Value</div>
        <div style="font-size:24px;font-weight:900;color:#14532d;">${total_value:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Build sorted positions list (largest value first)
    positions = []
    for t, shares in HOLDINGS.items():
        px  = prices_live.get(t)
        val = px * shares if px else 0.0
        positions.append((t, shares, px, val))
    positions.sort(key=lambda x: x[3], reverse=True)

    selected_ticker = None
    for t, shares, px, val in positions:
        pct     = (val / total_value * 100) if total_value > 0 and val > 0 else 0
        px_str  = f"${px:,.2f}" if px  else "–"
        val_str = f"${val:,.2f}" if val else "–"
        pct_str = f"{pct:.1f}%"

        # Color bar width for visual allocation
        bar_w = max(2, int(pct * 2.2))  # scale to max ~100px for ~45%

        c_btn, c_info = st.columns([1, 2])
        with c_btn:
            if st.button(t, key=f"pos_{t}", use_container_width=True):
                selected_ticker = t
        with c_info:
            st.markdown(
                f"<div style='font-size:11px;line-height:1.6;color:#475569;padding-top:1px'>"
                f"<span style='font-weight:700;color:#1e293b'>{px_str}</span>"
                f"&nbsp;<span style='color:#059669;font-weight:700'>{val_str}</span><br>"
                f"<div style='display:flex;align-items:center;gap:6px;margin-top:2px'>"
                f"<div style='background:#e2e8f0;border-radius:3px;height:5px;width:80px;overflow:hidden'>"
                f"<div style='background:#16a34a;height:5px;width:{bar_w}px;border-radius:3px'></div>"
                f"</div>"
                f"<span style='font-size:10px;color:#64748b;font-weight:600'>{pct_str}</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("### 🔍 Other Ticker")
    manual_ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").upper().strip()

    st.markdown("---")
    depth = st.select_slider("🎯 Analysis Depth",
                             options=["Quick", "Standard", "Deep"],
                             value="Standard")
    DEPTH_TOKENS = {"Quick": 600, "Standard": 1000, "Deep": 1500}

    st.markdown("---")
    st.caption("**US Stock Analyzer v1**\nMulti-Agent powered by Claude.\n\nFlow: Fundamental · Technical · News · Sentiment → Bull/Bear → Risk Judge")


# ── Page routing ─────────────────────────────────────────────────────────────
if page == "WMA Scanner":
    st.markdown("# 🔍 WMA200 Entry Zone Scanner")
    st.markdown("*Scan saham US yang berada di entry zone: harga di antara Daily WMA200 dan Weekly WMA200*")

    # ── Stock universe definitions ────────────────────────────────────────────
    UNIVERSES = {
        "My Portfolio":   ["AMD","TSLA","DUOL","NVDA","HIMS","PLTR","IBIT","VOO","MSFT"],
        "Technology":     ["AAPL","MSFT","NVDA","META","GOOGL","AMZN","TSLA","AMD","AVGO","ORCL",
                           "INTC","QCOM","TXN","MU","AMAT","NOW","SNOW","PLTR","CRWD","PANW",
                           "DDOG","NET","ZS","FTNT","ADBE","CRM","INTU","WDAY","HUBS","DUOL",
                           "ASTS","RKLB","SOFI","HOOD","COIN","MSTR","SMCI","DELL","HPQ","IBM"],
        "S&P 500 Top 50": ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","LLY","JPM","V",
                           "UNH","XOM","TSLA","MA","PG","JNJ","HD","AVGO","MRK","CVX",
                           "ABBV","COST","KO","PEP","WMT","BAC","MCD","NFLX","ADBE","CRM",
                           "TMO","ACN","LIN","DIS","CSCO","ABT","NKE","DHR","TXN","NEE",
                           "PM","RTX","AMGN","ORCL","HON","QCOM","IBM","GE","CAT","NOW"],
        "S&P 500 Top 100": ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "TSLA", "META", "WMT", "BRK.B", "LLY", "MU", "JPM", "AMD", "XOM", "V", "ORCL", "INTC", "JNJ", "CSCO", "COST", "MA", "CAT", "CVX", "ABBV", "NFLX", "UNH", "LRCX", "BAC", "KO", "AMAT", "PG", "PLTR", "MS", "HD", "PM", "GE", "GEV", "GS", "TXN", "MRK", "KLAC", "LIN", "RTX", "WFC", "AXP", "QCOM", "C", "SNDK", "IBM", "ADI", "PEP", "TMUS", "PANW", "MCD", "NEE", "VZ", "ANET", "DIS", "STX", "AMGN", "BA", "APP", "BLK", "T", "WDC", "GLW", "TJX", "TMO", "GILD", "UNP", "SCHW", "DELL", "ETN", "APH", "UBER", "DE", "CRWD", "WELL", "ISRG", "COP", "ABT", "PFE", "BX", "VRT", "CRM", "HON", "PLD", "CB", "CVS", "LOW", "MO", "SBUX", "BKNG", "SPGI", "LMT", "SYK", "PGR", "COF"],
        "S&P 500 Top 200": ["NEM", "BMY", "PWR", "DHR", "VRTX", "PH", "INTU", "CME", "EQIX", "SO", "HWM", "ACN", "TT", "ADBE", "NOW", "MDT", "CEG", "SNPS", "CMI", "CDNS", "WMB", "DUK", "HCA", "MAR", "BK", "MCK", "FCX", "GD", "FTNT", "FDX", "CMCSA", "WM", "JCI", "ICE", "KKR", "ADP", "PNC", "ELV", "MNST", "CSX", "UPS", "SLB", "USB", "AMT", "ABNB", "CIEN", "BSX", "APO", "MRSH", "MDLZ", "NOC", "MMM", "MPWR", "LITE", "CI", "MCO", "COHR", "KMI", "EOG", "EMR", "VLO", "MPC", "DDOG", "SHW", "NXPI", "ORLY", "REGN", "HLT", "ITW", "NSC", "PSX", "CL", "FIX", "RCL", "ECL", "HOOD", "DASH", "CRH", "ROST", "AEP", "AON", "WBD", "GM", "CTAS", "DLR", "APD", "MSI", "SPG", "TDG", "RSG", "TRV", "BKR", "NKE", "URI", "GWW", "TEL", "KEYS", "AFL", "OXY", "SRE"],
        "Single Stock":   [],
    }

    col_u, col_s = st.columns([2, 1])
    with col_u:
        universe_choice = st.selectbox("📂 Stock Universe", list(UNIVERSES.keys()))
    with col_s:
        min_cagr = st.number_input("Min Revenue CAGR (%)", value=10.0, step=1.0,
                                   help="Filter saham dengan Revenue CAGR 12 quartal ≥ nilai ini")

    if universe_choice == "Single Stock":
        single_input = st.text_input("Masukkan ticker (pisah koma jika multiple, e.g. NVDA,AAPL,MSFT)").upper()
        scan_tickers = [t.strip() for t in single_input.split(",") if t.strip()] if single_input else []
    else:
        scan_tickers = UNIVERSES[universe_choice]

    st.caption(f"Universe: **{len(scan_tickers)} stocks** · Filter CAGR ≥ {min_cagr:.0f}%")

    scan_btn = st.button("🚀 Run WMA Scanner", type="primary", use_container_width=True)

    if scan_btn and scan_tickers:
        # ── WMA + CAGR fetch ──────────────────────────────────────────────────
        import math

        @st.cache_data(ttl=1800)
        def scan_wma_cagr(tickers_tuple):
            import pandas as pd
            results = []
            ticker_list = list(tickers_tuple)
            prog = st.progress(0)
            stat = st.empty()

            for i, t in enumerate(ticker_list):
                stat.markdown(f"⏳ Scanning **{t}** ({i+1}/{len(ticker_list)})...")
                prog.progress((i+1) / len(ticker_list))
                try:
                    tk     = yf.Ticker(t)
                    info   = tk.info or {}
                    daily  = tk.history(period="1y",  interval="1d")
                    weekly = tk.history(period="4y",  interval="1wk")

                    if daily.empty or len(daily) < 50:
                        continue

                    price = float(daily["Close"].iloc[-1])

                    def wma(series, n):
                        w = pd.Series(range(1, n+1))
                        return series.rolling(n).apply(
                            lambda x: (x * w).sum() / w.sum(), raw=True
                        )

                    # Daily WMA200
                    d_wma_s = wma(daily["Close"], 200)
                    d_wma   = float(d_wma_s.iloc[-1]) if len(d_wma_s.dropna()) > 0 else None

                    # Weekly WMA200
                    if len(weekly) >= 200:
                        w_wma_s = wma(weekly["Close"], 200)
                        w_wma   = float(w_wma_s.iloc[-1]) if len(w_wma_s.dropna()) > 0 else None
                    else:
                        w_wma = None

                    if not d_wma or not w_wma:
                        continue

                    d_pct = (price / d_wma - 1) * 100
                    w_pct = (price / w_wma - 1) * 100

                    # Zone check: price BELOW daily WMA200 AND ABOVE weekly WMA200
                    in_zone     = (price < d_wma) and (price > w_wma)
                    zone_depth  = ((d_wma - price) / (d_wma - w_wma) * 100) if (d_wma > w_wma) else None

                    # Revenue CAGR – 12 quarters
                    cagr = None
                    try:
                        qr = tk.quarterly_financials
                        if qr is not None and not qr.empty:
                            rev_row = None
                            for lbl in ["Total Revenue","Revenue","Net Revenue"]:
                                if lbl in qr.index:
                                    rev_row = qr.loc[lbl]
                                    break
                            if rev_row is not None:
                                rev = rev_row.dropna().sort_index()
                                if len(rev) >= 2:
                                    n_periods = min(12, len(rev) - 1)
                                    r_start   = float(rev.iloc[-(n_periods+1)])
                                    r_end     = float(rev.iloc[-1])
                                    if r_start > 0 and r_end > 0:
                                        # Annualize: n_periods quarters = n_periods/4 years
                                        years = n_periods / 4.0
                                        cagr  = (math.pow(r_end / r_start, 1 / years) - 1) * 100
                    except Exception:
                        pass

                    results.append({
                        "Ticker":       t,
                        "Price":        price,
                        "Daily WMA200": d_wma,
                        "Weekly WMA200":w_wma,
                        "vs Daily (%)": round(d_pct, 2),
                        "vs Weekly (%)":round(w_pct, 2),
                        "In Zone":      in_zone,
                        "Zone Depth (%)": round(zone_depth, 1) if zone_depth else None,
                        "Rev CAGR (%)": round(cagr, 1) if cagr else None,
                        "Sector":       info.get("sector","–"),
                        "Market Cap":   info.get("marketCap", 0),
                    })
                except Exception:
                    pass

            prog.empty()
            stat.empty()
            return results

        with st.spinner("🔍 Scanning..."):
            all_results = scan_wma_cagr(tuple(scan_tickers))

        if not all_results:
            st.warning("Tidak ada data yang berhasil di-fetch.")
        else:
            import pandas as pd

            df_all = pd.DataFrame(all_results)

            # ── Tabs: In Zone | All Results ───────────────────────────────────
            tab_zone, tab_all = st.tabs([
                f"🎯 In Entry Zone ({df_all['In Zone'].sum()})",
                f"📋 All Results ({len(df_all)})"
            ])

            def style_table(df_display):
                """Apply green/red coloring to pct columns."""
                def color_cell(val):
                    if isinstance(val, str) and val.endswith("%"):
                        try:
                            v = float(val.replace("%","").replace("+",""))
                            if v > 0:   return "color:#16a34a;font-weight:700"
                            elif v < 0: return "color:#dc2626;font-weight:700"
                        except:
                            pass
                    return ""

                def highlight_zone(val):
                    if val == "✅ In Zone": return "background:#f0fdf4;color:#16a34a;font-weight:700"
                    if val == "❌ Below":   return "background:#fff1f2;color:#dc2626"
                    if val == "⬆️ Above":   return "background:#fffbeb;color:#d97706"
                    return ""

                pct_cols = [c for c in df_display.columns if "%" in c]
                styled = df_display.style.map(color_cell, subset=pct_cols)
                if "Status" in df_display.columns:
                    styled = styled.map(highlight_zone, subset=["Status"])
                return styled

            def format_df(df_in, zone_only=False):
                df = df_in.copy()
                if zone_only:
                    df = df[df["In Zone"] == True].copy()
                    if df.empty:
                        return df

                # Sort: zone first, then by zone depth
                df = df.sort_values(
                    ["In Zone", "Zone Depth (%)"],
                    ascending=[False, False]
                )

                df["Status"] = df.apply(
                    lambda r: "✅ In Zone" if r["In Zone"]
                    else ("⬆️ Above" if r["vs Daily (%)"] > 0 else "❌ Below"),
                    axis=1
                )

                # Format display
                df["Price"]          = df["Price"].apply(lambda x: f"${x:,.2f}")
                df["Daily WMA200"]   = df["Daily WMA200"].apply(lambda x: f"${x:,.2f}")
                df["Weekly WMA200"]  = df["Weekly WMA200"].apply(lambda x: f"${x:,.2f}")
                df["vs Daily (%)"]   = df["vs Daily (%)"].apply(lambda x: f"{x:+.2f}%")
                df["vs Weekly (%)"]  = df["vs Weekly (%)"].apply(lambda x: f"{x:+.2f}%")
                df["Zone Depth (%)"] = df["Zone Depth (%)"].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else "–"
                )
                df["Rev CAGR (%)"]   = df["Rev CAGR (%)"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "–"
                )
                df["Market Cap"]     = df["Market Cap"].apply(
                    lambda x: f"${x/1e9:.1f}B" if x > 0 else "–"
                )

                cols_show = ["Ticker","Status","Price","vs Daily (%)","vs Weekly (%)",
                             "Zone Depth (%)","Rev CAGR (%)","Sector","Market Cap"]
                return df[[c for c in cols_show if c in df.columns]]

            with tab_zone:
                df_zone = format_df(df_all, zone_only=True)
                # Apply CAGR filter
                df_zone_f = df_zone.copy()
                if not df_zone_f.empty and "Rev CAGR (%)" in df_zone_f.columns:
                    def parse_cagr(v):
                        try: return float(str(v).replace("%","").replace("+",""))
                        except: return -999
                    mask = df_zone_f["Rev CAGR (%)"].apply(parse_cagr) >= min_cagr
                    df_zone_filtered = df_zone_f[mask]
                else:
                    df_zone_filtered = df_zone_f

                if df_zone_filtered.empty:
                    st.info(f"Tidak ada saham yang masuk entry zone dengan CAGR ≥ {min_cagr:.0f}%.")
                else:
                    st.success(f"**{len(df_zone_filtered)} saham** masuk Entry Zone dengan Revenue CAGR ≥ {min_cagr:.0f}%")
                    st.dataframe(
                        style_table(df_zone_filtered),
                        use_container_width=True, hide_index=True
                    )
                    # Legend
                    st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:12px;color:#475569">
<b>📖 Cara baca:</b> &nbsp;
<b>vs Daily (%)</b> = jarak harga dari Daily WMA200 (negatif = di bawah) &nbsp;·&nbsp;
<b>vs Weekly (%)</b> = jarak harga dari Weekly WMA200 (positif = di atas) &nbsp;·&nbsp;
<b>Zone Depth</b> = posisi dalam zone (0% = baru masuk, 100% = di Weekly WMA200) &nbsp;·&nbsp;
<b>Rev CAGR</b> = CAGR revenue 12 kuartal terakhir (annualized)
</div>
""", unsafe_allow_html=True)

            with tab_all:
                df_formatted = format_df(df_all, zone_only=False)
                st.dataframe(
                    style_table(df_formatted),
                    use_container_width=True, hide_index=True
                )

    elif scan_btn and not scan_tickers:
        st.warning("Masukkan ticker terlebih dahulu.")

    st.stop()

# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("# 📊 US Stock Multi-Agent Analyzer")
st.markdown("*Powered by Claude · 7-Agent Debate Framework*")

ticker = manual_ticker if manual_ticker else (selected_ticker or "")

if not ticker:
    # ── Portfolio Performance Dashboard ──────────────────────────────────────
    st.markdown("## 📈 Portfolio Performance")

    @st.cache_data(ttl=600)
    def fetch_price_history(tickers, period):
        """Return rebased price series (start=100) for each ticker."""
        from datetime import date, timedelta
        import pandas as pd

        period_map = {"1Y": "1y", "YTD": "ytd", "3M": "3mo"}
        yf_period  = period_map[period]

        all_series = {}
        for t in tickers:
            try:
                tk   = yf.Ticker(t)
                hist = tk.history(period=yf_period, interval="1d")
                if hist.empty or len(hist) < 3:
                    continue
                hist.index = pd.DatetimeIndex(
                    hist.index.tz_localize(None) if hist.index.tzinfo else hist.index
                )
                close = hist["Close"].dropna()
                # Rebase to 100 at period start
                rebased = (close / close.iloc[0]) * 100
                all_series[t] = rebased
            except Exception:
                pass
        return all_series

    # Period selector
    period_choice = st.radio(
        "Pilih Periode",
        options=["1Y", "YTD", "3M"],
        format_func=lambda x: {"1Y": "📅 1 Year", "YTD": "🗓️ Year to Date", "3M": "📆 3 Months"}[x],
        horizontal=True,
    )
    period_labels = {"1Y": "1 Year", "YTD": "Year to Date", "3M": "3 Months"}

    with st.spinner("📡 Fetching price history..."):
        series_data = fetch_price_history(tuple(HOLDINGS.keys()), period_choice)

    if series_data:
        import plotly.graph_objects as go

        # Color palette – distinct colors for each ticker
        COLORS = {
            "AMD":  "#ef4444",
            "TSLA": "#f97316",
            "DUOL": "#eab308",
            "NVDA": "#22c55e",
            "HIMS": "#14b8a6",
            "PLTR": "#3b82f6",
            "IBIT": "#8b5cf6",
            "VOO":  "#ec4899",
            "MSFT": "#06b6d4",
        }
        DEFAULT_COLORS = [
            "#ef4444","#f97316","#eab308","#22c55e","#14b8a6",
            "#3b82f6","#8b5cf6","#ec4899","#06b6d4","#64748b",
        ]

        # Sort legend by final return (best on top)
        final_returns = {t: s.iloc[-1] - 100 for t, s in series_data.items()}
        sorted_tickers = sorted(series_data.keys(),
                                key=lambda t: final_returns[t], reverse=True)

        fig = go.Figure()

        # Baseline at 100
        if series_data:
            all_dates = sorted(set(
                d for s in series_data.values() for d in s.index
            ))
            fig.add_trace(go.Scatter(
                x=[all_dates[0], all_dates[-1]],
                y=[100, 100],
                mode="lines",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
                name="Baseline",
                hoverinfo="skip",
                showlegend=False,
            ))

        for i, t in enumerate(sorted_tickers):
            s     = series_data[t]
            color = COLORS.get(t, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
            ret   = final_returns[t]
            label = f"{t} ({ret:+.2f}%)"

            fig.add_trace(go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2.2),
                hovertemplate=f"<b>{t}</b>  |  %{{x|%b %d, %Y}}  |  %{{y:.2f}}  |  %{{customdata:+.2f}}%<extra></extra>",
                customdata=s.values - 100,
            ))

        fig.update_layout(
            title=dict(
                text=f"Portfolio Performance — {period_labels[period_choice]} (Rebased to 100)",
                x=0.5,
                font=dict(size=16, color="#1e293b"),
            ),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            height=500,
            xaxis=dict(
                title="Date",
                gridcolor="#e2e8f0",
                tickformat="%b %Y",
                tickfont=dict(size=11, color="#475569"),
                color="#475569",
                showspikes=True,
                spikecolor="#94a3b8",
                spikethickness=1,
                spikedash="dot",
            ),
            yaxis=dict(
                title="Indexed Value (Start = 100)",
                gridcolor="#e2e8f0",
                tickfont=dict(size=11, color="#475569"),
                color="#475569",
                zeroline=False,
            ),
            legend=dict(
                orientation="v",
                x=1.01, y=1,
                xanchor="left",
                font=dict(size=12),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e2e8f0",
                borderwidth=1,
            ),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", font_size=12),
            margin=dict(l=10, r=140, t=55, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── WMA200 Cards ──────────────────────────────────────────────────────
        st.markdown("### 📐 Price vs WMA 200")

        @st.cache_data(ttl=600)
        def fetch_wma200(tickers):
            import pandas as pd
            result = {}
            for t in tickers:
                try:
                    tk     = yf.Ticker(t)
                    daily  = tk.history(period="1y",  interval="1d")
                    weekly = tk.history(period="4y",  interval="1wk")

                    def wma(series, n):
                        weights = pd.Series(range(1, n + 1))
                        return series.rolling(n).apply(
                            lambda x: (x * weights).sum() / weights.sum(), raw=True
                        )

                    if daily.empty or weekly.empty:
                        continue

                    price  = float(daily["Close"].iloc[-1])
                    d_wma  = wma(daily["Close"],  200)
                    d_val  = float(d_wma.iloc[-1])  if pd.notna(d_wma.iloc[-1])  else None
                    d_pct  = (price / d_val  - 1) * 100 if d_val  else None
                    w_wma  = wma(weekly["Close"], 200)
                    w_val  = float(w_wma.iloc[-1]) if pd.notna(w_wma.iloc[-1]) else None
                    w_pct  = (price / w_val  - 1) * 100 if w_val  else None

                    result[t] = {"price": price, "d_pct": d_pct, "w_pct": w_pct}
                except Exception:
                    pass
            return result

        with st.spinner("📐 Calculating WMA200..."):
            wma_data = fetch_wma200(tuple(HOLDINGS.keys()))

        def pct_badge(val):
            if val is None:
                return "#64748b", "N/A"
            return ("#16a34a" if val >= 0 else "#dc2626"), f"{val:+.2f}%"

        tickers_wma = list(HOLDINGS.keys())

        # Build HTML table – label column + one column per ticker
        rows = {"Price": {}, "Daily WMA200": {}, "Weekly WMA200": {}}
        for t in tickers_wma:
            d = wma_data.get(t, {})
            price          = d.get("price")
            d_color, d_str = pct_badge(d.get("d_pct"))
            w_color, w_str = pct_badge(d.get("w_pct"))
            rows["Price"][t]        = (f"${price:,.2f}" if price else "–", "#1e293b",  True)
            rows["Daily WMA200"][t] = (d_str, d_color, False)
            rows["Weekly WMA200"][t]= (w_str, w_color, False)

        # Header row
        header_cells = '<th style="background:#f1f5f9;padding:10px 14px;text-align:left;'
        header_cells += 'font-size:12px;font-weight:700;color:#475569;border:1px solid #e2e8f0;'
        header_cells += 'min-width:80px;"></th>'
        for t in tickers_wma:
            # Highlight NVDA-style: bold border on last? just style each header
            header_cells += (
                f'<th style="background:#f1f5f9;padding:10px 14px;text-align:center;'
                f'font-size:13px;font-weight:800;color:#1e293b;'
                f'border:1px solid #e2e8f0;min-width:90px;">{t}</th>'
            )

        # Data rows
        data_rows_html = ""
        for row_label, ticker_vals in rows.items():
            is_price = (row_label == "Price")
            row_bg   = "#ffffff" if is_price else "#fafafa"
            lbl_style = (
                f'font-size:12px;font-weight:{"700" if is_price else "600"};'
                f'color:#{"1e293b" if is_price else "475569"};'
                f'padding:10px 14px;border:1px solid #e2e8f0;'
                f'background:{row_bg};white-space:nowrap;'
            )
            data_rows_html += f'<tr><td style="{lbl_style}">{row_label}</td>'
            for t in tickers_wma:
                val, color, bold = ticker_vals[t]
                cell_style = (
                    f'text-align:center;padding:10px 14px;'
                    f'border:1px solid #e2e8f0;background:{row_bg};'
                    f'font-size:{"15px" if is_price else "13px"};'
                    f'font-weight:{"800" if bold else "700"};'
                    f'color:{color};'
                )
                data_rows_html += f'<td style="{cell_style}">{val}</td>'
            data_rows_html += '</tr>'

        table_html = f"""
<div style="overflow-x:auto;margin-top:4px;">
  <table style="border-collapse:collapse;width:100%;background:#ffffff;
                border-radius:12px;overflow:hidden;
                box-shadow:0 1px 4px rgba(0,0,0,0.07);">
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{data_rows_html}</tbody>
  </table>
</div>
"""
        st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Klik ticker di **My Positions** atau ketik di **Other Ticker** untuk analisa detail.")
    st.stop()

if not api_key:
    st.warning("⚠️ Masukkan Anthropic API key di sidebar.")
    st.stop()


# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_stock_data(ticker: str):
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}
        hist = tk.history(period="6mo", interval="1d")
        if hist.empty:
            return None, None, []
        hist = calc_indicators(hist)
        news = []
        try:
            news = tk.news[:8] if tk.news else []
        except Exception:
            pass
        return info, hist, news
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None, None, []


def safe(v, fmt=".2f"):
    try:
        return f"{v:{fmt}}" if v is not None and pd.notna(v) else "N/A"
    except Exception:
        return "N/A"


def build_context(ticker, info, hist, news) -> str:
    c    = hist["Close"].iloc[-1]
    prev = hist["Close"].iloc[-2]
    chg  = (c - prev) / prev * 100

    rsi    = hist["RSI"].iloc[-1]
    sma20  = hist["SMA20"].iloc[-1]
    sma50  = hist["SMA50"].iloc[-1]
    sma200 = hist["SMA200"].iloc[-1]
    atr    = hist["ATR"].iloc[-1]
    macd   = hist["MACD"].iloc[-1]
    bb_up  = hist["BB_UP"].iloc[-1]
    bb_lo  = hist["BB_LO"].iloc[-1]
    vol20  = hist["Volume"].tail(20).mean()
    vol    = hist["Volume"].iloc[-1]

    p200 = (c / sma200 - 1) * 100 if pd.notna(sma200) and sma200 > 0 else None

    rsi_flag = ("⚠️ OVERBOUGHT" if pd.notna(rsi) and rsi > 70
                else ("⚠️ OVERSOLD" if pd.notna(rsi) and rsi < 30 else ""))
    bb_flag  = ("Above upper band ⚠️" if pd.notna(bb_up) and c > bb_up
                else ("Below lower band 🟢" if pd.notna(bb_lo) and c < bb_lo
                      else "Within bands"))

    fund = {
        "Sector":         info.get("sector"),
        "Industry":       info.get("industry"),
        "Market Cap":     f"${info.get('marketCap',0)/1e9:.1f}B" if info.get("marketCap") else None,
        "Forward P/E":    info.get("forwardPE"),
        "Trailing P/E":   info.get("trailingPE"),
        "PEG Ratio":      info.get("pegRatio"),
        "EPS (fwd)":      info.get("forwardEps"),
        "Revenue Growth": f"{info.get('revenueGrowth',0)*100:.1f}%" if info.get("revenueGrowth") else None,
        "Profit Margin":  f"{info.get('profitMargins',0)*100:.1f}%" if info.get("profitMargins") else None,
        "Debt/Equity":    info.get("debtToEquity"),
        "Free Cash Flow": f"${info.get('freeCashflow',0)/1e9:.2f}B" if info.get("freeCashflow") else None,
        "52W High":       info.get("fiftyTwoWeekHigh"),
        "52W Low":        info.get("fiftyTwoWeekLow"),
        "Analyst Target": info.get("targetMeanPrice"),
        "Recommendation": info.get("recommendationKey"),
        "Beta":           info.get("beta"),
    }
    fund_str = "\n".join(f"  {k}: {v}" for k, v in fund.items() if v is not None)
    news_str = "\n".join(f"- {n.get('title','')}" for n in news[:5]) or "No recent news"

    return f"""
=== {ticker} DATA ({datetime.now().strftime('%Y-%m-%d')}) ===

PRICE:
  Current:       ${safe(c)}
  Daily Change:  {safe(chg)}%
  Volume:        {safe(vol,',.0f')} (20d avg: {safe(vol20,',.0f')})

TECHNICALS:
  RSI(14):       {safe(rsi)} {rsi_flag}
  ATR(14):       {safe(atr)}
  MACD:          {safe(macd)}
  SMA 20:        {safe(sma20)}
  SMA 50:        {safe(sma50)}
  SMA 200:       {safe(sma200)}
  Price vs 200d: {safe(p200)}%
  BB Upper:      {safe(bb_up)}
  BB Lower:      {safe(bb_lo)}
  BB Position:   {bb_flag}

FUNDAMENTALS:
{fund_str}

NEWS (recent):
{news_str}
"""


# ── Chart ─────────────────────────────────────────────────────────────────────
def render_chart(hist, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
        increasing_line_color="#00d084", decreasing_line_color="#ff4b6e"),
        row=1, col=1)

    for col, color, lbl in [("SMA20","#f0b429","SMA20"),
                              ("SMA50","#38bdf8","SMA50"),
                              ("SMA200","#ff4b6e","SMA200")]:
        if col in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[col],
                                      line=dict(color=color, width=1.3),
                                      name=lbl, opacity=0.9), row=1, col=1)

    if "BB_UP" in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_UP"],
                                  line=dict(color="#7c6ff7", width=1, dash="dot"),
                                  name="BB Upper", opacity=0.7), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_LO"],
                                  line=dict(color="#7c6ff7", width=1, dash="dot"),
                                  fill="tonexty", fillcolor="rgba(124,111,247,0.06)",
                                  name="BB Lower", opacity=0.7), row=1, col=1)

    clr = ["#00d084" if c >= o else "#ff4b6e"
           for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
                          marker_color=clr, name="Volume", opacity=0.65),
                  row=2, col=1)

    if "RSI" in hist.columns:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI"],
                                  line=dict(color="#fb923c", width=1.5),
                                  name="RSI"), row=3, col=1)
        for lvl, clr2 in [(70, "#ff4b6e"), (30, "#00d084"), (50, "#4a4f5c")]:
            fig.add_hline(y=lvl, line_color=clr2, line_dash="dot",
                          line_width=1, row=3, col=1)

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        height=560, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"{ticker} – 6 Month Chart", x=0.5,
                   font=dict(size=14, color="#1e293b")),
        font=dict(color="#475569"),
    )
    for r in [1, 2, 3]:
        fig.update_yaxes(gridcolor="#e2e8f0", row=r, col=1)
        fig.update_xaxes(gridcolor="#e2e8f0", row=r, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


# ── Agent call ────────────────────────────────────────────────────────────────
def call_agent(client, system: str, user: str, max_tokens: int) -> str:
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"[Agent error: {e}]"


# ── Run ───────────────────────────────────────────────────────────────────────
if st.button(f"🚀 Run Analysis: {ticker}", type="primary", use_container_width=True):
    client  = anthropic.Anthropic(api_key=api_key)
    max_tok = DEPTH_TOKENS[depth]

    with st.spinner("📡 Fetching market data..."):
        info, hist, news = get_stock_data(ticker)

    if hist is None or hist.empty:
        st.error(f"❌ Tidak bisa fetch data untuk **{ticker}**.")
        st.stop()

    company   = info.get("longName", ticker)
    price     = hist["Close"].iloc[-1]
    prev      = hist["Close"].iloc[-2]
    chg       = (price - prev) / prev * 100
    ctx       = build_context(ticker, info, hist, news)

    # Header
    st.markdown(f"## {company} ({ticker})")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price",       f"${price:.2f}", f"{chg:+.2f}%")
    m2.metric("Market Cap",  f"${info.get('marketCap',0)/1e9:.1f}B" if info.get("marketCap") else "N/A")
    m3.metric("Forward P/E", f"{info.get('forwardPE'):.1f}" if info.get("forwardPE") else "N/A")
    m4.metric("52W High",    f"${info.get('fiftyTwoWeekHigh'):.2f}" if info.get("fiftyTwoWeekHigh") else "N/A")
    rsi_now = hist["RSI"].iloc[-1]
    m5.metric("RSI(14)", f"{rsi_now:.1f}" if pd.notna(rsi_now) else "N/A",
              "Overbought" if pd.notna(rsi_now) and rsi_now > 70 else
              ("Oversold"  if pd.notna(rsi_now) and rsi_now < 30 else "Neutral"))

    st.plotly_chart(render_chart(hist, ticker), use_container_width=True)
    st.markdown("---")
    st.markdown("## 🤖 Multi-Agent Analysis")

    results  = {}
    progress = st.progress(0)
    status   = st.empty()

    # 1 ── Fundamental
    status.markdown("⏳ **[1/7]** 📈 Fundamental Analyst...")
    results["fund"] = call_agent(client,
        system="""You are a Senior Fundamental Analyst specializing in US equities.
Analyze the stock data. Cover:
- Valuation (P/E, PEG vs peers & growth rate)
- Revenue & earnings quality and trajectory
- Balance sheet: FCF generation, debt burden
- Competitive moat & sector positioning
- Intrinsic value estimate vs current price
Be data-driven. End with BULLISH / BEARISH / NEUTRAL verdict.""",
        user=f"Fundamental analysis for {ticker}:\n{ctx}", max_tokens=max_tok)
    progress.progress(1/7)

    # 2 ── Technical
    status.markdown("⏳ **[2/7]** 📉 Technical Analyst...")
    results["tech"] = call_agent(client,
        system="""You are a Senior Technical Analyst with 20 years of experience.
Analyze technical indicators. Cover:
- Trend structure (price vs SMAs, MA alignment/crossovers)
- Momentum (RSI level & divergence, MACD crossover)
- Volatility (ATR, Bollinger Band position/squeeze)
- Key support & resistance price levels
- Specific entry zone and stop-loss price
Give concrete price numbers. End with BULLISH / BEARISH / NEUTRAL.""",
        user=f"Technical analysis for {ticker}:\n{ctx}", max_tokens=max_tok)
    progress.progress(2/7)

    # 3 ── News
    status.markdown("⏳ **[3/7]** 📰 News & Macro Analyst...")
    results["news"] = call_agent(client,
        system="""You are a News and Macro Analyst covering US equities.
Assess news headlines and macro backdrop. Cover:
- Key near-term catalysts (positive & negative)
- Sector tailwinds or headwinds
- Regulatory, competitive, or geopolitical risks
- Upcoming events (earnings date, product launches, Fed)
- Overall news sentiment signal
End with BULLISH / BEARISH / NEUTRAL.""",
        user=f"News & macro analysis for {ticker}:\n{ctx}", max_tokens=max_tok)
    progress.progress(3/7)

    # 4 ── Sentiment
    status.markdown("⏳ **[4/7]** 💬 Sentiment Analyst...")
    results["sent"] = call_agent(client,
        system="""You are a Market Sentiment Analyst.
Assess sentiment using available proxies. Cover:
- Analyst consensus & price target spread vs current price
- Volume trend as institutional activity proxy
- Momentum sentiment (RSI extremes as crowd behavior signals)
- Short squeeze potential or crowded-long risk
- Contrarian angle: is consensus too bullish/bearish?
End with BULLISH / BEARISH / NEUTRAL.""",
        user=f"Sentiment analysis for {ticker}:\n{ctx}", max_tokens=max_tok)
    progress.progress(4/7)

    # Summary for debate
    summary = (f"FUNDAMENTAL: {results['fund'][:500]}\n"
               f"TECHNICAL:   {results['tech'][:500]}\n"
               f"NEWS:        {results['news'][:500]}\n"
               f"SENTIMENT:   {results['sent'][:500]}")

    # 5 ── Bull
    status.markdown("⏳ **[5/7]** 🐂 Bull Researcher...")
    results["bull"] = call_agent(client,
        system="""You are the Bull Researcher at a professional trading firm.
Build the STRONGEST possible bullish case. Be aggressive but factual.
Present 4-5 bullet points — your best arguments for buying NOW.
Focus on what bears are missing, underweighting, or wrong about.""",
        user=f"Bull case for {ticker} @ ${price:.2f}:\n{ctx}\n\nANALYST SUMMARY:\n{summary}",
        max_tokens=max_tok)
    progress.progress(5/7)

    # 6 ── Bear
    status.markdown("⏳ **[6/7]** 🐻 Bear Researcher...")
    results["bear"] = call_agent(client,
        system="""You are the Bear Researcher at a professional trading firm.
Build the STRONGEST possible bearish case. Be incisive but factual.
Present 4-5 bullet points — your best arguments for avoiding or selling.
Focus on what bulls are ignoring, rationalizing, or getting wrong.""",
        user=f"Bear case for {ticker} @ ${price:.2f}:\n{ctx}\n\nANALYST SUMMARY:\n{summary}",
        max_tokens=max_tok)
    progress.progress(6/7)

    # 7 ── Risk Judge
    status.markdown("⏳ **[7/7]** ⚖️ Risk Judge — Final Decision...")
    full = (f"MARKET DATA:\n{ctx}\n\n"
            f"ANALYST REPORTS:\n{summary}\n\n"
            f"BULL CASE:\n{results['bull']}\n\n"
            f"BEAR CASE:\n{results['bear']}")
    results["judge"] = call_agent(client,
        system="""You are the Chief Risk Officer delivering the final trade decision.
Review all reports and the bull/bear debate. Structure your output EXACTLY as:

1. DECISION: BUY / HOLD / SELL  |  Conviction: HIGH / MEDIUM / LOW
2. RATIONALE: Why this side wins (2-3 sentences)
3. ENTRY PLAN:
   - Immediate entry: X% of intended position
   - Limit order(s): $XX.XX level(s) for remainder
4. RISK MANAGEMENT:
   - Hard stop-loss: $XX.XX
   - Thesis invalidation: [specific scenario]
5. PROFIT TARGETS:
   - Primary target: $XX.XX
   - Trim 25-30% at: $XX.XX
6. #1 RISK TO MONITOR: [single most important forward risk]

Be specific with price levels. Be decisive. No hedging.""",
        user=f"Final decision for {ticker} @ ${price:.2f}:\n{full}",
        max_tokens=max_tok + 300)
    progress.progress(7/7)
    status.empty()
    progress.empty()

    # ── Render ────────────────────────────────────────────────────────────────
    j_up = results["judge"].upper()
    if "BUY" in j_up[:300]:
        dcls, dicon, dlbl = "decision-buy",  "🟢", "BUY"
    elif "SELL" in j_up[:300]:
        dcls, dicon, dlbl = "decision-sell", "🔴", "SELL"
    else:
        dcls, dicon, dlbl = "decision-hold", "🟡", "HOLD"

    st.markdown(f"""
<div class="{dcls}">
<h2>{dicon} FINAL DECISION — {dlbl}</h2>
<p style="white-space:pre-wrap;line-height:1.8">{results['judge']}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    cb, cs = st.columns(2)
    with cb:
        st.markdown('<div class="agent-card"><div class="agent-header bull">🐂 Bull Researcher</div>',
                    unsafe_allow_html=True)
        st.markdown(results["bull"])
        st.markdown("</div>", unsafe_allow_html=True)
    with cs:
        st.markdown('<div class="agent-card"><div class="agent-header bear">🐻 Bear Researcher</div>',
                    unsafe_allow_html=True)
        st.markdown(results["bear"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Specialist Analyst Reports")
    t1, t2, t3, t4 = st.tabs(["📈 Fundamental", "📉 Technical", "📰 News & Macro", "💬 Sentiment"])
    for tab, key, cls, lbl in [
        (t1, "fund", "fund",  "Fundamental Analyst"),
        (t2, "tech", "tech",  "Technical Analyst"),
        (t3, "news", "newsc", "News & Macro Analyst"),
        (t4, "sent", "sent",  "Sentiment Analyst"),
    ]:
        with tab:
            st.markdown(
                f'<div class="agent-card"><div class="agent-header {cls}">{lbl}</div>\n\n'
                f'{results[key]}</div>',
                unsafe_allow_html=True)

    with st.expander("🔬 Raw Market Data Used by Agents"):
        st.code(ctx, language="text")

    st.markdown("---")
    st.caption(
        f"⏱ {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        f"Model: claude-sonnet-4-5  ·  Depth: {depth}  ·  "
        f"⚠️ Educational purposes only. Not financial advice."
    )
