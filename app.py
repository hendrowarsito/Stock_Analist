import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import html as _html

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Stock Analyzer – Multi-Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_dark = st.session_state.get("dark_mode", False)

_BG       = "#0f172a" if _dark else "#f5f7fa"
_SIDEBAR  = "#1e293b" if _dark else "#ffffff"
_CARD     = "#1e293b" if _dark else "#ffffff"
_BORDER   = "#334155" if _dark else "#e2e8f0"
_TEXT     = "#f1f5f9" if _dark else "#1e293b"
_SUBTEXT  = "#94a3b8" if _dark else "#475569"
_TABLIST  = "#0f172a" if _dark else "#f1f5f9"
_TABSEL   = "#334155" if _dark else "#ffffff"
_METRIC   = "#1e293b" if _dark else "#ffffff"
_EXPANDER = "#1e293b" if _dark else "#ffffff"

_BUY_BG   = "#052e16" if _dark else "#f0fdf4"
_BUY_TX   = "#86efac" if _dark else "#14532d"
_SELL_BG  = "#450a0a" if _dark else "#fff1f2"
_SELL_TX  = "#fca5a5" if _dark else "#7f1d1d"
_HOLD_BG  = "#431407" if _dark else "#fffbeb"
_HOLD_TX  = "#fcd34d" if _dark else "#78350f"
_HDR_BDR  = "#334155" if _dark else "#f1f5f9"

st.markdown(f"""
<style>
    /* ── Global ── */
    .stApp {{ background: {_BG}; color: {_TEXT}; }}
    section[data-testid="stSidebar"] {{ background: {_SIDEBAR}; border-right: 1px solid {_BORDER}; }}
    section[data-testid="stSidebar"] * {{ color: {_TEXT}; }}
    .stApp p, .stApp li, .stApp label {{ color: {_TEXT}; }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: {_TEXT}; }}

    /* ── Agent cards ── */
    .agent-card {{
        background: {_CARD};
        border: 1px solid {_BORDER};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        color: {_TEXT};
    }}
    .agent-header {{
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 2px solid {_HDR_BDR};
    }}

    /* ── Agent accent colors ── */
    .bull   {{ color: #34d399; }}
    .bear   {{ color: #f87171; }}
    .tech   {{ color: #38bdf8; }}
    .fund   {{ color: #fb923c; }}
    .newsc  {{ color: #a78bfa; }}
    .sent   {{ color: #22d3ee; }}

    /* ── Decision boxes ── */
    .decision-buy  {{
        background: {_BUY_BG};
        border: 2px solid #16a34a;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: {_BUY_TX};
        box-shadow: 0 2px 8px rgba(22,163,74,0.15);
    }}
    .decision-sell {{
        background: {_SELL_BG};
        border: 2px solid #dc2626;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: {_SELL_TX};
        box-shadow: 0 2px 8px rgba(220,38,38,0.15);
    }}
    .decision-hold {{
        background: {_HOLD_BG};
        border: 2px solid #d97706;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 22px;
        color: {_HOLD_TX};
        box-shadow: 0 2px 8px rgba(217,119,6,0.15);
    }}

    /* ── Misc ── */
    div[data-testid="stExpander"] {{
        border: 1px solid {_BORDER};
        border-radius: 10px;
        background: {_EXPANDER};
    }}
    .stTabs [data-baseweb="tab-list"] {{ background: {_TABLIST}; border-radius: 8px; padding: 4px; }}
    .stTabs [data-baseweb="tab"]      {{ border-radius: 6px; color: {_SUBTEXT}; font-weight: 600; }}
    .stTabs [aria-selected="true"]    {{ background: {_TABSEL}; color: {_TEXT}; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }}
    .stButton > button                {{ border-radius: 8px; font-weight: 600; }}
    div[data-testid="stMetric"]       {{ background: {_METRIC}; border: 1px solid {_BORDER}; border-radius: 10px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    div[data-testid="stMetric"] *     {{ color: {_TEXT} !important; }}

    /* ── Dataframe / table ── */
    .stDataFrame {{ background: {_CARD}; }}
    iframe[title="st.dataframe"]      {{ background: {_CARD}; }}
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

    dark_mode = st.toggle("🌙 Dark Mode", value=False, key="dark_mode")

    st.markdown("---")
    st.markdown("### 💼 My Positions")

    HOLDINGS = {
        "AMD":  3.701284677,
        "DUOL": 13.562294282,
        "TSLA": 4.177292654,
        "HIMS": 39.76698875,
        "PLTR": 8.755586634,
        "NVDA": 4.488739476,
        "MSFT": 0.718677631,
        "IBRX": 25.732694913,
        "NOW":  0.939990522,
    }

    # Average cost per share (derived from Robinhood cost basis data)
    AVERAGE_COST = {
        "AMD":  190.78,
        "DUOL": 133.71,
        "TSLA": 402.41,
        "HIMS": 34.44,
        "PLTR": 148.25,
        "NVDA": 159.61,
        "MSFT": 416.19,
        "IBRX": 7.75,
        "NOW":  132.80,
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
        "My Portfolio":        ["AMD","DUOL","TSLA","HIMS","PLTR","NVDA","MSFT","IBRX","NOW"],
        "Technology":          ["AAPL","MSFT","NVDA","META","GOOGL","AMZN","TSLA","AMD","AVGO","ORCL",
                                "INTC","QCOM","TXN","MU","AMAT","NOW","SNOW","PLTR","CRWD","PANW",
                                "DDOG","NET","ZS","FTNT","ADBE","CRM","INTU","WDAY","HUBS","DUOL",
                                "ASTS","RKLB","SOFI","HOOD","COIN","MSTR","SMCI","DELL","HPQ","IBM"],
        "Healthcare":          ["UNH","JNJ","LLY","MRK","ABBV","ABT","TMO","DHR","BMY","AMGN",
                                "PFE","GILD","ISRG","SYK","BSX","MDT","VRTX","REGN","CI","HUM",
                                "CVS","MCK","HIMS","IQVIA","CRL","A","IQV","ZBH","BAX","DXCM",
                                "PODD","INSP","TMDX","RXRX","EXAS","NTRA","ACAD","SANA","BEAM",
                                "CRSP","NTLA","EDIT","BLUE","SGMO","FATE","KYMR","ARVN","BDTX"],
        "Consumer Cyclical":   ["AMZN","TSLA","HD","MCD","NKE","SBUX","LOW","TJX","BKNG","MAR",
                                "GM","F","ORLY","AZO","APTV","LVS","WYNN","MGM","RCL","CCL",
                                "ABNB","UBER","LYFT","DASH","ETSY","EBAY","RVLV","CHWY","W","RDFN",
                                "DKNG","PENN","FL","ANF","DUOL","CELH","LULU","SKX","HAS","MAT"],
        # ── S&P 500 by index weight ranking (source: slickcharts.com) ─────────
        "S&P 500 #1-100":      ["NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","AVGO","TSLA","META",
                                "BRK-B","MU","LLY","WMT","AMD","JPM","V","XOM","INTC","JNJ","ORCL",
                                "CSCO","COST","MA","CAT","LRCX","ABBV","NFLX","CVX","BAC","AMAT",
                                "KO","UNH","PG","GE","PLTR","MS","HD","MRK","GS","TXN","PM","GEV",
                                "KLAC","QCOM","IBM","RTX","LIN","SNDK","WFC","AXP","C","TMUS","VZ",
                                "ADI","PEP","PANW","DELL","MCD","STX","ANET","APP","WDC","AMGN",
                                "NEE","DIS","BA","T","TJX","APH","TMO","BLK","GILD","CRWD","UNP",
                                "GLW","ETN","WELL","SCHW","PFE","ABT","HON","ISRG","CRM","BX",
                                "UBER","DE","COP","PLD","BKNG","CB","VRT","LMT","SPGI","LOW",
                                "DHR","MO","BMY","COF","SYK","CVS"],
        "S&P 500 #101-200":    ["PGR","NEM","SBUX","VRTX","PWR","ACN","PH","EQIX","SO","NOW",
                                "CEG","HWM","CDNS","TT","MAR","CME","SNPS","MDT","FDX","DUK",
                                "ADBE","BNY","FTNT","GD","CMI","WMB","FCX","MCK","CMCSA","PNC",
                                "UPS","MNST","CSX","HCA","AMT","ADP","WM","USB","KKR","SLB",
                                "JCI","ELV","ICE","INTU","CIEN","NXPI","MMM","MDLZ","MPWR","ABNB",
                                "EMR","MCO","NOC","DDOG","MRSH","SHW","HLT","GM","ROST","CI",
                                "APO","COHR","ECL","BSX","RCL","ORLY","CL","MPC","NSC","ITW",
                                "EOG","KMI","VLO","CRH","AEP","PSX","LITE","TDG","DASH","HOOD",
                                "WBD","MSI","CTAS","AON","NKE","DLR","SPG","FIX","REGN","TRV",
                                "APD","F","RSG","BKR","URI","TEL","TFC","SRE","KEYS","PCAR"],
        "S&P 500 #201-300":    ["D","AFL","GWW","TGT","TER","O","LHX","TRGP","OXY","OKE",
                                "NUE","FANG","ALL","CARR","VST","PSA","MET","CTVA","DAL","AJG",
                                "AME","MCHP","COR","DVN","FAST","NDAQ","ETR","ROK","CVNA","XEL",
                                "EA","ADSK","EBAY","AZO","EW","HPE","ON","EXC","CAH","COIN",
                                "GRMN","ODFL","FITB","WAB","MSCI","STT","IDXX","VTR","XYZ","YUM",
                                "CMG","DHI","TTWO","KDP","BDX","HSY","AIG","ED","KR","PEG",
                                "AMP","CCI","JBL","LYV","EME","ADM","PYPL","CCL","IRM","CBRE",
                                "STLD","WEC","HUM","UAL","CBOE","PCG","HIG","SYY","IBKR","SATS",
                                "VMC","PRU","EQT","MLM","KVUE","PAYX","LVS","WAT","ZTS","Q",
                                "HAL","KMB","EL","A","HBAN","ROP","ACGL","CPRT","MTB","AXON"],
        "S&P 500 #301-400":    ["NTRS","VICI","AEE","WDAY","EXR","DTE","RMD","FISV","FICO","CASY",
                                "ATO","BIIB","FSLR","NRG","GEHC","CNC","KHC","ARES","DOV","TDY",
                                "TPR","RJF","CNP","NTAP","TPL","IR","EIX","OTIS","IQV","EXPE",
                                "DXCM","FE","VRSN","PPL","CFG","AVB","ES","XYL","VEEV","HUBB",
                                "STZ","JBHT","CINF","EQR","PPG","CTSH","DOW","WRB","AWK","ROL",
                                "SYF","RF","WSM","WTW","DRI","BG","TSN","KEY","CHD","CPAY",
                                "DG","CMS","HPQ","NI","SMCI","PHM","RL","PFG","EXE","TROW",
                                "VRSK","ULTA","LEN","L","MTD","FFIV","LYB","WST","FIS","DGX",
                                "SBAC","OMC","LH","EXPD","LUV","ALB","STE","CHRW","AKAM","VLTO",
                                "SW","GPN","IFF","DD","INCY","EVRG","PKG","EFX","BRO","SNA"],
        "S&P 500 #401-503":    ["LNT","MRNA","VTRS","FTV","AMCR","CHTR","DLTR","CF","ESS","GIS",
                                "LII","WY","INVH","IP","BR","KIM","PTC","LDOS","BEN","NVR",
                                "GNRC","TXT","ZBH","HST","NDSN","DECK","TSCO","LULU","IEX","MAA",
                                "BALL","GEN","CDW","REG","TKO","MAS","J","DOC","EG","GPC",
                                "BBY","SOLV","CSGP","FOXA","APA","FOX","APTV","MKC","TRMB","DVA",
                                "AIZ","HII","PNW","TYL","HAS","UDR","SWK","AVY","IVZ","GL",
                                "SWKS","PSKY","BF-B","ZBRA","COO","CLX","ALGN","PNR","ERIE","HRL",
                                "GDDY","ALLE","SJM","CPT","MGM","RVTY","IT","TTD","FRT","AES",
                                "DPZ","WYNN","BAX","PODD","NWSA","JKHY","BXP","UHS","HSIC","ARE",
                                "FDS","NCLH","BLDR","AOS","TAP","CRL","MOS","TECH","POOL","CAG",
                                "CPB","NWS","EPAM"],
        # ── Thematic universes ────────────────────────────────────────────────
        "Semikonduktor":       ["TSM","MRVL","MU","ASML","LRCX","KLAC","ANET","DELL","CSCO","HPE"],
        "Energi AI":           ["VRT","CEG","VST","GEV","ETN","PWR","TLN","BWXT","NRG","OKLO","MOD"],
        "Optik/Jaringan":      ["LITE","COHR","FN","CIEN"],
        "Keamanan Siber":      ["PANW","S","OKTA","CRWV","NTSK"],
        "Kuantum":             ["IONQ","RGTI","QBTS","QUBT","ARQQ"],
        "Antariksa":           ["RKLB","IRDM","PL","BKSY","LUNR","RDW","KTOS","LHX","ASTS"],
        "Teknologi IDX":       ["GOTO.JK","BUKA.JK","EMTK.JK","DCII.JK","DMMX.JK",
                                "MTDL.JK","MLPT.JK","EDGE.JK","WIFI.JK","TLKM.JK",
                                "EXCL.JK","ISAT.JK","FREN.JK","ARTO.JK","BBYB.JK"],
        "Single Stock":        [],
    }

    # ── ① Universe selector ───────────────────────────────────────────────────
    universe_choice = st.selectbox("📂 ① Stock Universe", list(UNIVERSES.keys()))

    if universe_choice == "Single Stock":
        single_input = st.text_input(
            "Masukkan ticker (pisah koma jika multiple, e.g. NVDA,AAPL,MSFT)"
        ).upper()
        scan_tickers = [t.strip() for t in single_input.split(",") if t.strip()] if single_input else []
    else:
        scan_tickers = UNIVERSES[universe_choice]

    # ── Filter criteria ②③④ ───────────────────────────────────────────────────
    st.markdown("##### 🎛️ Filter Kriteria")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        min_cagr = st.number_input(
            "② Min Revenue CAGR (%)", value=10.0, step=1.0,
            help="Revenue CAGR 12 kuartal terakhir (annualized) ≥ nilai ini. Set 0 untuk skip."
        )
    with fc2:
        cash_debt_mode = st.selectbox(
            "③ Kas vs Utang",
            options=["Semua (no filter)", "Kas > Utang (≥ 1×)", "Kas ≥ 2× Utang"],
            help="Filter berdasarkan rasio kas terhadap total utang. Kas > Utang = net-cash position."
        )
    with fc3:
        min_rule40 = st.number_input(
            "④ Min Rule of 40", value=0.0, step=5.0,
            help="Rev Growth YoY (%) + FCF Margin (%) ≥ nilai ini. Benchmark SaaS = 40. Set 0 untuk skip."
        )

    active_filters = []
    if min_cagr > 0:                            active_filters.append(f"CAGR ≥ {min_cagr:.0f}%")
    if cash_debt_mode != "Semua (no filter)":   active_filters.append(cash_debt_mode)
    if min_rule40 > 0:                          active_filters.append(f"Rule of 40 ≥ {min_rule40:.0f}")

    st.caption(
        f"Universe: **{len(scan_tickers)} stocks** · "
        + (f"Filter aktif: **{' · '.join(active_filters)}**" if active_filters else "Tidak ada filter tambahan")
    )

    scan_btn = st.button("🚀 Run WMA Scanner", type="primary", use_container_width=True)

    if scan_btn and scan_tickers:
        import math

        prog  = st.progress(0)
        stat  = st.empty()
        all_results  = []
        skip_no_hist = 0   # < 50 daily bars
        skip_no_wma  = 0   # can't compute daily or weekly WMA200
        skip_error   = 0   # unhandled exception

        def wma(series, n):
            weights = np.arange(1, n + 1, dtype=float)
            return series.rolling(n).apply(
                lambda x: (x * weights).sum() / weights.sum(), raw=True
            )

        for i, t in enumerate(scan_tickers):
            stat.markdown(f"⏳ Scanning **{t}** ({i+1}/{len(scan_tickers)})…")
            prog.progress((i + 1) / len(scan_tickers))

            # ── Fetch price history (critical – skip if unavailable) ──────────
            try:
                tk     = yf.Ticker(t)
                daily  = tk.history(period="1y",  interval="1d")
                weekly = tk.history(period="5y",  interval="1wk")  # 5y ≈ 260 wks
            except Exception:
                skip_error += 1
                continue

            if daily.empty or len(daily) < 50:
                skip_no_hist += 1
                continue

            # ── Fetch fundamentals (non-critical – degrade gracefully) ────────
            try:
                info = tk.info or {}
            except Exception:
                info = {}

            # ── WMA200 ────────────────────────────────────────────────────────
            try:
                price   = float(daily["Close"].iloc[-1])
                d_valid = wma(daily["Close"], 200).dropna()
                d_wma   = float(d_valid.iloc[-1]) if len(d_valid) > 0 else None

                w_valid = wma(weekly["Close"], 200).dropna() if len(weekly) >= 200 else pd.Series([], dtype=float)
                w_wma   = float(w_valid.iloc[-1]) if len(w_valid) > 0 else None
            except Exception:
                skip_error += 1
                continue

            if d_wma is None or w_wma is None:
                skip_no_wma += 1
                continue

            d_pct      = (price / d_wma - 1) * 100
            w_pct      = (price / w_wma - 1) * 100
            in_zone    = (min(d_wma, w_wma) < price < max(d_wma, w_wma))
            zone_depth = (abs(max(d_wma, w_wma) - price) / abs(d_wma - w_wma) * 100) if (d_wma != w_wma) else None

            # ── ② Revenue CAGR + YoY Growth + TTM Revenue ───────────────────────
            # Both CAGR and YoY use annual income_stmt (up to 4 fiscal years).
            # Annual data is far more reliable than quarterly which yfinance often
            # returns with gaps, wrong quarter alignment, or only 4 quarters.
            # TTM revenue still uses quarterly data for a more current figure.
            cagr           = None
            rev_growth_pct = None
            total_revenue  = 0.0

            # --- CAGR + YoY from annual data ---
            try:
                ann = tk.income_stmt          # annual, up to 4 fiscal years
                if ann is None or ann.empty:
                    ann = tk.financials       # older yfinance alias
                if ann is not None and not ann.empty:
                    for lbl in ["Total Revenue", "Revenue", "Net Revenue"]:
                        if lbl in ann.index:
                            rev_ann = ann.loc[lbl].dropna().sort_index()  # oldest→newest
                            n_yr = len(rev_ann)
                            if n_yr >= 2:
                                r0 = float(rev_ann.iloc[0])
                                r1 = float(rev_ann.iloc[-1])
                                years = float(n_yr - 1)   # e.g. 3.0 for 4 data points
                                if r0 > 0 and r1 > 0:
                                    cagr = (math.pow(r1 / r0, 1.0 / years) - 1) * 100
                                # YoY = most recent FY vs prior FY (always 1 step)
                                r_prev = float(rev_ann.iloc[-2])
                                if r_prev > 0:
                                    rev_growth_pct = (r1 / r_prev - 1) * 100
                            break
            except Exception:
                pass

            # --- TTM Revenue + Quarterly YoY + save rev_q for R40 calc ---
            # Combine quarterly_income_stmt + quarterly_financials to maximise
            # the number of quarters available (each API may cover a diff window).
            rev_growth_q  = None   # quarterly YoY: TTM vs prior TTM (or Q vs Q-4)
            _rev_q_series = None   # kept for per-quarter R40 computation below
            try:
                _rev_parts = []
                for _api in [tk.quarterly_income_stmt, tk.quarterly_financials]:
                    try:
                        _df = _api
                        if _df is not None and not _df.empty:
                            for lbl in ["Total Revenue", "Revenue", "Net Revenue"]:
                                if lbl in _df.index:
                                    _rev_parts.append(_df.loc[lbl].dropna())
                                    break
                    except Exception:
                        pass
                if _rev_parts:
                    rev_q = pd.concat(_rev_parts)
                    rev_q = rev_q[~rev_q.index.duplicated(keep="last")].sort_index()
                    n_q   = len(rev_q)
                    total_revenue = float(rev_q.iloc[-4:].sum()) if n_q >= 4 else float(rev_q.sum())
                    _rev_q_series = rev_q   # save for per-quarter R40
                    # Quarterly YoY: TTM vs prior TTM
                    if n_q >= 8:
                        ttm_new = float(rev_q.iloc[-4:].sum())
                        ttm_old = float(rev_q.iloc[-8:-4].sum())
                        if ttm_old > 0:
                            rev_growth_q = (ttm_new / ttm_old - 1) * 100
                    elif n_q >= 5:
                        r_now = float(rev_q.iloc[-1])
                        r_yr  = float(rev_q.iloc[-5])
                        if r_yr > 0:
                            rev_growth_q = (r_now / r_yr - 1) * 100
            except Exception:
                pass

            # Deceleration flag: quarterly YoY < 50% of 3Y CAGR
            decel_flag = (
                rev_growth_q is not None and cagr is not None
                and cagr > 0 and rev_growth_q < cagr * 0.5
            )

            # ── ③ Cash vs Debt (balance sheet) ───────────────────────────────
            total_cash = 0.0
            total_debt = 0.0
            try:
                qbs = tk.quarterly_balance_sheet
                if qbs is not None and not qbs.empty:
                    for lbl in ["Cash And Cash Equivalents",
                                "Cash Cash Equivalents And Short Term Investments",
                                "Cash"]:
                        if lbl in qbs.index:
                            v = qbs.loc[lbl].dropna()
                            if len(v) > 0:
                                total_cash = float(v.iloc[0])
                            break
                    for lbl in ["Total Debt",
                                "Long Term Debt And Capital Lease Obligation",
                                "Long Term Debt"]:
                        if lbl in qbs.index:
                            v = qbs.loc[lbl].dropna()
                            if len(v) > 0:
                                total_debt = float(v.iloc[0])
                            break
            except Exception:
                pass

            if total_debt > 0:
                cash_debt_ratio = total_cash / total_debt
            elif total_cash > 0:
                cash_debt_ratio = 99.0   # no debt, has cash
            else:
                cash_debt_ratio = None

            # ── ④ FCF (cashflow) — TTM for FCF Margin + per-quarter for R40 ──
            fcf             = 0.0
            _fcf_q_series   = None   # per-quarter FCF series (oldest→newest)
            try:
                qcf = tk.quarterly_cashflow
                if qcf is not None and not qcf.empty:
                    if "Free Cash Flow" in qcf.index:
                        v_fcf = qcf.loc["Free Cash Flow"].dropna().sort_index()  # oldest→newest
                        _fcf_q_series = v_fcf
                        n_q = min(4, len(v_fcf))
                        if n_q > 0:
                            fcf = float(v_fcf.iloc[-n_q:].sum()) * (4 / n_q)
                    if fcf == 0:   # fallback: operating CF − capex
                        ocf_lbl = next(
                            (l for l in ["Operating Cash Flow",
                                         "Cash Flows From Operations",
                                         "Cash From Operations"]
                             if l in qcf.index), None)
                        cap_lbl = next(
                            (l for l in ["Capital Expenditure",
                                         "Purchase Of Property Plant And Equipment",
                                         "Purchases Of Property Plant And Equipment"]
                             if l in qcf.index), None)
                        if ocf_lbl:
                            v_ocf   = qcf.loc[ocf_lbl].dropna().sort_index()
                            n_q     = min(4, len(v_ocf))
                            ocf_ttm = float(v_ocf.iloc[-n_q:].sum()) * (4 / n_q)
                            capex_ttm = 0.0
                            if cap_lbl:
                                v_cap     = qcf.loc[cap_lbl].dropna().sort_index()
                                n_c       = min(4, len(v_cap))
                                capex_ttm = float(v_cap.iloc[-n_c:].sum()) * (4 / n_c)
                                # build per-quarter fallback series
                                if _fcf_q_series is None and len(v_ocf) > 0:
                                    if len(v_cap) > 0:
                                        _fcf_q_series = (v_ocf - v_cap.reindex(v_ocf.index, fill_value=0).abs())
                                    else:
                                        _fcf_q_series = v_ocf
                            fcf = ocf_ttm - abs(capex_ttm)
            except Exception:
                pass

            fcf_margin = (fcf / total_revenue * 100) if total_revenue > 0 else None

            # ── ④b Per-quarter Rule of 40 (last 3 quarters) ─────────────────
            # R40_i = Rev YoY (Q_i vs Q_{i-4}) + FCF Margin (Q_i / Rev_Q_i)
            # Primary: use quarterly series (needs ≥ i+5 revenue quarters).
            # Fallback: if quarterly lacks year-ago data, derive from annual CAGR
            # to approximate the year-ago revenue for that quarter.
            r40_q = [None, None, None]   # [latest, Q-1, Q-2]
            if _rev_q_series is not None and _fcf_q_series is not None:
                n_r = len(_rev_q_series)
                n_f = len(_fcf_q_series)
                # Annual revenue series for fallback (already fetched for CAGR)
                _ann_rev = None
                try:
                    _a = tk.income_stmt
                    if _a is None or _a.empty:
                        _a = tk.financials
                    if _a is not None and not _a.empty:
                        for lbl in ["Total Revenue", "Revenue", "Net Revenue"]:
                            if lbl in _a.index:
                                _ann_rev = _a.loc[lbl].dropna().sort_index()
                                break
                except Exception:
                    pass

                for i in range(3):
                    if n_f < i + 1:
                        continue   # not enough FCF quarters
                    try:
                        r_curr = float(_rev_q_series.iloc[-(i + 1)])
                        f_curr = float(_fcf_q_series.iloc[-(i + 1)])
                        if r_curr <= 0:
                            continue
                        fcm_i = f_curr / r_curr * 100

                        # --- Rev YoY: quarterly direct (preferred) ---
                        if n_r >= i + 5:
                            r_year_ago = float(_rev_q_series.iloc[-(i + 5)])
                            if r_year_ago > 0:
                                yoy_i = (r_curr / r_year_ago - 1) * 100
                                r40_q[i] = round(yoy_i + fcm_i, 1)
                        # --- Rev YoY fallback: use annual growth rate ---
                        elif _ann_rev is not None and len(_ann_rev) >= 2:
                            # Derive implied quarterly YoY from annual rev growth
                            # latest annual vs prior annual ≈ YoY for recent quarters
                            r_ann_new = float(_ann_rev.iloc[-1])
                            r_ann_old = float(_ann_rev.iloc[-2])
                            if r_ann_old > 0:
                                yoy_i = (r_ann_new / r_ann_old - 1) * 100
                                r40_q[i] = round(yoy_i + fcm_i, 1)
                    except Exception:
                        pass

            # ── Sector + Market Cap (info → fast_info fallback) ───────────────
            sector, market_cap = "–", 0
            try:
                sector     = info.get("sector") or "–"
                market_cap = info.get("marketCap") or 0
            except Exception:
                pass
            if not market_cap or sector == "–":
                try:
                    fi = tk.fast_info
                    if not market_cap:
                        market_cap = getattr(fi, "market_cap", 0) or 0
                except Exception:
                    pass

            all_results.append({
                "Ticker":             t,
                "Price":              price,
                "Daily WMA200":       d_wma,
                "Weekly WMA200":      w_wma,
                "vs Daily (%)":       round(d_pct, 2),
                "vs Weekly (%)":      round(w_pct, 2),
                "In Zone":            in_zone,
                "Zone Depth (%)":     round(zone_depth, 1) if zone_depth is not None else None,
                "Rev CAGR (%)":       round(cagr, 1) if cagr is not None else None,
                "Rev Growth Q (%)":   round(rev_growth_q, 1) if rev_growth_q is not None else None,
                "Decel":              decel_flag,
                "Cash ($B)":          round(total_cash / 1e9, 2) if total_cash else None,
                "Debt ($B)":          round(total_debt / 1e9, 2) if total_debt else None,
                "Cash/Debt":          round(cash_debt_ratio, 2) if cash_debt_ratio is not None else None,
                "Rev Growth YoY (%)": round(rev_growth_pct, 1) if rev_growth_pct is not None else None,
                "FCF Margin (%)":     round(fcf_margin, 1) if fcf_margin is not None else None,
                "R40 (Q)":            r40_q[0],
                "R40 (Q-1)":          r40_q[1],
                "R40 (Q-2)":          r40_q[2],
                "Sector":             sector,
                "Market Cap":         market_cap,
            })

        prog.empty()
        stat.empty()

        # ── Diagnostics ───────────────────────────────────────────────────────
        total_skipped = skip_no_hist + skip_no_wma + skip_error
        if total_skipped > 0:
            diag_parts = []
            if skip_no_hist: diag_parts.append(f"{skip_no_hist} data harga < 50 hari")
            if skip_no_wma:  diag_parts.append(f"{skip_no_wma} WMA200 tidak bisa dihitung (histori terlalu pendek)")
            if skip_error:   diag_parts.append(f"{skip_error} error API/network")
            st.caption(
                f"ℹ️ {total_skipped} ticker dilewati: {' · '.join(diag_parts)}"
            )

        if not all_results:
            st.warning(
                "Tidak ada data yang berhasil di-fetch. "
                "Kemungkinan penyebab: (1) saham baru IPO < 4 tahun sehingga data Weekly WMA200 "
                "tidak tersedia, (2) rate-limit yfinance, atau (3) gangguan network. "
                "Coba pilih universe lain atau jalankan ulang."
            )
        else:
            df_all = pd.DataFrame(all_results)

            tab_zone, tab_all, tab_chart = st.tabs([
                f"🎯 In Entry Zone ({int(df_all['In Zone'].sum())})",
                f"📋 All Results ({len(df_all)})",
                "📊 Kuadran R40 vs Cash/Debt",
            ])

            # ── Filter helper (operates on raw numeric df) ────────────────────
            def apply_filters(df_raw):
                df = df_raw.copy()
                if df.empty:
                    return df
                if min_cagr > 0 and "Rev CAGR (%)" in df.columns:
                    df = df[df["Rev CAGR (%)"].apply(
                        lambda x: x >= min_cagr if pd.notna(x) else False
                    )]
                if cash_debt_mode == "Kas > Utang (≥ 1×)" and "Cash/Debt" in df.columns:
                    df = df[df["Cash/Debt"].apply(
                        lambda x: x >= 1.0 if pd.notna(x) else False
                    )]
                elif cash_debt_mode == "Kas ≥ 2× Utang" and "Cash/Debt" in df.columns:
                    df = df[df["Cash/Debt"].apply(
                        lambda x: x >= 2.0 if pd.notna(x) else False
                    )]
                if min_rule40 > 0 and "R40 (Q)" in df.columns:
                    df = df[df["R40 (Q)"].apply(
                        lambda x: x >= min_rule40 if pd.notna(x) else False
                    )]
                return df

            # ── Format for display (raw → strings) ───────────────────────────
            def format_df(df_in):
                df = df_in.copy()
                if df.empty:
                    return df
                sort_cols = [c for c in ["In Zone", "Zone Depth (%)"] if c in df.columns]
                if sort_cols:
                    df = df.sort_values(
                        sort_cols,
                        ascending=[False] * len(sort_cols),
                        na_position="last",
                    )
                df["Status"] = df.apply(
                    lambda r: "✅ In Zone" if r.get("In Zone", False)
                    else ("⬆️ Above" if r.get("vs Daily (%)", 0) > 0 else "❌ Below"),
                    axis=1
                )
                df["Price"]              = df["Price"].apply(lambda x: f"${x:,.2f}")
                df["Daily WMA200"]       = df["Daily WMA200"].apply(lambda x: f"${x:,.2f}")
                df["Weekly WMA200"]      = df["Weekly WMA200"].apply(lambda x: f"${x:,.2f}")
                df["vs Daily (%)"]       = df["vs Daily (%)"].apply(lambda x: f"{x:+.2f}%")
                df["vs Weekly (%)"]      = df["vs Weekly (%)"].apply(lambda x: f"{x:+.2f}%")
                df["Zone Depth (%)"]     = df["Zone Depth (%)"].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else "–"
                )
                df["Rev CAGR (%)"]       = df["Rev CAGR (%)"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "–"
                )
                df["Rev Growth Q (%)"]   = df["Rev Growth Q (%)"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "–"
                )
                df["Decel"]              = df["Decel"].apply(
                    lambda x: "⚠️ Decel" if x else "–"
                )
                df["Cash/Debt"]          = df["Cash/Debt"].apply(
                    lambda x: ("No Debt" if x == 99.0 else f"{x:.2f}×") if pd.notna(x) else "–"
                )
                df["Rev Growth YoY (%)"] = df["Rev Growth YoY (%)"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "–"
                )
                df["FCF Margin (%)"]     = df["FCF Margin (%)"].apply(
                    lambda x: f"{x:+.1f}%" if pd.notna(x) else "–"
                )
                for col in ["R40 (Q)", "R40 (Q-1)", "R40 (Q-2)"]:
                    if col in df.columns:
                        df[col] = df[col].apply(
                            lambda x: f"{x:+.1f}" if pd.notna(x) else "–"
                        )
                df["Market Cap"]         = df["Market Cap"].apply(
                    lambda x: f"${x/1e9:.1f}B" if x > 0 else "–"
                )
                cols_show = [
                    "Ticker", "Status", "Price",
                    "vs Daily (%)", "vs Weekly (%)", "Zone Depth (%)",
                    "Rev CAGR (%)", "Rev Growth Q (%)", "Decel",
                    "Cash/Debt",
                    "R40 (Q)", "R40 (Q-1)", "R40 (Q-2)",
                    "Rev Growth YoY (%)", "FCF Margin (%)",
                    "Sector", "Market Cap",
                ]
                return df[[c for c in cols_show if c in df.columns]]

            # ── Styler ────────────────────────────────────────────────────────
            def style_table(df_display):
                def color_pct(val):
                    if isinstance(val, str) and val.endswith("%"):
                        try:
                            v = float(val.replace("%", "").replace("+", ""))
                            if v > 0:   return "color:#16a34a;font-weight:700"
                            elif v < 0: return "color:#dc2626;font-weight:700"
                        except: pass
                    return ""

                def highlight_zone(val):
                    if val == "✅ In Zone": return "background:#f0fdf4;color:#16a34a;font-weight:700"
                    if val == "❌ Below":   return "background:#fff1f2;color:#dc2626"
                    if val == "⬆️ Above":   return "background:#fffbeb;color:#d97706"
                    return ""

                def color_rule40(val):
                    try:
                        v = float(str(val).replace("+", ""))
                        if v >= 40:   return "color:#16a34a;font-weight:800"
                        elif v >= 20: return "color:#d97706;font-weight:700"
                        else:         return "color:#dc2626"
                    except: return ""

                def color_cash_debt(val):
                    if val == "No Debt": return "color:#16a34a;font-weight:800"
                    try:
                        v = float(str(val).replace("×", ""))
                        if v >= 2:   return "color:#16a34a;font-weight:800"
                        elif v >= 1: return "color:#d97706;font-weight:700"
                        else:        return "color:#dc2626"
                    except: return ""

                def color_decel(val):
                    if val == "⚠️ Decel": return "color:#dc2626;font-weight:700"
                    return "color:#94a3b8"

                pct_cols = [c for c in df_display.columns if "%" in c]
                styled = df_display.style.map(color_pct, subset=pct_cols)
                if "Status" in df_display.columns:
                    styled = styled.map(highlight_zone, subset=["Status"])
                r40_cols = [c for c in ["R40 (Q)", "R40 (Q-1)", "R40 (Q-2)"] if c in df_display.columns]
                if r40_cols:
                    styled = styled.map(color_rule40, subset=r40_cols)
                if "Cash/Debt" in df_display.columns:
                    styled = styled.map(color_cash_debt, subset=["Cash/Debt"])
                if "Decel" in df_display.columns:
                    styled = styled.map(color_decel, subset=["Decel"])
                return styled

            # ── Quadrant chart ────────────────────────────────────────────────
            def render_quadrant_chart(df_raw):
                """R40 (Q) vs Cash/Debt scatter with 4 coloured quadrants."""
                if df_raw.empty:
                    st.info("Tidak ada data untuk ditampilkan.")
                    return

                # Build plot DataFrame – drop rows missing either axis
                cols_needed = ["Ticker", "R40 (Q)", "Cash/Debt", "In Zone", "vs Daily (%)"]
                plot_df = df_raw[[c for c in cols_needed if c in df_raw.columns]].copy()
                plot_df = plot_df.dropna(subset=["R40 (Q)", "Cash/Debt"])
                if plot_df.empty:
                    st.info("Data R40 atau Cash/Debt tidak tersedia untuk universe ini.")
                    return

                # Cap Cash/Debt at 5 for display (99 = No Debt sentinel)
                DISPLAY_CAP = 5.0
                plot_df["cd_disp"] = plot_df["Cash/Debt"].apply(
                    lambda v: DISPLAY_CAP if v >= DISPLAY_CAP else float(v)
                )
                plot_df["cd_label"] = plot_df["Cash/Debt"].apply(
                    lambda v: "No Debt" if v == 99.0 else f"{v:.2f}×"
                )

                # Axis range padding
                x_vals = plot_df["R40 (Q)"].values
                y_vals = plot_df["cd_disp"].values
                x_min = min(float(min(x_vals)) - 10, -20)
                x_max = max(float(max(x_vals)) + 10,  80)
                y_min = -0.1
                y_max = DISPLAY_CAP + 0.3

                fig = go.Figure()

                # ── Quadrant background shapes ────────────────────────────────
                quads = [
                    # (x0, x1, y0, y1, rgba_fill)
                    (40,    x_max, 1,     y_max, "rgba(34,197,94,0.10)"),   # top-right  green
                    (40,    x_max, y_min, 1,     "rgba(249,115,22,0.10)"),  # bot-right  orange
                    (x_min, 40,   1,     y_max, "rgba(234,179,8,0.10)"),   # top-left   yellow
                    (x_min, 40,   y_min, 1,     "rgba(239,68,68,0.10)"),   # bot-left   red
                ]
                shapes = []
                for x0, x1, y0, y1, fill in quads:
                    shapes.append(dict(
                        type="rect", xref="x", yref="y",
                        x0=x0, x1=x1, y0=y0, y1=y1,
                        fillcolor=fill, line_width=0, layer="below"
                    ))
                # Reference lines
                shapes.append(dict(type="line", xref="x", yref="y",
                    x0=40, x1=40, y0=y_min, y1=y_max,
                    line=dict(color="#475569", width=1.5, dash="dash")))
                shapes.append(dict(type="line", xref="x", yref="y",
                    x0=x_min, x1=x_max, y0=1, y1=1,
                    line=dict(color="#475569", width=1.5, dash="dash")))
                fig.update_layout(shapes=shapes)

                # ── Scatter points ────────────────────────────────────────────
                def point_color(r40, cd):
                    if r40 >= 40 and cd >= 1:   return "#16a34a"   # green
                    if r40 >= 40 and cd < 1:    return "#ea580c"   # orange
                    if r40 < 40  and cd >= 1:   return "#ca8a04"   # yellow
                    return "#dc2626"                                 # red

                for _, row in plot_df.iterrows():
                    r40 = float(row["R40 (Q)"])
                    cd  = float(row["cd_disp"])
                    cd_raw = float(row["Cash/Debt"])
                    in_zone = bool(row.get("In Zone", False))
                    vs_daily = row.get("vs Daily (%)", 0)
                    try:
                        vs_daily = float(vs_daily) if pd.notna(vs_daily) else 0
                    except (ValueError, TypeError):
                        vs_daily = 0
                    is_below = (not in_zone) and (vs_daily <= 0)
                    color = point_color(r40, cd_raw)

                    if in_zone:
                        symbol, size, status_label = "star",    16, "✅ In Entry Zone"
                    elif is_below:
                        symbol, size, status_label = "diamond",  13, "❌ Below WMA200"
                    else:
                        symbol, size, status_label = "circle",   11, "⬆️ Above WMA200"

                    hover = (
                        f"<b>{row['Ticker']}</b><br>"
                        f"R40 (Q): {r40:+.1f}<br>"
                        f"Cash/Debt: {row['cd_label']}<br>"
                        f"{status_label}"
                    )
                    fig.add_trace(go.Scatter(
                        x=[r40], y=[cd],
                        mode="markers+text",
                        marker=dict(
                            color=color,
                            size=size,
                            symbol=symbol,
                            line=dict(color="white", width=1.5),
                            opacity=0.92,
                        ),
                        text=[row["Ticker"]],
                        textposition="top center",
                        textfont=dict(size=10, color=color),
                        hovertemplate=hover + "<extra></extra>",
                        showlegend=False,
                    ))


                fig.update_layout(
                    xaxis=dict(title="Rule of 40 (Q)", range=[x_min, x_max],
                               gridcolor="#e2e8f0", zeroline=False),
                    yaxis=dict(title="Cash / Debt",    range=[y_min, y_max],
                               gridcolor="#e2e8f0", zeroline=False,
                               tickvals=[0, 0.5, 1, 2, 3, 4, 5],
                               ticktext=["0", "0.5×", "1×", "2×", "3×", "4×", "≥5×"]),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    height=580,
                    margin=dict(l=60, r=40, t=40, b=60),
                    hoverlabel=dict(bgcolor="white", font_size=13),
                )

                st.plotly_chart(fig, use_container_width=True)

                # Legend
                st.markdown("""
<div style="display:flex;gap:20px;font-size:12px;color:#475569;margin-top:-8px;flex-wrap:wrap">
<span>⭐ = In Entry Zone &nbsp;|&nbsp; ◆ = Below WMA200 &nbsp;|&nbsp; ● = Above WMA200</span>
<span style="color:#16a34a;font-weight:700">🟢 Kuat & Sehat (R40≥40, C/D≥1)</span>
<span style="color:#ea580c;font-weight:700">🟠 Tumbuh, Debt Tinggi (R40≥40, C/D&lt;1)</span>
<span style="color:#ca8a04;font-weight:700">🟡 Kas Kuat, R40 Rendah (R40&lt;40, C/D≥1)</span>
<span style="color:#dc2626;font-weight:700">🔴 Lemah & Berisiko (R40&lt;40, C/D&lt;1)</span>
</div>
""", unsafe_allow_html=True)

            # ── Tab: Entry Zone ───────────────────────────────────────────────
            with tab_zone:
                df_zone_raw     = df_all[df_all["In Zone"] == True].copy()
                df_filtered_raw = apply_filters(df_zone_raw)
                df_zone_disp    = format_df(df_filtered_raw)

                filter_label = " · ".join(active_filters) if active_filters else "tanpa filter tambahan"
                if df_zone_disp.empty:
                    st.info(f"Tidak ada saham yang masuk entry zone dengan kriteria: **{filter_label}**")
                else:
                    st.success(f"**{len(df_zone_disp)} saham** lulus semua kriteria: {filter_label}")
                    st.dataframe(style_table(df_zone_disp), use_container_width=True, hide_index=True)
                    st.markdown("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px;margin-top:8px;font-size:12px;color:#475569;line-height:1.8">
<b>📖 Cara baca kolom:</b><br>
<b>vs Daily/Weekly (%)</b> = jarak harga dari WMA200 (– = di bawah, + = di atas) &nbsp;·&nbsp;
<b>Zone Depth</b> = posisi dalam zone (0% = tepat di bawah Daily WMA, 100% = tepat di atas Weekly WMA) &nbsp;·&nbsp;
<b>Rev CAGR (%)</b> = CAGR revenue tahunan (multi-year, dari annual income stmt) &nbsp;·&nbsp;
<b>Rev Growth Q (%)</b> = YoY pertumbuhan revenue quarterly terbaru (TTM vs TTM-4Q, atau Q vs Q-4) &nbsp;·&nbsp;
<b>⚠️ Decel</b> = Rev Growth Q &lt; 50% dari CAGR (pertumbuhan melambat signifikan) &nbsp;·&nbsp;
<b>Cash/Debt</b> = rasio kas terhadap total utang
<span style="color:#16a34a;font-weight:700">≥ 2× 🟢</span> &nbsp;
<span style="color:#d97706;font-weight:700">≥ 1× 🟡</span> &nbsp;
<span style="color:#dc2626">&lt; 1× 🔴</span> &nbsp;·&nbsp;
<b>R40 (Q) / (Q-1) / (Q-2)</b> = Rule of 40 per kuartal = Rev YoY Q% + FCF Margin Q% &nbsp;·&nbsp; tren 3 kuartal terbaru
<span style="color:#16a34a;font-weight:700">≥ 40 🟢</span> &nbsp;
<span style="color:#d97706;font-weight:700">≥ 20 🟡</span> &nbsp;
<span style="color:#dc2626">&lt; 20 🔴</span>
</div>
""", unsafe_allow_html=True)

            # ── Tab: All Results ──────────────────────────────────────────────
            with tab_all:
                st.dataframe(
                    style_table(format_df(df_all)),
                    use_container_width=True, hide_index=True
                )

            # ── Tab: Quadrant Chart ───────────────────────────────────────────
            with tab_chart:
                render_quadrant_chart(df_all)

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

        @st.cache_data(ttl=3600)
        def fetch_rev_growth_q(tickers):
            result = {}
            for t in tickers:
                rev_q = None
                try:
                    tk = yf.Ticker(t)
                    _parts = []
                    for _api in [tk.quarterly_income_stmt, tk.quarterly_financials]:
                        try:
                            _df = _api
                            if _df is not None and not _df.empty:
                                for lbl in ["Total Revenue", "Revenue", "Net Revenue"]:
                                    if lbl in _df.index:
                                        _parts.append(_df.loc[lbl].dropna())
                                        break
                        except Exception:
                            pass
                    if _parts:
                        rq = pd.concat(_parts)
                        rq = rq[~rq.index.duplicated(keep="last")].sort_index()
                        n = len(rq)
                        if n >= 8:
                            t_new = float(rq.iloc[-4:].sum())
                            t_old = float(rq.iloc[-8:-4].sum())
                            if t_old > 0:
                                rev_q = (t_new / t_old - 1) * 100
                        elif n >= 5:
                            r1 = float(rq.iloc[-1])
                            r0 = float(rq.iloc[-5])
                            if r0 > 0:
                                rev_q = (r1 / r0 - 1) * 100
                except Exception:
                    pass
                result[t] = rev_q
            return result

        with st.spinner("📐 Calculating WMA200 & Rev Growth..."):
            wma_data    = fetch_wma200(tuple(HOLDINGS.keys()))
            rev_q_data  = fetch_rev_growth_q(tuple(HOLDINGS.keys()))

        def pct_badge(val):
            if val is None:
                return "#64748b", "N/A"
            return ("#16a34a" if val >= 0 else "#dc2626"), f"{val:+.2f}%"

        tickers_wma  = list(HOLDINGS.keys())
        total_val_wma = sum(
            (wma_data.get(t, {}).get("price") or 0) * HOLDINGS.get(t, 0)
            for t in tickers_wma
        ) or total_value or 1

        # ── Row definitions: (label, section)
        # section: "portfolio" | "price" | "wma" | "growth"
        ROW_META = [
            ("Nilai Posisi ($)",    "portfolio"),
            ("% Portfolio",         "portfolio"),
            ("Harga Saat Ini",      "price"),
            ("Avg Cost/Share",      "price"),
            ("Pertumbuhan (%)",     "gain"),
            ("vs Daily WMA200",     "wma"),
            ("vs Weekly WMA200",    "wma"),
            ("Rev Growth Q (%)",    "growth"),
        ]
        SECTION_BG  = {"portfolio": "#f0f9ff", "price": "#ffffff",
                       "gain": "#f0fdf4", "wma": "#fafafa", "growth": "#fdf4ff"}
        SECTION_LBL = {"portfolio": "#0369a1", "price": "#1e293b",
                       "gain": "#15803d", "wma": "#475569", "growth": "#7c3aed"}

        rows = {lbl: {} for lbl, _ in ROW_META}
        for t in tickers_wma:
            d        = wma_data.get(t, {})
            price    = d.get("price")
            shares   = HOLDINGS.get(t, 0)
            nominal  = (price * shares) if price else None
            avg_cost = AVERAGE_COST.get(t)
            pct_port = (nominal / total_val_wma * 100) if nominal else None
            rev_q_val = rev_q_data.get(t)

            # Portfolio growth = (current_price - avg_cost) / avg_cost * 100
            growth = ((price - avg_cost) / avg_cost * 100) if (price and avg_cost) else None

            d_color, d_str = pct_badge(d.get("d_pct"))
            w_color, w_str = pct_badge(d.get("w_pct"))

            rows["Nilai Posisi ($)"][t]  = (f"${nominal:,.0f}" if nominal else "–",     "#0f172a", True)
            rows["% Portfolio"][t]       = (f"{pct_port:.1f}%" if pct_port else "–",    "#0369a1", False)
            rows["Harga Saat Ini"][t]    = (f"${price:,.2f}"   if price    else "–",    "#1e293b", True)
            rows["Avg Cost/Share"][t]    = (f"${avg_cost:,.2f}" if avg_cost else "–",   "#475569", False)

            if growth is not None:
                g_color = "#16a34a" if growth >= 0 else "#dc2626"
                rows["Pertumbuhan (%)"][t] = (f"{growth:+.1f}%", g_color, True)
            else:
                rows["Pertumbuhan (%)"][t] = ("–", "#94a3b8", False)

            rows["vs Daily WMA200"][t]   = (d_str, d_color, False)
            rows["vs Weekly WMA200"][t]  = (w_str, w_color, False)

            if rev_q_val is not None:
                rq_color = "#16a34a" if rev_q_val >= 0 else "#dc2626"
                rows["Rev Growth Q (%)"][t] = (f"{rev_q_val:+.1f}%", rq_color, False)
            else:
                rows["Rev Growth Q (%)"][t] = ("–", "#94a3b8", False)

        # ── Sticky-header + sticky-first-col table ───────────────────────────
        TH_BASE = (
            "padding:10px 14px;border:1px solid #e2e8f0;white-space:nowrap;"
            "position:sticky;top:0;z-index:2;"
        )
        # Corner cell (sticky top + left)
        corner = (
            f'<th style="{TH_BASE}left:0;z-index:3;background:#f1f5f9;'
            f'font-size:11px;font-weight:700;color:#94a3b8;min-width:130px;'
            f'text-align:left;"></th>'
        )
        ticker_ths = ""
        for t in tickers_wma:
            ticker_ths += (
                f'<th style="{TH_BASE}background:#f1f5f9;text-align:center;'
                f'font-size:13px;font-weight:800;color:#1e293b;min-width:90px;">{t}</th>'
            )

        # Data rows
        data_rows_html = ""
        for row_label, section in ROW_META:
            ticker_vals = rows[row_label]
            row_bg  = SECTION_BG[section]
            lbl_clr = SECTION_LBL[section]
            is_bold_lbl = section in ("portfolio", "gain")
            # Sticky first column label cell
            lbl_style = (
                f"font-size:11px;font-weight:{'700' if is_bold_lbl else '600'};"
                f"color:{lbl_clr};"
                f"padding:10px 14px;border:1px solid #e2e8f0;"
                f"background:{row_bg};white-space:nowrap;"
                f"position:sticky;left:0;z-index:1;"
            )
            data_rows_html += f'<tr><td style="{lbl_style}">{row_label}</td>'
            for t in tickers_wma:
                val, color, bold = ticker_vals[t]
                is_large = row_label in ("Nilai Posisi ($)", "Harga Saat Ini")
                cell_style = (
                    f"text-align:center;padding:10px 14px;"
                    f"border:1px solid #e2e8f0;background:{row_bg};"
                    f"font-size:{'15px' if is_large else '13px'};"
                    f"font-weight:{'800' if bold else '600'};"
                    f"color:{color};"
                )
                data_rows_html += f'<td style="{cell_style}">{val}</td>'
            data_rows_html += '</tr>'

        table_html = f"""
<div style="overflow-x:auto;overflow-y:visible;margin-top:4px;
            border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,0.07);">
  <table style="border-collapse:collapse;background:#ffffff;table-layout:auto;">
    <thead><tr>{corner}{ticker_ths}</tr></thead>
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


# ── Email helpers ─────────────────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """Very minimal markdown → HTML: bold, newlines. No external deps."""
    import re
    text = _html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',     r'<em>\1</em>',         text)
    text = text.replace('\n', '<br>')
    return text


def _build_email_html(ticker, company, price, chg, dcls, dlbl, results, depth):
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    dec_colors = {
        "decision-buy":  ("#f0fdf4", "#16a34a", "#14532d"),
        "decision-sell": ("#fff1f2", "#dc2626", "#7f1d1d"),
        "decision-hold": ("#fffbeb", "#d97706", "#78350f"),
    }
    bg, border, fg = dec_colors.get(dcls, ("#f8fafc", "#64748b", "#1e293b"))
    chg_color = "#16a34a" if chg >= 0 else "#dc2626"
    chg_sign  = "+" if chg >= 0 else ""

    sections_html = ""
    for key, label, color in [
        ("fund", "📈 Fundamental Analyst",  "#ea580c"),
        ("tech", "📉 Technical Analyst",    "#0284c7"),
        ("news", "📰 News & Macro Analyst", "#7c3aed"),
        ("sent", "💬 Sentiment Analyst",    "#0891b2"),
    ]:
        sections_html += f"""
        <div style="margin-bottom:20px;padding:16px 20px;background:#ffffff;
                    border:1px solid #e2e8f0;border-left:4px solid {color};border-radius:8px;">
          <div style="font-size:11px;font-weight:800;letter-spacing:1.5px;
                      text-transform:uppercase;color:{color};margin-bottom:10px;">{label}</div>
          <div style="font-size:14px;line-height:1.7;color:#334155;">
            {_md_to_html(results.get(key, ''))}
          </div>
        </div>"""

    bull_html = f"""
        <div style="padding:16px 20px;background:#f0fdf4;border:1px solid #bbf7d0;
                    border-radius:8px;margin-bottom:12px;">
          <div style="font-size:11px;font-weight:800;letter-spacing:1.5px;
                      text-transform:uppercase;color:#059669;margin-bottom:10px;">🐂 Bull Case</div>
          <div style="font-size:14px;line-height:1.7;color:#14532d;">
            {_md_to_html(results.get('bull', ''))}
          </div>
        </div>"""

    bear_html = f"""
        <div style="padding:16px 20px;background:#fff1f2;border:1px solid #fecaca;
                    border-radius:8px;margin-bottom:20px;">
          <div style="font-size:11px;font-weight:800;letter-spacing:1.5px;
                      text-transform:uppercase;color:#dc2626;margin-bottom:10px;">🐻 Bear Case</div>
          <div style="font-size:14px;line-height:1.7;color:#7f1d1d;">
            {_md_to_html(results.get('bear', ''))}
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
<div style="max-width:680px;margin:24px auto;background:#f1f5f9;">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1e3a5f,#2563eb);padding:28px 32px;
              border-radius:12px 12px 0 0;color:#ffffff;">
    <div style="font-size:22px;font-weight:900;letter-spacing:-0.3px;">{company}</div>
    <div style="font-size:15px;opacity:0.8;margin-top:2px;">{ticker} &nbsp;·&nbsp; {now_str}</div>
    <div style="margin-top:14px;display:flex;gap:20px;flex-wrap:wrap;">
      <span style="font-size:26px;font-weight:800;">${price:.2f}</span>
      <span style="font-size:16px;font-weight:700;color:{chg_color};
                   background:rgba(255,255,255,0.15);padding:4px 10px;
                   border-radius:6px;align-self:center;">
        {chg_sign}{chg:.2f}%
      </span>
      <span style="font-size:13px;opacity:0.7;align-self:center;">Depth: {depth}</span>
    </div>
  </div>

  <!-- Body -->
  <div style="background:#f8fafc;padding:24px 32px;border-radius:0 0 12px 12px;">

    <!-- Final Decision -->
    <div style="margin-bottom:24px;padding:20px 24px;background:{bg};
                border:2px solid {border};border-radius:10px;">
      <div style="font-size:18px;font-weight:900;color:{fg};margin-bottom:12px;">
        ⚖️ FINAL DECISION — {dlbl}
      </div>
      <div style="font-size:14px;line-height:1.8;color:{fg};">
        {_md_to_html(results.get('judge', ''))}
      </div>
    </div>

    <!-- Bull / Bear -->
    {bull_html}
    {bear_html}

    <!-- Specialist Reports -->
    <div style="font-size:13px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;
                color:#64748b;margin-bottom:14px;padding-bottom:6px;
                border-bottom:1px solid #e2e8f0;">📋 Specialist Reports</div>
    {sections_html}

    <!-- Footer -->
    <div style="margin-top:24px;padding-top:16px;border-top:1px solid #e2e8f0;
                font-size:11px;color:#94a3b8;text-align:center;line-height:1.6;">
      US Stock Multi-Agent Analyzer &nbsp;·&nbsp; {now_str}<br>
      ⚠️ Untuk keperluan edukasi. Bukan saran keuangan.
    </div>
  </div>
</div>
</body></html>"""


def send_email_analysis(ticker, company, price, chg, dcls, dlbl, results, depth):
    """Send analysis email via Gmail SMTP. Returns (ok: bool, error_msg: str)."""
    try:
        smtp_user = st.secrets.get("SMTP_USER", "")
        smtp_pass = st.secrets.get("SMTP_PASS", "")
    except Exception:
        smtp_user, smtp_pass = "", ""

    if not smtp_user or not smtp_pass:
        return False, (
            "SMTP belum dikonfigurasi. Tambahkan **SMTP_USER** dan **SMTP_PASS** "
            "di *Settings → Secrets* pada Streamlit Cloud."
        )

    to_email  = "hendro.warsito@gmail.com"
    now_str   = datetime.now().strftime('%Y-%m-%d %H:%M')
    subject   = f"[Stock Analysis] {ticker} → {dlbl}  |  {now_str}"
    body_html = _build_email_html(ticker, company, price, chg, dcls, dlbl, results, depth)

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = smtp_user
        msg["To"]      = to_email
        msg.attach(MIMEText(body_html, "html", "utf-8"))

        with smtplib.SMTP("smtp.gmail.com", 587) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(smtp_user, smtp_pass)
            srv.send_message(msg)

        return True, ""
    except Exception as e:
        return False, str(e)


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

    # Persist analysis to session state for email button (survives re-render)
    st.session_state["_last_analysis"] = dict(
        ticker=ticker, company=company, price=price, chg=chg,
        dcls=dcls, dlbl=dlbl, results=results, depth=depth,
    )

    st.markdown("---")
    st.caption(
        f"⏱ {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
        f"Model: claude-sonnet-4-5  ·  Depth: {depth}  ·  "
        f"⚠️ Educational purposes only. Not financial advice."
    )

# ── Email button (rendered every run if analysis exists in session state) ─────
if "_last_analysis" in st.session_state:
    la = st.session_state["_last_analysis"]
    st.markdown("---")
    ec1, ec2, ec3 = st.columns([3, 2, 1])
    with ec2:
        if st.button("📧 Kirim Analisis ke Email", use_container_width=True, key="btn_send_email"):
            with st.spinner("Mengirim email ke hendro.warsito@gmail.com..."):
                ok, err = send_email_analysis(**la)
            if ok:
                st.success("✅ Analisis berhasil dikirim ke **hendro.warsito@gmail.com**")
            else:
                st.error(f"❌ Gagal mengirim: {err}")
                with st.expander("💡 Cara konfigurasi SMTP"):
                    st.markdown("""
**Langkah setup Gmail SMTP di Streamlit Cloud:**

1. Aktifkan **2-Step Verification** di akun Gmail pengirim
2. Buat **App Password**: myaccount.google.com → Security → App Passwords
3. Di Streamlit Cloud → App Settings → **Secrets**, tambahkan:
```toml
SMTP_USER = "alamat_gmail_pengirim@gmail.com"
SMTP_PASS = "xxxx xxxx xxxx xxxx"   # 16-char App Password
```
4. Redeploy app (atau tunggu auto-reload)
""")
    st.session_state.setdefault("_email_sent_ticker", None)
