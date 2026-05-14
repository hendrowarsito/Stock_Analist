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

    selected_ticker = None
    for t, shares in HOLDINGS.items():
        px  = prices_live.get(t)
        val = px * shares if px else None
        px_str  = f"${px:,.2f}"  if px  else "–"
        val_str = f"${val:,.2f}" if val else "–"
        c_btn, c_info = st.columns([1, 2])
        with c_btn:
            if st.button(t, key=f"pos_{t}", use_container_width=True):
                selected_ticker = t
        with c_info:
            st.markdown(
                f"<div style='font-size:11px;line-height:1.65;color:#475569;padding-top:2px'>"
                f"<span style='font-weight:700;color:#1e293b'>{px_str}</span><br>"
                f"{shares} sh &nbsp;·&nbsp; <span style='color:#059669;font-weight:700'>{val_str}</span>"
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


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("# 📊 US Stock Multi-Agent Analyzer")
st.markdown("*Powered by Claude · 7-Agent Debate Framework*")

ticker = manual_ticker if manual_ticker else (selected_ticker or "")

if not ticker:
    st.info("👈 Pilih ticker dari watchlist atau ketik manual untuk mulai analisa.")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1️⃣ Data**\nyfinance: price, fundamentals & news")
    c2.markdown("**2️⃣ Analysts**\nFundamental · Technical · News · Sentiment")
    c3.markdown("**3️⃣ Debate**\nBull Researcher vs Bear Researcher")
    c4.markdown("**4️⃣ Decision**\nRisk Judge → Entry · Stop · Target")
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
