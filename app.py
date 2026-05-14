import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import anthropic
import json
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Stock Analyzer – Multi-Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background: #0e1117; }
    .agent-card {
        background: #1a1d24;
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .agent-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .bull  { color: #00d084; }
    .bear  { color: #ff4b6e; }
    .neut  { color: #f0b429; }
    .judge { color: #7c6ff7; }
    .tech  { color: #38bdf8; }
    .fund  { color: #fb923c; }
    .news  { color: #a78bfa; }
    .sent  { color: #34d399; }
    .decision-buy  { background:#00d08422; border:2px solid #00d084; border-radius:12px; padding:20px; }
    .decision-sell { background:#ff4b6e22; border:2px solid #ff4b6e; border-radius:12px; padding:20px; }
    .decision-hold { background:#f0b42922; border:2px solid #f0b429; border-radius:12px; padding:20px; }
    .metric-box {
        background: #1a1d24;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stProgress > div > div { background: linear-gradient(90deg, #7c6ff7, #38bdf8); }
    div[data-testid="stExpander"] { border: 1px solid #2d3139; border-radius: 8px; }
    .watchlist-btn button { width: 100%; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password",
                            help="sk-ant-... key from console.anthropic.com")

    st.markdown("---")
    st.markdown("### 📋 Watchlist")
    DEFAULT_WATCHLIST = ["NVDA", "PLTR", "TSLA", "DUOL", "AMD", "META", "AMZN", "ASTS"]
    
    selected_ticker = None
    cols = st.columns(2)
    for i, t in enumerate(DEFAULT_WATCHLIST):
        if cols[i % 2].button(t, key=f"wb_{t}", use_container_width=True):
            selected_ticker = t

    st.markdown("---")
    st.markdown("### 🔍 Or Enter Ticker")
    manual_ticker = st.text_input("Ticker Symbol", placeholder="e.g. AAPL").upper().strip()

    st.markdown("---")
    st.markdown("### 🎯 Analysis Depth")
    depth = st.select_slider("Agent Reasoning Depth",
                             options=["Quick", "Standard", "Deep"],
                             value="Standard")
    depth_tokens = {"Quick": 600, "Standard": 1000, "Deep": 1500}

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("**US Stock Analyzer v1**\nMulti-Agent system powered by Claude.\n\nAgents: Fundamental · Technical · News · Sentiment → Bull/Bear Debate → Risk Judge")

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 📊 US Stock Multi-Agent Analyzer")
st.markdown("*Powered by Claude · Multi-Agent Debate Framework*")

ticker = manual_ticker if manual_ticker else (selected_ticker or "")

if not ticker:
    st.info("👈 Select a ticker from the watchlist or enter one manually to begin analysis.")
    
    st.markdown("---")
    st.markdown("### 🚀 How It Works")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**1️⃣ Data Collection**\n\nPrice data, financials, and technicals via yfinance")
    with c2:
        st.markdown("**2️⃣ Specialist Agents**\n\nFundamental, Technical, Sentiment analysts work in parallel")
    with c3:
        st.markdown("**3️⃣ Bull/Bear Debate**\n\nTwo researcher agents argue opposing sides")
    with c4:
        st.markdown("**4️⃣ Risk Judge**\n\nFinal decision with tiered entry plan & stop-loss")
    st.stop()

if not api_key:
    st.warning("⚠️ Please enter your Anthropic API key in the sidebar to run the analysis.")
    st.stop()

# ── Data Fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_stock_data(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        hist = tk.history(period="6mo", interval="1d")
        hist_1y = tk.history(period="1y", interval="1d")
        
        if hist.empty:
            return None, None, None, None
        
        # Technical indicators
        hist.ta.rsi(append=True)
        hist.ta.macd(append=True)
        hist.ta.bbands(append=True)
        hist.ta.sma(length=20, append=True)
        hist.ta.sma(length=50, append=True)
        hist.ta.sma(length=200, append=True)
        hist.ta.atr(append=True)
        
        # Earnings
        try:
            earnings = tk.earnings_dates
        except:
            earnings = None
            
        return info, hist, hist_1y, earnings
    except Exception as e:
        return None, None, None, None

@st.cache_data(ttl=300)
def get_news(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        news = tk.news
        if news:
            return news[:8]
        return []
    except:
        return []

# ── Agent Calls ───────────────────────────────────────────────────────────────
def call_agent(client, system_prompt: str, user_prompt: str, max_tokens: int = 1000) -> str:
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"[Agent error: {str(e)}]"

def build_data_context(ticker, info, hist, news_items) -> str:
    close = hist["Close"].iloc[-1]
    prev  = hist["Close"].iloc[-2]
    chg   = (close - prev) / prev * 100
    
    sma20  = hist.get("SMA_20",  pd.Series([None])).iloc[-1]
    sma50  = hist.get("SMA_50",  pd.Series([None])).iloc[-1]
    sma200 = hist.get("SMA_200", pd.Series([None])).iloc[-1]
    rsi    = hist.filter(like="RSI").iloc[-1, 0] if not hist.filter(like="RSI").empty else None
    atr    = hist.filter(like="ATRr").iloc[-1, 0] if not hist.filter(like="ATRr").empty else None
    macd   = hist.filter(like="MACD_").iloc[-1, 0] if not hist.filter(like="MACD_").empty else None
    bb_up  = hist.filter(like="BBU").iloc[-1, 0] if not hist.filter(like="BBU").empty else None
    bb_lo  = hist.filter(like="BBL").iloc[-1, 0] if not hist.filter(like="BBL").empty else None
    bb_mid = hist.filter(like="BBM").iloc[-1, 0] if not hist.filter(like="BBM").empty else None
    
    vol_avg = hist["Volume"].tail(20).mean()
    vol_now = hist["Volume"].iloc[-1]
    
    safe = lambda v, fmt=".2f": f"{v:{fmt}}" if v is not None and pd.notna(v) else "N/A"
    
    news_text = "\n".join([f"- {n.get('title','')}" for n in news_items[:5]]) or "No recent news"
    
    fundamentals = {
        "Market Cap":    info.get("marketCap"),
        "Forward P/E":   info.get("forwardPE"),
        "Trailing P/E":  info.get("trailingPE"),
        "PEG Ratio":     info.get("pegRatio"),
        "Revenue Growth":info.get("revenueGrowth"),
        "EPS Forward":   info.get("forwardEps"),
        "Profit Margin": info.get("profitMargins"),
        "Debt/Equity":   info.get("debtToEquity"),
        "Free Cash Flow":info.get("freeCashflow"),
        "52W High":      info.get("fiftyTwoWeekHigh"),
        "52W Low":       info.get("fiftyTwoWeekLow"),
        "Analyst Target":info.get("targetMeanPrice"),
        "Recommendation":info.get("recommendationKey"),
        "Sector":        info.get("sector"),
        "Industry":      info.get("industry"),
    }
    
    fund_str = "\n".join([f"  {k}: {v}" for k, v in fundamentals.items() if v is not None])
    
    return f"""
=== {ticker} MARKET DATA ({datetime.now().strftime('%Y-%m-%d')}) ===

PRICE ACTION:
  Current Price: ${safe(close)}
  Daily Change:  {safe(chg)}%
  Volume:        {safe(vol_now, ',.0f')} (20d avg: {safe(vol_avg, ',.0f')})

TECHNICAL INDICATORS:
  RSI (14):      {safe(rsi)}
  ATR (14):      {safe(atr)}
  MACD:          {safe(macd)}
  SMA 20:        {safe(sma20)}
  SMA 50:        {safe(sma50)}
  SMA 200:       {safe(sma200)}
  BB Upper:      {safe(bb_up)}
  BB Middle:     {safe(bb_mid)}
  BB Lower:      {safe(bb_lo)}
  Price vs 200d: {safe((close/sma200 - 1)*100 if sma200 else None)}%

FUNDAMENTALS:
{fund_str}

RECENT NEWS (last 5):
{news_text}
"""

# ── Chart ─────────────────────────────────────────────────────────────────────
def render_chart(hist, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"],
        name="Price", increasing_line_color="#00d084",
        decreasing_line_color="#ff4b6e"), row=1, col=1)
    
    # MAs
    for col_name, color, label in [
        ("SMA_20", "#f0b429", "SMA20"),
        ("SMA_50", "#38bdf8", "SMA50"),
        ("SMA_200","#ff4b6e", "SMA200"),
    ]:
        if col_name in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[col_name],
                                     line=dict(color=color, width=1.2),
                                     name=label, opacity=0.9), row=1, col=1)
    
    # BB
    bb_up = hist.filter(like="BBU")
    bb_lo = hist.filter(like="BBL")
    if not bb_up.empty and not bb_lo.empty:
        fig.add_trace(go.Scatter(x=hist.index, y=bb_up.iloc[:, 0],
                                  line=dict(color="#7c6ff7", width=1, dash="dot"),
                                  name="BB Upper", opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=bb_lo.iloc[:, 0],
                                  line=dict(color="#7c6ff7", width=1, dash="dot"),
                                  name="BB Lower", opacity=0.6,
                                  fill="tonexty", fillcolor="rgba(124,111,247,0.05)"),
                      row=1, col=1)
    
    # Volume
    colors = ["#00d084" if c >= o else "#ff4b6e"
              for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"],
                          name="Volume", marker_color=colors, opacity=0.6),
                  row=2, col=1)
    
    # RSI
    rsi_col = hist.filter(like="RSI")
    if not rsi_col.empty:
        fig.add_trace(go.Scatter(x=hist.index, y=rsi_col.iloc[:, 0],
                                  line=dict(color="#fb923c", width=1.5),
                                  name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_color="#ff4b6e", line_dash="dot",
                      line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_color="#00d084", line_dash="dot",
                      line_width=1, row=3, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=550,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"{ticker} – 6 Month Chart", x=0.5,
                   font=dict(size=14, color="#e2e8f0"))
    )
    fig.update_yaxes(gridcolor="#1e2129", row=1, col=1)
    fig.update_yaxes(gridcolor="#1e2129", row=2, col=1)
    fig.update_yaxes(gridcolor="#1e2129", row=3, col=1, range=[0, 100])
    
    return fig

# ── Run Analysis ──────────────────────────────────────────────────────────────
run_btn = st.button(f"🚀 Run Multi-Agent Analysis: {ticker}", type="primary",
                    use_container_width=True)

if run_btn:
    client = anthropic.Anthropic(api_key=api_key)
    max_tok = depth_tokens[depth]
    
    # 1. Fetch Data
    with st.spinner("📡 Fetching market data..."):
        info, hist, hist_1y, earnings = get_stock_data(ticker)
        news_items = get_news(ticker)
    
    if hist is None or hist.empty:
        st.error(f"❌ Could not fetch data for **{ticker}**. Please check the ticker symbol.")
        st.stop()
    
    company_name = info.get("longName", ticker)
    current_price = hist["Close"].iloc[-1]
    data_ctx = build_data_context(ticker, info, hist, news_items)
    
    # ── Header ──
    st.markdown(f"## {company_name} ({ticker})")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    prev_close = hist["Close"].iloc[-2]
    daily_chg  = (current_price - prev_close) / prev_close * 100
    
    col1.metric("Current Price",  f"${current_price:.2f}", f"{daily_chg:+.2f}%")
    col2.metric("Market Cap",     f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else "N/A")
    col3.metric("Forward P/E",    f"{info.get('forwardPE', 'N/A'):.1f}" if info.get('forwardPE') else "N/A")
    col4.metric("52W High",       f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else "N/A")
    
    rsi_val = hist.filter(like="RSI").iloc[-1, 0] if not hist.filter(like="RSI").empty else None
    col5.metric("RSI (14)",       f"{rsi_val:.1f}" if rsi_val else "N/A",
                "Overbought" if rsi_val and rsi_val > 70 else
                ("Oversold" if rsi_val and rsi_val < 30 else "Neutral"))
    
    # Chart
    st.plotly_chart(render_chart(hist, ticker), use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 🤖 Agent Analysis")
    
    results = {}
    progress = st.progress(0)
    status   = st.empty()
    
    # ── Agent 1: Fundamental ──────────────────────────────────────────────────
    status.markdown("**1/7** 📈 Fundamental Analyst working...")
    results["fundamental"] = call_agent(client,
        system_prompt="""You are a Senior Fundamental Analyst specializing in US equities.
Analyze the stock data provided. Focus on:
- Valuation (P/E, PEG, forward estimates)
- Growth metrics and quality of earnings
- Balance sheet health (debt, FCF)
- Competitive moat assessment
- Intrinsic value vs current price
Be concise but data-driven. End with a clear BULLISH / BEARISH / NEUTRAL stance.""",
        user_prompt=f"Perform fundamental analysis on {ticker}:\n{data_ctx}",
        max_tokens=max_tok)
    progress.progress(1/7)
    
    # ── Agent 2: Technical ────────────────────────────────────────────────────
    status.markdown("**2/7** 📉 Technical Analyst working...")
    results["technical"] = call_agent(client,
        system_prompt="""You are a Senior Technical Analyst with 20+ years of market experience.
Analyze the technical indicators provided. Focus on:
- Trend analysis (price vs MAs, golden/death cross)
- Momentum (RSI, MACD signals)
- Volatility (ATR, Bollinger Band positioning)
- Support and resistance levels
- Entry/exit price levels
Be specific with price targets. End with BULLISH / BEARISH / NEUTRAL.""",
        user_prompt=f"Perform technical analysis on {ticker}:\n{data_ctx}",
        max_tokens=max_tok)
    progress.progress(2/7)
    
    # ── Agent 3: News/Macro ───────────────────────────────────────────────────
    status.markdown("**3/7** 📰 News & Macro Analyst working...")
    results["news"] = call_agent(client,
        system_prompt="""You are a News and Macro Analyst covering US equities.
Analyze the recent headlines and assess:
- Key catalysts (positive or negative)
- Sector/macro tailwinds or headwinds
- Regulatory or competitive risks
- Upcoming events (earnings, product launches, macro data)
- Sentiment from news flow
End with BULLISH / BEARISH / NEUTRAL.""",
        user_prompt=f"Analyze news and macro context for {ticker}:\n{data_ctx}",
        max_tokens=max_tok)
    progress.progress(3/7)
    
    # ── Agent 4: Sentiment ────────────────────────────────────────────────────
    status.markdown("**4/7** 💬 Sentiment Analyst working...")
    results["sentiment"] = call_agent(client,
        system_prompt="""You are a Market Sentiment Analyst.
Based on available data, assess:
- Institutional vs retail sentiment
- Short interest implications
- Analyst consensus and price target spread
- Volume and momentum signals as sentiment proxies
- Contrarian indicators (extreme bullish = bearish signal, etc.)
End with BULLISH / BEARISH / NEUTRAL.""",
        user_prompt=f"Assess market sentiment for {ticker}:\n{data_ctx}",
        max_tokens=max_tok)
    progress.progress(4/7)
    
    # ── Agent 5: Bull Researcher ───────────────────────────────────────────────
    status.markdown("**5/7** 🐂 Bull Researcher building case...")
    bull_ctx = f"""
ANALYST REPORTS SUMMARY:
FUNDAMENTAL: {results['fundamental'][:500]}
TECHNICAL: {results['technical'][:500]}
NEWS: {results['news'][:500]}
SENTIMENT: {results['sentiment'][:500]}
"""
    results["bull"] = call_agent(client,
        system_prompt="""You are a Bull Researcher — your job is to make the strongest possible BULLISH case.
Use the analyst reports and data to argue WHY this stock should be bought NOW.
Highlight the most compelling upside catalysts, growth drivers, and technical setups.
Be aggressive but grounded in facts. 3-5 bullet points with your top bull arguments.""",
        user_prompt=f"Make the bull case for {ticker}:\n{data_ctx}\n{bull_ctx}",
        max_tokens=max_tok)
    progress.progress(5/7)
    
    # ── Agent 6: Bear Researcher ──────────────────────────────────────────────
    status.markdown("**6/7** 🐻 Bear Researcher building case...")
    results["bear"] = call_agent(client,
        system_prompt="""You are a Bear Researcher — your job is to make the strongest possible BEARISH case.
Use the analyst reports and data to argue WHY this stock is risky or should be avoided/sold.
Highlight valuation concerns, technical red flags, macro risks, and downside scenarios.
Be incisive but factual. 3-5 bullet points with your top bear arguments.""",
        user_prompt=f"Make the bear case for {ticker}:\n{data_ctx}\n{bull_ctx}",
        max_tokens=max_tok)
    progress.progress(6/7)
    
    # ── Agent 7: Risk Judge (Final Decision) ──────────────────────────────────
    status.markdown("**7/7** ⚖️ Risk Judge deliberating final decision...")
    judge_ctx = f"""
FULL ANALYST REPORTS:
=== FUNDAMENTAL ===
{results['fundamental']}

=== TECHNICAL ===
{results['technical']}

=== NEWS & MACRO ===
{results['news']}

=== SENTIMENT ===
{results['sentiment']}

=== BULL CASE ===
{results['bull']}

=== BEAR CASE ===
{results['bear']}
"""
    results["judge"] = call_agent(client,
        system_prompt="""You are the Chief Risk Officer and final decision-maker for a professional trading firm.
You have reviewed all analyst reports and the bull/bear debate.
Your job is to deliver a FINAL TRADE DECISION with:

1. DECISION: BUY / HOLD / SELL (with conviction level: HIGH/MEDIUM/LOW)
2. RATIONALE: 2-3 sentences on why this outweighs the opposition
3. ENTRY PLAN:
   - Immediate entry %
   - Limit order levels (if applicable)
4. RISK MANAGEMENT:
   - Hard stop-loss price level
   - Key invalidation scenario
5. PROFIT TARGET:
   - Primary target price
   - Trim level (partial profit taking)
6. KEY RISK TO WATCH: The #1 thing that would change your mind

Be decisive. Be specific with price levels. Use the current price data.""",
        user_prompt=f"Make final trade decision for {ticker} (current price ${current_price:.2f}):\n{data_ctx}\n{judge_ctx}",
        max_tokens=max_tok + 200)
    progress.progress(7/7)
    status.empty()
    progress.empty()
    
    # ── Render Results ─────────────────────────────────────────────────────────
    
    # Determine decision color
    judge_text = results["judge"].upper()
    if "BUY" in judge_text[:200]:
        dec_class = "decision-buy";  dec_icon = "🟢"
    elif "SELL" in judge_text[:200]:
        dec_class = "decision-sell"; dec_icon = "🔴"
    else:
        dec_class = "decision-hold"; dec_icon = "🟡"
    
    # Final Decision Box
    st.markdown(f"""
<div class="{dec_class}">
<h2>{dec_icon} FINAL DECISION</h2>
{results['judge'].replace(chr(10), '<br>')}
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bull / Bear
    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.markdown("""<div class="agent-card">
<div class="agent-header bull">🐂 Bull Researcher</div>""", unsafe_allow_html=True)
        st.markdown(results["bull"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_bear:
        st.markdown("""<div class="agent-card">
<div class="agent-header bear">🐻 Bear Researcher</div>""", unsafe_allow_html=True)
        st.markdown(results["bear"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 4 Analyst Reports
    st.markdown("### 📋 Specialist Analyst Reports")
    ta1, ta2, ta3, ta4 = st.tabs([
        "📈 Fundamental", "📉 Technical", "📰 News & Macro", "💬 Sentiment"
    ])
    
    with ta1:
        st.markdown(f"""<div class="agent-card">
<div class="agent-header fund">Fundamental Analyst</div>
{results['fundamental']}
</div>""", unsafe_allow_html=True)
    
    with ta2:
        st.markdown(f"""<div class="agent-card">
<div class="agent-header tech">Technical Analyst</div>
{results['technical']}
</div>""", unsafe_allow_html=True)
    
    with ta3:
        st.markdown(f"""<div class="agent-card">
<div class="agent-header news">News & Macro Analyst</div>
{results['news']}
</div>""", unsafe_allow_html=True)
    
    with ta4:
        st.markdown(f"""<div class="agent-card">
<div class="agent-header sent">Sentiment Analyst</div>
{results['sentiment']}
</div>""", unsafe_allow_html=True)
    
    # Raw Data Expander
    with st.expander("🔬 Raw Market Data Used by Agents"):
        st.code(data_ctx, language="text")
    
    # Footer
    st.markdown("---")
    st.caption(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · Model: claude-sonnet-4-20250514 · Depth: {depth} · ⚠️ For educational purposes only. Not financial advice.")
