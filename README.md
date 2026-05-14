# 📊 US Stock Multi-Agent Analyzer v1

Multi-agent AI system untuk analisa saham US, powered by **Claude (Anthropic)**.

## 🤖 Agent Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   DATA LAYER                            │
│         yfinance + pandas_ta + NewsAPI                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              ANALYST TEAM (Parallel)                    │
│  📈 Fundamental  📉 Technical  📰 News  💬 Sentiment    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│               RESEARCHER DEBATE                         │
│         🐂 Bull Researcher ↔ 🐻 Bear Researcher         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              ⚖️ RISK JUDGE (Final Decision)              │
│    BUY/HOLD/SELL + Entry Plan + Stop Loss + Targets     │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Deploy ke Streamlit Cloud

### 1. Upload ke GitHub
```bash
# Buat repo baru di GitHub, lalu:
git init
git add .
git commit -m "US Stock Analyzer v1"
git remote add origin https://github.com/USERNAME/us-stock-analyzer.git
git push -u origin main
```

### 2. Deploy di Streamlit Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Connect GitHub repo
3. Set `app.py` sebagai main file
4. Di **Secrets**, tambahkan:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

### 3. Jalankan Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Tech Stack
- **Frontend**: Streamlit
- **AI**: Claude (claude-sonnet-4-20250514) via Anthropic API
- **Market Data**: yfinance
- **Technical Analysis**: pandas_ta
- **Charts**: Plotly

## 🔮 Roadmap v2+
- [ ] Memory system (simpan & review keputusan lama)
- [ ] Portfolio scan (analisa semua watchlist sekaligus)
- [ ] Earnings calendar integration
- [ ] PDF export report
- [ ] IDX stocks support
- [ ] Macro context (Fed, DXY, VIX)

## ⚠️ Disclaimer
For educational purposes only. Not financial advice.
