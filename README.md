# 🧠 Nova Intelligence Agent

> **Not just news. Intelligence.**

A voice-powered multi-agent news intelligence system using **Amazon Nova AI**. Fetches news from multiple sources in parallel, analyzes sentiment, extracts trends, and generates AI summaries.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)
![Nova AI](https://img.shields.io/badge/Amazon_Nova-AI-orange)

---

## ⚡ Features at a Glance

| Feature | Description | Backend |
|---------|-------------|---------|
| 📰 **Multi-Source Fetch** | Parallel news from Tavily, GNews & RSS | `asyncio` concurrent pipelines |
| 🔄 **Auto-Failover** | Quota/failure-aware source switching | Priority queue + retry logic |
| 🧠 **AI Summary** | Nova-powered intelligent digests | Amazon Nova Lite LLM |
| 💭 **Sentiment Intelligence** | Institutional analyst-style analysis | NLP + Nova AI inference |
| 📊 **Trend Extraction** | Hot topics via entity recognition | Regex NER + frequency analysis |
| 💾 **Multi-Format Export** | JSON, Markdown, CSV reports | Template engine + streaming |
| 🎤 **Voice Interface** | Browser speech recognition | Web Speech API |
| 🕒 **Search History** | Persisted recent searches | LocalStorage + JSON store |
| 📖 **Dictionary Lookup** | Instant word definitions | Merriam-Webster API |
| 🌐 **Translation** | Translate reports to 18+ languages | MyMemory API |
| ⚙️ **Settings Panel** | Customize language & dictionary prefs | LocalStorage persistence |
| 🎛️ **Dynamic Panels** | Toggle-responsive layout with smooth animations | CSS Grid + auto-fit |
| 💊 **Summary Chip** | Collapsed execution summary, expandable trace view | JS state management |
| 📦 **Package Builder** | Intelligence packaging with preview, multi-format export | Quality badges + smart recommendations |

---

## 🌐 Language & Dictionary Tools

> *Understand any term. Read in any language.*

### 📖 Dictionary Lookup

```
┌──────────────────────────────────────────────────────────────┐
│  🧠 Intelligence Report                    [DICT] [Translate]│
├──────────────────────────────────────────────────────────────┤
│  Click DICT → Type word → Press GO                           │
│  ──────────────────────────────────────────────────          │
│  ┌─────────────────────────────────┐                         │
│  │ 📖 momentum                     │                         │
│  │ noun                            │                         │
│  │ • the strength or force of      │                         │
│  │   movement or change            │                         │
│  │ Source: Merriam-Webster         │                         │
│  └─────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

**How it works:**
1. Click **DICT** button in Intelligence Report header
2. Type any English word (e.g., "tariff", "bullish", "volatility")
3. Press **GO** or hit Enter
4. Definition popup appears instantly

**Backend:** Merriam-Webster Collegiate Dictionary API (1000 queries/day free)

---

### 🌐 Translation System

```
┌────────────────────────────────────────────────────────────┐
│  ⚙️ Settings                                               │
├────────────────────────────────────────────────────────────┤
│  🌐 Translation Languages (select up to 3)                 │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │☑ Hindi │ │☐ Spanish│ │☐ French│ │☐ German│ │☐ Chinese│  │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
│  ... (18 languages available)                              │
│                                           [💾 Save Settings]│
└────────────────────────────────────────────────────────────┘
```

**Workflow:**
1. Click **⚙️ Settings** button (top-right)
2. Select up to **3 languages** from 18 available
3. Click **Save Settings**
4. Use **Translate** dropdown on Intelligence Report
5. Translated text appears with **Show Original** button

**Supported Languages:**
`English` `Hindi` `Spanish` `French` `German` `Chinese` `Japanese` `Korean` `Arabic` `Portuguese` `Russian` `Italian` `Tamil` `Telugu` `Bengali` `Marathi` `Gujarati` `Punjabi`

**Backend:** MyMemory Translation API (10,000 chars/day free, no key required)

---

### 🎯 Smart Tooltips

Finance terms in Sentiment Intelligence are highlighted and show explanations on hover:

| Term | Explanation |
|------|-------------|
| **Momentum** | How fast and strong sentiment is changing |
| **Risk-On** | Investors favor risky assets (stocks, crypto) |
| **Risk-Off** | Investors prefer safe assets (bonds, gold) |
| **Bullish Signals** | Factors driving positive sentiment |
| **Confidence** | Certainty level based on data consistency |

*Hover over blue-highlighted terms for 0.5s to see tooltip.*

---

### 💾 Multi-Layer Storage Architecture

| Layer | Storage | Data | Retention |
|-------|---------|------|-----------|
| **Frontend** | LocalStorage | Search history, language prefs, settings | Permanent |
| **Backend** | `app/memory/plans.json` | User task plans | All sessions |
| **Backend** | `app/memory/results.json` | Execution outputs | All sessions |
| **Backend** | `app/memory/logs.json` | Runtime logs & diagnostics | All sessions |
| **Export** | `output/` folder | Timestamped reports | Permanent |

---

## 🏗️ Sentiment & Intelligence Layer Architecture

> *Not just features. A mini intelligence platform.*

### 🔄 End-to-End System Flow

```
User Input → Planner Agent → Task JSON → Executor Agent
                                              ↓
            ┌────────────────────────────────────────────────┐
            │  Tools: News → Summary → Sentiment → Trends   │
            └────────────────────────────────────────────────┘
                                              ↓
                              Memory Store → UI Rendering
                                              ↓
                    Supplementary: /translate │ /dictionary │ /languages
```

### ⚙️ Dictionary Engine

| Aspect | Design |
|--------|--------|
| **Primary API** | Merriam-Webster (if key available) |
| **Fallback API** | Free Dictionary API (zero config) |
| **Trigger** | Click DICT → Type → GO |
| **Reliability** | Dynamic API fallback, zero breakage |

### 🌐 Translation Engine

| Aspect | Design |
|--------|--------|
| **API** | MyMemory (no key required) |
| **Limit** | 500 chars/request (quota protection) |
| **Languages** | 18 languages, max 3 selected |
| **Prefs** | Persisted in LocalStorage |

### ⭐ Architecture Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│  ✅ Multi-layer persistence (Frontend + Backend)               │
│  ✅ External API fallback reliability model                    │
│  ✅ Preference-aware UI behavior                               │
│  ✅ Modular tool-driven intelligence pipeline                  │
│  ✅ Feature toggle driven UX                                   │
│  ✅ Cost-aware planning (fewer tools = less cost + faster)     │
└─────────────────────────────────────────────────────────────────┘
```

> **💡 Smart Execution:** Feature toggles dynamically build commands — planner only invokes tools you enable, reducing API costs and execution time.

### 🛡️ Error Handling & Resilience

| Capability | Status |
|------------|--------|
| **Tool Error Isolation** | ✅ Step failure doesn't kill pipeline |
| **Per-Step Error Logging** | ✅ Errors stored individually |
| **Graceful Degradation** | ✅ Partial results returned |
| **Tool Retry Logic** | ✅ 2 retries with exponential backoff |
| **Dependency Graph Execution** | ✅ Skip tool if dependency failed |
| **Alternate Tool Fallback** | ✅ Fallback summarizer & sentiment |
| **Auto Step Regeneration** | ✅ Retry with reduced params |
| **Dynamic Plan Rewriting** | ✅ Recovery on critical failures |

>
## 🎯 Feature Deep-Dive

### 📰 Multi-Source News Intelligence

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Tavily    │  │   GNews     │  │  RSS Feeds  │
│  (Web AI)   │  │   (API)     │  │ (Google)    │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌─────────────────┐
              │ Parallel Fetch  │ ← asyncio.gather()
              │ + Failover      │ ← Quota/Error handling
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Deduplication  │ ← Title similarity + URL hash
              │  & Merge        │ ← Source priority scoring
              └─────────────────┘
```

**Under the Hood:**
- **Concurrent Pipeline** — All sources queried in parallel via `asyncio`
- **Smart Failover** — Auto-switches sources on quota exhaustion or HTTP errors
- **Deduplication Engine** — Fuzzy title matching + URL normalization

---

### 💭 Sentiment Intelligence V2 — *"Narrative, Not Numbers"*

> 🧮 **OLD:** Word counting → *"60% positive"*  
> 🧠 **NEW:** Market narrative intelligence → *"Bullish momentum with regulatory headwinds"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🔮 INTELLIGENCE OUTPUT                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  mood_label     →  "Strong bullish momentum" / "Risk-off prevailing"   │
│  direction      →  improving ↗ │ stable → │ deteriorating ↘            │
│  confidence     →  high ● │ medium ◐ │ low ○                           │
│  market_bias    →  risk_on 🟢 │ balanced 🟡 │ risk_off 🔴               │
│  risk_level     →  Regulatory threat scan + crisis detection           │
│  reasoning      →  "Coverage shows X, momentum suggests Y..."          │
├─────────────────────────────────────────────────────────────────────────┤
│  ✅ positive_signals  →  ["M&A activity", "Product launches"...]       │
│  ⚠️  negative_signals  →  ["Regulatory concerns", "Market decline"...] │
│  🔥 emerging_themes   →  ["Tesla", "Fed", "Nvidia"...] (NER-extracted) │
└─────────────────────────────────────────────────────────────────────────┘
```

**🔧 Under the Hood:**

| Layer | What It Does |
|-------|--------------|
| **Nova LLM** | Bloomberg-style analyst prompt → narrative reasoning |
| **Keyword Velocity** | Momentum words (`surge`, `rally`) + risk words (`lawsuit`, `probe`) |
| **Signal Extractor** | Categorizes drivers into bullish vs bearish buckets |
| **Theme NER** | Proper noun extraction → trending entity detection |
| **Validation Layer** | `_validate_sentiment()` ensures production-safe output |
| **Fallback Engine** | Smart mock when LLM unavailable (not random — deterministic)

---

### 🧠 AI Summarization Engine

| Aspect | Details |
|--------|---------|
| **Model** | Amazon Nova Lite v1 |
| **Context** | Up to 12 headlines per analysis |
| **Output** | 2-3 sentence executive digest |
| **Fallback** | Template-based summary on error |

---

### 📊 Trend Intelligence V2 — *"Velocity, Not Just Frequency"*

> 📝 **OLD:** Keyword counting → *"OpenAI: 5 mentions"*  
> 🔥 **NEW:** Time-weighted velocity detection → *"🔥 OpenAI GPT rising fast"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📊 TREND INTELLIGENCE V2                                               │
├─────────────────────────────────────────────────────────────────────────┤
│  🔥 Rising Fast    │  📈 Rising    │  ➡️ Stable    │  📉 Fading        │
│  India Trade Deal  │  Market Rally │  Fed Policy   │  Old Topic        │
│  score: 12.5       │  score: 8.2   │  score: 5.0   │                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ⏱️ Time Weighting  │  <2hrs=3x │ <6hrs=2.5x │ <24hrs=1.5x            │
│  📝 N-Gram Phrases  │  "OpenAI GPT" not "OpenAI" + "GPT"              │
│  ⚖️ Source Weights   │  Tavily=1.5x │ GNews=1.3x │ RSS=1.0x            │
└─────────────────────────────────────────────────────────────────────────┘
```

**🔧 Under the Hood:**

| Feature | What It Does |
|---------|--------------|
| **Time Weighting** | Recent articles get 3x weight (2hrs) → 1.5x (24hrs) |
| **N-Gram Phrases** | Bigram/trigram extraction: "OpenAI GPT" not split |
| **Velocity Tracking** | History-based rising/fading via `trends_history.json` |
| **Source Reliability** | Tavily=1.5x, GNews=1.3x, RSS=1.0x scoring |
| **Story Direction** | 🟢 Positive / 🔴 Critical/Risk / 🟡 Controversial / 🔵 Emerging |
| **News Cycle Stage** | 🆕 Breaking / 🔥 Peak Focus / 📰 Major / 📉 Losing Attention |
| **Why Trending (LLM)** | Nova-powered: *"Trade deal announced after Modi-Trump call"* |
| **News Summary** | Topic-aware: *"Active coverage on India, Trump themes"* |

> 🧠 **Hybrid Analysis:** Score-based velocity + Nova LLM headline analysis  
> 💡 **Mental Model:** *Trend = Attention* | *Sentiment = Framing* | *Narrative = Story*

---

### 📦 Intelligence Package Builder — *"Not Save. Deliver."*

> 📁 **OLD:** Basic file export → *"Download JSON"*  
> 📦 **NEW:** Intelligence Packaging Center → *Preview → Quality Badge → Smart Format → Multi-Export*

```
┌─────────────────────────────────────────────────────────────────────────┐
│  📦 INTELLIGENCE PACKAGE BUILDER                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Quality     │  🟢 Full Report │ 🟡 Partial │ 🔴 Raw Data              │
│  Preview     │  ✔ News  ✔ Summary  ✔ Sentiment  ✗ Trends              │
│  Stats       │  📰 15 Articles │ 📁 3 Sections │ ~45 KB                │
│  Recommend   │  💡 Markdown for rich formatting                        │
├─────────────────────────────────────────────────────────────────────────┤
│  [{ } JSON]  [📄 Markdown]  [📊 CSV]                                   │
│  [⬇️ Download All Formats]                                              │
│  [📋 Copy JSON to Clipboard]                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

**🔧 Under the Hood:**

| Feature | What It Does |
|---------|--------------|
| **Quality Badges** | 🟢 Full (3+ sections) → 🟡 Partial (2) → 🔴 Raw (1) |
| **Smart Recommend** | Markdown for rich, CSV for data, JSON for API |
| **Toggle-Aware** | Exports only enabled features |
| **Multi-Export** | One-click download all 3 formats |
| **Copy Clipboard** | Instant JSON copy for dev use |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Voice/Text    │────▶│   Nova Planner   │────▶│    Executor     │
│     Input       │     │  (Amazon Nova)   │     │     Agent       │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        │                                 │                                 │
                        ▼                                 ▼                                 ▼
              ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
              │  Tavily Search  │              │   GNews API     │              │   RSS Feeds     │
              │  (Web Search)   │              │  (News API)     │              │ (Google News)   │
              └────────┬────────┘              └────────┬────────┘              └────────┬────────┘
                       │                                │                                │
                       └────────────────────────────────┼────────────────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │   Deduplicator  │
                                              │  & Merger       │
                                              └────────┬────────┘
                                                       │
                        ┌──────────────────────────────┼──────────────────────────────┐
                        │                              │                              │
                        ▼                              ▼                              ▼
              ┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
              │   Summarizer    │            │   Sentiment     │            │    Trends       │
              │   (Nova AI)     │            │   Analyzer      │            │   Extractor     │
              └─────────────────┘            └─────────────────┘            └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │    Exporter     │
                                              │  JSON/MD/CSV    │
                                              └─────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
copy .env.example .env
```

Edit `.env` with your API keys:

```env
# AWS for Nova AI
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# News APIs
TAVILY_API_KEY=your_tavily_key    # https://tavily.com
GNEWS_API_KEY=your_gnews_key      # https://gnews.io
```

### 3. Run Server

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Open Browser

Navigate to `http://localhost:8000`

---

## 🎤 Usage

### Voice Commands
```
"Tesla news"
"India US trade deal with sentiment"
"AI news with summary and trends"
"Stock market analysis"
```

### Feature Toggles
Click badges in the UI to enable/disable:
- 📰 **News** - Always on
- 🧠 **Summary** - AI digest
- 💭 **Sentiment** - Tone analysis
- 📊 **Trends** - Hot topics
- 💾 **Export** - Save results

---

## 📁 Project Structure

```
NovaAI/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI entry point
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner_agent.py       # Nova AI task planning
│   │   └── executor_agent.py      # Tool orchestration & context
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # REST API endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tool_registry.py       # Tool management
│   │   └── plan_validator.py      # Plan validation
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py               # Logging & persistence
│   │   ├── logs.json              # Execution logs
│   │   ├── plans.json             # Saved plans
│   │   └── results.json           # Execution results
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic models
│   │
│   └── tools/
│       ├── __init__.py
│       ├── multi_fetcher.py       # Parallel multi-source fetch
│       ├── tavily_fetcher.py      # Tavily web search API
│       ├── gnews_fetcher.py       # GNews API
│       ├── rss_fetcher.py         # Google News RSS
│       ├── news_fetcher.py        # Legacy single-source
│       ├── summarizer.py          # Nova AI summary
│       ├── sentiment.py           # Nova AI sentiment
│       ├── trends.py              # Topic extraction
│       └── exporter.py            # JSON/MD/CSV export
│
├── frontend/
│   ├── index.html                 # Voice UI interface
│   ├── app.js                     # Frontend logic & history
│   └── style.css                  # Professional dark theme
│
├── output/                        # Exported report files
├── .env                           # API keys (not in git)
├── .env.example                   # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/command` | Process voice/text command |
| GET | `/api/capabilities` | Get available tools |
| GET | `/api/history` | Get recent commands |
| POST | `/api/translate` | Translate text to target language |
| GET | `/api/languages` | Get available translation languages |
| GET | `/api/dictionary/{word}` | Get word definition |

### Example Request

```bash
curl -X POST http://localhost:8000/api/command \
  -H "Content-Type: application/json" \
  -d '{"text": "Tesla news with sentiment"}'
```

---

## 🔧 Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | AWS IAM key for Nova |
| `AWS_SECRET_ACCESS_KEY` | Yes | AWS secret key |
| `AWS_REGION` | Yes | AWS region (us-east-1) |
| `TAVILY_API_KEY` | Optional | Tavily web search |
| `GNEWS_API_KEY` | Optional | GNews API |
| `USE_MOCK_PLANNER` | Optional | Use mock for testing |

---

## 📊 Sample Output

```json
{
  "summary": "The Tesla-India trade deal focuses on...",
  "sentiment": {
    "overall": "positive",
    "score": 0.72,
    "breakdown": {"positive": 5, "neutral": 3, "negative": 2}
  },
  "trends": [
    {"topic": "Tesla", "mentions": 8},
    {"topic": "India", "mentions": 6}
  ],
  "articles": [...]
}
```

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

Built with ❤️ using Amazon Nova AI
MUKUL PRASAD
