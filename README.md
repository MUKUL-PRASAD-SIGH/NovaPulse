# 🧠 Nova — Multi-Agent Intelligence System

> **9 agents. One query. Zero compromise.**

Nova is a **voice-powered Multi-Agent System (MAS)** built on **Amazon Nova AI** that orchestrates 9 specialist agents through a dependency-aware execution graph. Speak a query, and Nova's planner decomposes it into a parallel task tree — fetching, scraping, analyzing, and synthesizing intelligence across news, social, academic, and visual domains simultaneously.

News intelligence is the flagship pipeline. The architecture is the product.

- 🎥 **Demo**: [Watch on YouTube](https://youtu.be/KDNrGJ994Cw)

![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent_System-ff6b6b?style=for-the-badge)
![Agents](https://img.shields.io/badge/Agents-9_Specialist-blueviolet?style=for-the-badge)
![AI](https://img.shields.io/badge/LLM-Amazon_Nova-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?style=for-the-badge)

---

## 🔮 What Makes This a MAS — Not Just Another News App

Most "AI news tools" are glorified API wrappers. Nova is architecturally different:

```
                         +========================+
                         |     PLANNER AGENT      |  <-- Amazon Nova LLM
                         |  Decomposes query into  |      decomposes intent
                         |  a dependency-aware DAG |      into executable plan
                         +===========+============+
                                     |
                                     v
                           +-------------------+
                           |  EXECUTOR AGENT   |  <-- Dependency graph
                           |  (Orchestrator)   |      orchestration
                           +--------+----------+
                                    |
        +----------+----------+-----+-----+----------+----------+
        |          |          |     |     |          |          |
        v          v          v     v     v          v          v
    +--------+ +--------+ +------+----+ +--------+ +--------+ +--------+
    |  News  | |  Summ- | | Senti-    | |  Web   | | Entity | | Image  |
    | Fetcher| | arizer | |  ment     | |Scraper | |Extract | |Analyst |
    +--------+ +--------+ +------+----+ +--------+ +--------+ +--------+
                                  |                      |          |
                             +----+----+           +-----+----+----+-----+
                             |  Trend  |           |  Social  | Research |
                             |Detector |           | Monitor  |  Asst   |
                             +---------+           +----------+---------+
```

> 📰 News · 🧠 Summarizer · 💭 Sentiment · 📊 Trends · 🌐 Scraper · 👤 Entities · 🖼️ Images · 📱 Social · 📚 Research

| Principle | How Nova Implements It |
|-----------|----------------------|
| **Agent Autonomy** | Each agent is self-contained with its own fallback logic |
| **Parallel Execution** | `asyncio.gather()` runs independent agents simultaneously |
| **Dependency Graphs** | Planner generates a DAG — Summarizer waits for News, Entities wait for Scraper |
| **Graceful Degradation** | If an agent fails, dependent agents skip cleanly — pipeline never crashes |
| **Shared Context** | All agents write to a unified context store consumed by the Package Builder |
| **Dynamic Planning** | LLM-generated plans adapt to which agents are toggled on |

---

## ⚡ The 9 Agents

### 🎯 Core Intelligence Pipeline

| Agent | What It Does | Under the Hood |
|-------|-------------|----------------|
| 📰 **News Fetcher** | Parallel news from Tavily, GNews & RSS with auto-failover | `asyncio.gather()` + priority queue |
| 🧠 **Summarizer** | Nova-powered executive digest with context-aware analysis | Amazon Nova Lite LLM |
| 💭 **Sentiment Analyst** | Bloomberg-style narrative intelligence — *not just percentages* | NLP + Nova AI inference |
| 📊 **Trend Detector** | Velocity-weighted hot topics with time decay & source scoring | Regex NER + n-gram + history |

### 🛠️ MAS Extension Agents

| Agent | What It Does | Under the Hood |
|-------|-------------|----------------|
| 🌐 **Web Scraper** | Full article extraction with og:image, metadata parsing | `httpx` + `BeautifulSoup` |
| 👤 **Entity Extractor** | NER — people, orgs, locations + relationship mapping | Regex NER + co-occurrence |
| 🖼️ **Image Analyst** | AI-powered vision analysis — scene description, object detection, type classification, manipulation forensics | Amazon Nova Vision + PIL + EXIF |
| 📱 **Social Monitor** | Reddit & Twitter trend tracking with per-platform sentiment | Reddit API + NLP |
| 📚 **Research Assistant** | arXiv papers, GitHub repos, StackOverflow — academic & dev intel | Multi-API aggregation |

### 🧩 Platform Capabilities

| Feature | Description | Backend |
|---------|-------------|---------|
| 🎤 **Voice Interface** | Speak your query — browser speech recognition | Web Speech API |
| 📦 **Package Builder** | 5-format export (JSON, MD, CSV, Word, PDF) with quality badges | Template engine + streaming |
| 🌐 **Translation** | Translate reports to 18+ languages instantly | MyMemory API |
| 📖 **Dictionary** | Click any word for instant definition | Merriam-Webster API |
| ⚙️ **Settings Panel** | Tool info, language prefs, dictionary config, data management | LocalStorage persistence |
| 🕒 **Search History** | Persisted recent queries with timestamp & status | LocalStorage + JSON store |
| 💊 **Execution Pipeline** | Live step-by-step progress overlay with retry/fallback tracking | JS state machine |
| 🎛️ **Dynamic Panels** | Toggle-responsive layout with smooth CSS animations | CSS Grid + auto-fit |
| 🎯 **Smart Tooltips** | Hover over finance terms for instant explanations | JS hover detection |

---

## 🏗️ System Architecture

### Orchestration Flow

```
User Input --> Planner Agent --> Task DAG --> Executor Agent
                                                   |
              +----------------------------------------------------+
              |  Core: News --> Summary --> Sentiment --> Trends    |
              |  MAS:  Scraper --> Entities --> Images              |
              |        Social Monitor    Research Assistant         |
              +----------------------------------------------------+
                                                   |
                                     Context Store --> UI Rendering
                                                   |
                        Supplementary: /translate | /dictionary | /languages
```

### Full Pipeline

```
+---------------+     +----------------+     +--------------+
|  Voice/Text   |---->|  Nova Planner  |---->|   Executor   |
|    Input      |     | (Amazon Nova)  |     |    Agent     |
+---------------+     +----------------+     +------+-------+
                                                    |
          +----------------+----------------+-------+-------+-----------+
          |                |                |               |           |
          v                v                v               v           v
    +-----------+    +-----------+    +-----------+   +-----------+ +----------+
    |  Tavily   |    |  GNews    |    | RSS Feeds |   | Reddit API| | arXiv/GH |
    |(Web Search|    |  (API)    |    | (Google)  |   | (Social)  | |(Research)|
    +-----+-----+    +-----+-----+    +-----+-----+   +-----+-----+ +----+-----+
          |                |                |               |             |
          +----------------+----------------+               |             |
                           |                                v             v
                    +------+--------+            +-------------------------+
                    | Deduplicator  |            | MAS Agent Aggregation   |
                    |   & Merger    |            +------------+------------+
                    +------+--------+                         |
                           |                                  |
          +----------+-----+------+----------+                |
          |          |            |          |                |
          v          v            v          v                |
    +----------+ +----------+ +----------+                    |
    |Summarizer| |Sentiment | |  Trend   |                    |
    |(Nova AI) | | Analyst  | | Detector |                    |
    +----+-----+ +----+-----+ +----+-----+                    |
         |            |            |                          |
         +------------+------------+                          |
                      |                                       |
               +------+-------+                               |
               | Context Store|<------------------------------+
               |  (Unified)   |
               +------+-------+
                      |
                      v
           +--------------------+
           |  Package Builder   |
           | JSON/MD/CSV/DOCX/  |
           |        PDF         |
           +--------------------+
```

### 💾 Multi-Layer Storage

| Layer | Storage | Data | Retention |
|-------|---------|------|-----------|
| **Frontend** | LocalStorage | Search history, language prefs, settings | Permanent |
| **Backend** | `app/memory/plans.json` | User task plans | All sessions |
| **Backend** | `app/memory/results.json` | Execution outputs | All sessions |
| **Backend** | `app/memory/logs.json` | Runtime logs & diagnostics | All sessions |
| **Export** | `output/` folder | Timestamped reports (5 formats) | Permanent |

### 🛡️ Resilience & Error Handling

| Capability | Status |
|------------|--------|
| **Tool Error Isolation** | ✅ Agent failure doesn't kill pipeline |
| **Per-Step Error Logging** | ✅ Errors stored individually |
| **Graceful Degradation** | ✅ Partial results returned |
| **Tool Retry Logic** | ✅ 2 retries with exponential backoff |
| **Dependency Graph Execution** | ✅ Skip agent if dependency failed |
| **Alternate Tool Fallback** | ✅ Fallback summarizer & sentiment |
| **Auto Step Regeneration** | ✅ Retry with reduced params |
| **Dynamic Plan Rewriting** | ✅ Recovery on critical failures |
| **Redirect Following** | ✅ Handles 302 redirects gracefully |
| **Direct URL Prioritization** | ✅ Prefers direct links over Google News redirects |

> **💡 Smart Execution:** Feature toggles dynamically build plans — planner only invokes agents you enable, reducing API costs and execution time.

---

## 🎯 Agent Deep-Dive

### 📰 Multi-Source News Intelligence

```
+-----------+    +-----------+    +-----------+
|  Tavily   |    |  GNews    |    | RSS Feeds |
| (Web AI)  |    |  (API)    |    | (Google)  |
+-----+-----+    +-----+-----+    +-----+-----+
      |                |                |
      +----------------+----------------+
                       |
                       v
             +-----------------+
             | Parallel Fetch  |  <-- asyncio.gather()
             | + Failover      |  <-- Quota/Error handling
             +--------+--------+
                      |
                      v
             +-----------------+
             | Deduplication   |  <-- Title similarity + URL hash
             | & Merge         |  <-- Source priority scoring
             +-----------------+
```

**Under the Hood:**
- **Concurrent Pipeline** — All sources queried in parallel via `asyncio`
- **Smart Failover** — Auto-switches sources on quota exhaustion or HTTP errors
- **Deduplication Engine** — Fuzzy title matching + URL normalization

---

### 💭 Sentiment Intelligence — *"Narrative, Not Numbers"*

> 🧮 **OLD:** Word counting → *"60% positive"*  
> 🧠 **NEW:** Market narrative intelligence → *"Bullish momentum with regulatory headwinds"*

```
+-------------------------------------------------------------------------+
|  INTELLIGENCE OUTPUT                                                    |
+-------------------------------------------------------------------------+
|  mood_label     -->  "Strong bullish momentum" / "Risk-off prevailing"   |
|  direction      -->  improving / stable / deteriorating                  |
|  confidence     -->  high / medium / low                                 |
|  market_bias    -->  risk_on / balanced / risk_off                       |
|  risk_level     -->  Regulatory threat scan + crisis detection           |
|  reasoning      -->  "Coverage shows X, momentum suggests Y..."         |
+-------------------------------------------------------------------------+
|  [+] positive_signals  -->  ["M&A activity", "Product launches"...]      |
|  [!] negative_signals  -->  ["Regulatory concerns", "Market decline"...]|
|  [*] emerging_themes   -->  ["Tesla", "Fed", "Nvidia"] (NER-extracted)  |
+-------------------------------------------------------------------------+
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

### 📊 Trend Intelligence — *"Velocity, Not Just Frequency"*

> 📝 **OLD:** Keyword counting → *"OpenAI: 5 mentions"*  
> 🔥 **NEW:** Time-weighted velocity detection → *"🔥 OpenAI GPT rising fast"*

```
+-------------------------------------------------------------------------+
|  TREND INTELLIGENCE                                                     |
+----------------+---------------+---------------+------------------------+
| RISING FAST    | RISING        | STABLE        | FADING                 |
| India Trade    | Market Rally  | Fed Policy    | Old Topic              |
| score: 12.5    | score: 8.2    | score: 5.0    |                        |
+----------------+---------------+---------------+------------------------+
| Time Weighting  |  <2hrs=3x  |  <6hrs=2.5x  |  <24hrs=1.5x            |
| N-Gram Phrases  |  "OpenAI GPT" not "OpenAI" + "GPT"                   |
| Source Weights  |  Tavily=1.5x  |  GNews=1.3x  |  RSS=1.0x             |
+-------------------------------------------------------------------------+
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

### 🛠️ MAS Extension Agents — *"5 Specialists, One Query"*

> 🔧 **The MAS layer extends the core pipeline with 5 specialized agents that run in parallel, orchestrated by the Executor Agent's dependency graph.**

```
+=========================================================================+
|  MULTI-AGENT SYSTEM (MAS)                                               |
+=========================================================================+
|                                                                         |
|  Web Scraper        -->  Full article text + og:image extraction         |
|  Entity Extractor   -->  NER: people, orgs, locations, relations        |
|  Image Analyst      -->  AI vision, type classification, forensics      |
|  Social Monitor     -->  Reddit threads, sentiment per-post             |
|  Research Assistant -->  arXiv papers, GitHub repos, SO threads          |
|                                                                         |
+-------------------------------------------------------------------------+
|  Execution: Parallel via asyncio    |  Auto-retry on failure            |
|  Context: Shared pipeline state     |  Export: All agents included      |
+=========================================================================+
```

| Agent | Input | Output | Fallback |
|-------|-------|--------|----------|
| **🌐 Web Scraper** | Article URLs (direct, not redirects) | Title, text, images, metadata | Skipped if URL blocked |
| **👤 Entity Extractor** | News article list | People, orgs, locations + relationships | Empty entity map |
| **🖼️ Image Analyst** | Scraper images → og:image fallback | AI description, objects, type, relevance, manipulation flags, EXIF | Zero images reported |
| **📱 Social Monitor** | Topic query | Reddit posts, scores, sentiment | Empty social section |
| **📚 Research Assistant** | Topic query | Papers, repos, SO answers | Empty research section |

---

### 🖼️ Image Intelligence — *"See, Don't Just Display"*

> 📷 **OLD:** Basic metadata → *"1200×800 JPEG"*  
> 🧠 **NEW:** AI Vision intelligence → *"Banner photo (1200×675), mood: neutral, relevance: medium, compression anomaly detected"*

```
+=========================================================================+
|  IMAGE INTELLIGENCE                                                     |
+=========================================================================+
|                                                                         |
|  1. Collect    -->  GNews image URLs + og:image fallback (deduped)      |
|  2. Validate   -->  Download + PIL check (reject non-image files)       |
|  3. Classify   -->  Pixel analysis: photo/chart/graphic/screenshot      |
|  4. Analyze    -->  Nova Vision AI or smart local heuristics            |
|  5. Forensics  -->  EXIF, compression, borders, editing software        |
|  6. Aggregate  -->  Cross-image: objects, types, descriptions           |
|                                                                         |
+-------------------------------------------------------------------------+
|  Dedup: URL normalization prevents duplicate analysis                   |
|  Filter: SVGs, favicons, tracking pixels, corrupt files auto-skipped   |
|  Mode: Nova AI (Bedrock) or Local Analysis (zero API cost)              |
+=========================================================================+
```

### 🖼️ Enhanced Image Intelligence (v2.0)
Advanced computer vision pipeline that goes beyond basic metadata:

1.  **AI Scene Understanding**: Uses `Amazon Nova Vision` (or robust local fallback) to generate detailed, context-aware descriptions of image content.
    *   *Example*: "Naturally lit, warm-toned wide banner-format news photo related to: India AI Impact Summit"
2.  **Object Detection**: Identifies key objects (people, technology, text) within the image.
3.  **Manipulation Forensics**: Analyzes compression artifacts, metadata consistency, and unnatural edge detection to flag potential deepfakes or edited images.
4.  **Smart Filtering**:
    *   **Deduplication**: Automatically detects and skips duplicate images across multiple articles.
    *   **Junk Filter**: Ignores icons, tracking pixels, and non-content images.
    *   **Validation**: Verifies image integrity before processing.
5.  **Relevance Scoring**: Scores images based on their semantic similarity to the article topic.

### 📚 Research Library (New!)
A powerful academic and technical research engine that aggregates knowledge from three major sources:

1.  **arXiv Papers**: Fetches recent academic papers with abstract summaries and PDF links.
    *   *Features*: intelligent query encoding, redirection handling, and metadata extraction.
2.  **GitHub Repositories**: Discovers relevant open-source projects.
    *   *Metrics*: Stars, language, description, and direct repo links.
3.  **StackOverflow Discussions**: Finds community Q&A threads related to the topic.
    *   *Details*: Answer status, score, tags, and discussion links.

> **Note**: The Research Library seamlessly handles 301 redirects and complex queries to ensure reliable data retrieval.

---

### 📦 Intelligence Package Builder — *"Not Save. Deliver."*

> 📁 **OLD:** Basic file export → *"Download JSON"*  
> 📦 **NEW:** Intelligence Packaging Center → *Preview → Quality Badge → Smart Format → Multi-Export*

```
+=========================================================================+
|  INTELLIGENCE PACKAGE BUILDER                                           |
+=========================================================================+
|  Quality   |  [*] Full (5+ agents) | [~] Partial (3-4) | [-] Raw (<3)  |
|  Preview   |  [x] News  [x] Summary  [x] Entities  [x] Social  [ ] Img |
|  Stats     |  15 Articles  |  7 Sections  |  ~85 KB                    |
|  Recommend |  JSON for structured intelligence data                     |
+-------------------------------------------------------------------------+
|  [JSON]   [Markdown]   [CSV]   [Word]   [PDF]                          |
|  [Download All 5 Formats]                                               |
|  [Copy JSON to Clipboard]                                               |
+=========================================================================+
```

**🔧 Under the Hood:**

| Feature | What It Does |
|---------|--------------|
| **Quality Badges** | 🟢 Full (5+ sections) → 🟡 Partial (3-4) → 🔴 Raw (<3) |
| **Smart Recommend** | JSON for MAS data, Markdown for narrative, CSV for tabular |
| **Toggle-Aware** | Exports only enabled features — all 9 agents supported |
| **5-Format Export** | JSON, Markdown, CSV, Word (DOCX), PDF |
| **Execution Quality** | Shows agents ran, retries, fallbacks, confidence badge |
| **Copy Clipboard** | Instant JSON copy for API/dev use |

---

### 🌐 Language & Dictionary Tools

> *Understand any term. Read in any language.*

**📖 Dictionary Lookup:**
1. Click **DICT** button in Intelligence Report header
2. Type any English word (e.g., "tariff", "bullish", "volatility")  
3. Press **GO** or hit Enter → Definition popup appears instantly

**🌐 Translation System:**
1. Click **⚙️ Settings** → Select up to **3 languages** from 18 available
2. Use **Translate** dropdown on Intelligence Report
3. Translated text appears with **Show Original** button

**Supported Languages:**
`English` `Hindi` `Spanish` `French` `German` `Chinese` `Japanese` `Korean` `Arabic` `Portuguese` `Russian` `Italian` `Tamil` `Telugu` `Bengali` `Marathi` `Gujarati` `Punjabi`

**🎯 Smart Tooltips** — Finance terms highlighted with hover explanations:

| Term | Explanation |
|------|-------------|
| **Momentum** | How fast and strong sentiment is changing |
| **Risk-On** | Investors favor risky assets (stocks, crypto) |
| **Risk-Off** | Investors prefer safe assets (bonds, gold) |
| **Bullish Signals** | Factors driving positive sentiment |
| **Confidence** | Certainty level based on data consistency |

---

##  Quick Start

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

## 🔌 MCP Server Integration

**Nova supports Model Context Protocol (MCP)** — connect the MAS to AI assistants like Claude!

```
"Use Nova to fetch the latest Tesla news and analyze sentiment"
```

The AI will automatically invoke Nova's 9 agents, fetch multi-source news, run entity extraction, analyze images, monitor social media, and deliver a comprehensive intelligence report.

### Quick MCP Setup

1. **Install:** `pip install mcp>=0.9.0`
2. **Test:** `python test_mcp.py`
3. **Configure:** Use `mcp_config.json` with Claude Desktop — see [MCP_README.md](MCP_README.md)

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `fetch_news` | Multi-source news fetching |
| `analyze_sentiment` | Institutional-grade sentiment analysis |
| `summarize_news` | AI-powered summaries |
| `extract_trends` | Trending topic detection |
| `intelligence_query` | Full 9-agent pipeline (recommended) |
| `get_history` | Access past queries |

📚 **Full MCP Documentation:** See [MCP_README.md](MCP_README.md)

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

**Core:**  📰 News • 🧠 Summary • 💭 Sentiment • 📊 Trends  
**MAS:**  🌐 Scraper • 👤 Entities • 🖼️ Images • 📱 Social • 📚 Research  
**Export:** 📦 Package Builder (JSON, MD, CSV, Word, PDF)

> ⚡ **Select All** for full 9-agent analysis or pick individual agents to customize your intelligence report.

---

## 📁 Project Structure

```
NovaAI/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI entry point
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner_agent.py         # Nova AI task planning
│   │   └── executor_agent.py        # Agent orchestration & context
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                # REST API endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tool_registry.py         # Agent management
│   │   └── plan_validator.py        # Plan validation
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── store.py                 # Logging & persistence
│   │   ├── logs.json                # Execution logs
│   │   ├── plans.json               # Saved plans
│   │   └── results.json             # Execution results
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py               # Pydantic models
│   │
│   └── tools/                       # 🛠️ 9 Intelligence Agents
│       ├── __init__.py
│       ├── multi_fetcher.py         # Parallel multi-source fetch
│       ├── tavily_fetcher.py        # Tavily web search API
│       ├── gnews_fetcher.py         # GNews API
│       ├── rss_fetcher.py           # Google News RSS
│       ├── news_fetcher.py          # Legacy single-source
│       ├── summarizer.py            # Nova AI summary
│       ├── sentiment.py             # Nova AI sentiment
│       ├── trends.py                # Topic extraction & velocity
│       ├── exporter.py              # 5-format export engine
│       ├── web_scraper.py           # 🌐 Full article scraper
│       ├── entity_extractor.py      # 👤 NER extraction
│       ├── image_analyzer.py        # 🖼️ Image forensics
│       ├── social_monitor.py        # 📱 Reddit/Twitter monitor
│       └── research_assistant.py    # 📚 arXiv/GitHub/SO search
│
├── frontend/
│   ├── index.html                   # Main UI + Settings + Package Builder
│   ├── app.js                       # Frontend logic (1800+ lines)
│   ├── style.css                    # Professional dark theme
│   └── mas_styles.css               # MAS agent panel styles
│
├── output/                          # Exported report files
├── .env                             # API keys (not in git)
├── .env.example                     # Template for .env
├── .gitignore
├── requirements.txt
├── mcp_config.json                  # MCP server configuration
├── MCP_README.md                    # MCP documentation
└── README.md
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/command` | Process voice/text command (triggers full agent pipeline) |
| POST | `/api/export` | Export intelligence report (json/markdown/csv/docx/pdf) |
| GET | `/api/capabilities` | Get available agents & system status |
| GET | `/api/history` | Get recent command history |
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
  "news": [...],
  "summary": "The India AI summit highlighted national AI strategy...",
  "sentiment": {
    "overall": "positive",
    "mood_label": "Strong bullish momentum",
    "direction": "improving",
    "confidence": "high",
    "market_bias": "risk_on",
    "score": 0.75,
    "positive_signals": ["Product launches", "Strategic partnerships"],
    "emerging_themes": ["India", "OpenAI", "Trump"]
  },
  "trends": {
    "trending_topics": [
      { "topic": "India", "score": 50.38, "velocity": "stable", "story_direction": "Strong Coverage" }
    ]
  },
  "entities": {
    "people": [{ "name": "Modi", "title": "PM" }],
    "organizations": [{ "name": "OpenAI", "mentions": 3 }],
    "locations": [{ "name": "India", "mentions": 12 }]
  },
  "social": {
    "reddit": { "posts": [...], "sentiment": { "positive": 45, "neutral": 40, "negative": 15 } }
  },
  "images": {
    "total_images": 3,
    "successful": 2,
    "detailed_results": [...]
  }
}
```

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

Built with ❤️ using Amazon Nova AI  
**MUKUL PRASAD**
