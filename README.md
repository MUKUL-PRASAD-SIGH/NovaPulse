# 🧠 Nova Intelligence Agent

> **Not just news. Intelligence.**

Voice-powered multi-agent news intelligence system using Amazon Nova.

## ⚡ Features

- 📰 **Multi-Source News** - Tavily, GNews, RSS in parallel
- 🔄 **Auto-Failover** - If one source fails, others continue
- 🧠 **AI Summary** - Nova-powered digests
- 💭 **Sentiment Analysis** - Tone detection
- 📊 **Trend Extraction** - Hot topics
- 💾 **Multi-Format Export** - JSON, Markdown, CSV
- 🎤 **Voice Interface** - Browser speech recognition

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy environment file
copy .env.example .env

# 3. Add your API keys to .env:
#    - AWS_ACCESS_KEY_ID (for Nova)
#    - TAVILY_API_KEY (from tavily.com)
#    - GNEWS_API_KEY (from gnews.io)

# 4. Run server
uvicorn app.main:app --reload --port 8000

# 5. Open browser
# http://localhost:8000
```

## 🎤 Voice Commands

```
"Get AI news"
"Stock market news"
"India US trade deal"
"Tesla news with sentiment"
```

## 🏗 Architecture

```
User Prompt → Nova Planner → Parallel Fetcher
                                   ├── Tavily (web search)
                                   ├── GNews (news API)  
                                   └── RSS (free feeds)
                                          ↓
                              Merger → Deduplicator → Output
```

## 📁 Structure

```
NovaAI/
├── app/
│   ├── agents/          # Planner + Executor
│   ├── tools/           # Multi-source fetchers
│   │   ├── tavily_fetcher.py
│   │   ├── gnews_fetcher.py
│   │   ├── rss_fetcher.py
│   │   └── multi_fetcher.py
│   ├── core/            # Registry + Validator
│   ├── memory/          # Persistence
│   ├── api/             # FastAPI routes
│   └── main.py
├── frontend/            # Voice UI
└── output/              # Exported files
```

## 📡 API Endpoints

- `POST /api/command` - Process voice/text command
- `GET /api/capabilities` - Get agent features
- `GET /api/history` - Get recent commands
