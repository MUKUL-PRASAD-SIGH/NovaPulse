# 🧠 Nova Intelligence Agent

> **Not just news. Intelligence.**

Voice-powered multi-agent news intelligence system using Amazon Nova.

## ⚡ Features

- 📰 **Multi-Source News** - Google, TechCrunch, Verge
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

# 3. Run server
uvicorn app.main:app --reload --port 8000

# 4. Open browser
# http://localhost:8000
```

## 🎤 Voice Commands

```
"Get top AI news"
"Get crypto news with sentiment analysis"
"What's trending in tech?"
"Summarize AI news and export as markdown"
```

## 📁 Structure

```
NovaAI/
├── app/
│   ├── agents/          # Planner + Executor
│   ├── tools/           # 5 intelligence tools
│   ├── core/            # Registry + Validator
│   ├── memory/          # Persistence
│   ├── api/             # FastAPI routes
│   └── main.py
├── frontend/            # Voice UI
└── output/              # Exported files
```

## 🔧 Configuration

Set `USE_MOCK_PLANNER=true` in `.env` to avoid API costs during development.

## 📡 API Endpoints

- `POST /api/command` - Process voice/text command
- `GET /api/capabilities` - Get agent features
- `GET /api/history` - Get recent commands
- `GET /api/health` - Health check
