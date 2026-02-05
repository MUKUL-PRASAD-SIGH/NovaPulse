# MCP Server Architecture Diagram

## Current Architecture (Standalone NovaAI)

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER                                    │
│                    (Voice/Text Input)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Web Frontend                                 │
│                  (index.html + app.js)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST /api/command
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Server                                │
│                    (app/main.py)                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Planner Agent                                  │
│              (Amazon Nova AI Planning)                          │
│         Converts query → Task Plan JSON                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Executor Agent                                 │
│         Runs tools in sequence with context                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ News Fetcher │    │ Summarizer   │    │  Sentiment   │
│ (Multi-src)  │    │ (Nova AI)    │    │  Analyzer    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌──────────────┐
                    │   Results    │
                    │   (JSON)     │
                    └──────────────┘
```

## New MCP Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Assistant                                 │
│              (Claude Desktop / ChatGPT)                         │
│                                                                 │
│  User: "Use NovaAI to analyze Tesla news sentiment"            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ MCP Protocol (stdio)
                            │ Tool Call: intelligence_query
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Server                                   │
│                  (mcp_server.py)                                │
│                                                                 │
│  @app.list_tools()    → Returns available tools                │
│  @app.call_tool()     → Executes NovaAI functions              │
│  @app.list_resources() → Returns data sources                  │
│  @app.read_resource()  → Reads history/capabilities            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Direct Python imports
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ plan_task()  │    │execute_plan()│    │ Tool imports │
│ (Planner)    │    │ (Executor)   │    │ (Direct)     │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                   ┌──────────────┐
                   │  NovaAI Core │
                   │    Tools     │
                   └──────┬───────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Multi-Source │  │ Nova AI      │  │  Sentiment   │
│ News Fetch   │  │ Summarizer   │  │  Analysis    │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                  ┌──────────────┐
                  │   Results    │
                  │   (JSON)     │
                  └──────┬───────┘
                         │ MCP Response
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI Assistant                                 │
│                                                                 │
│  "Here's the Tesla sentiment analysis from NovaAI:             │
│   - Overall: Bullish momentum                                  │
│   - Confidence: High                                            │
│   - Key signals: Product launches, market optimism..."         │
└─────────────────────────────────────────────────────────────────┘
```

## MCP Tool Flow Example

```
USER (in Claude):
  "Use NovaAI to get Tesla news with sentiment"

CLAUDE:
  [Calls MCP tool: intelligence_query]
  {
    "query": "Tesla news with sentiment",
    "include_summary": true,
    "include_sentiment": true,
    "include_trends": false
  }

MCP SERVER:
  1. Receives tool call via stdio
  2. Calls plan_task("Tesla news with sentiment")
  3. Planner creates task plan:
     {
       "intent": "Get Tesla news with sentiment",
       "steps": [
         {"tool": "news_fetcher", "params": {"topic": "Tesla"}},
         {"tool": "sentiment", "params": {}},
         {"tool": "summarizer", "params": {}}
       ]
     }
  4. Calls execute_plan(plan)
  5. Executor runs tools in sequence:
     - news_fetcher → Fetches from Tavily, GNews, RSS
     - sentiment → Analyzes with Nova AI
     - summarizer → Creates executive summary
  6. Returns JSON result via MCP

CLAUDE:
  [Receives result, formats for user]
  "I've analyzed Tesla news using NovaAI. Here's what I found:
   
   📰 Found 15 articles from multiple sources
   
   💭 Sentiment Analysis:
   - Overall mood: Strong bullish momentum
   - Direction: Improving ↗
   - Confidence: High ●
   - Market bias: Risk-on 🟢
   
   🧠 Summary:
   Tesla stock surges on strong Q4 deliveries and new product 
   announcements. Analysts upgrade price targets citing improved
   production efficiency and expanding market share..."
```

## Benefits of MCP Integration

1. **Natural Language Interface**
   - No need to learn NovaAI's web UI
   - Just ask the AI assistant naturally

2. **Composability**
   - AI can combine NovaAI with other MCP tools
   - Example: "Get Tesla news from NovaAI and create a presentation"

3. **Context Awareness**
   - AI assistant maintains conversation context
   - Can refine queries based on previous results

4. **Universal Access**
   - Works with any MCP-compatible AI
   - Same tools, different interfaces

5. **Automation Potential**
   - AI can schedule regular intelligence reports
   - Can trigger actions based on sentiment changes

## 🔌 Connecting to AI Assistants

### For Claude Desktop

Add to your Claude config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "nova-intelligence": {
      "command": "python",
      "args": [
        "c:\\Users\\Mukul Prasad\\Desktop\\PROJECTS\\NovaAI\\mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "c:\\Users\\Mukul Prasad\\Desktop\\PROJECTS\\NovaAI"
      }
    }
  }
}
```

### For Other MCP Clients

Use the provided `mcp_config.json` as a reference. The server runs in stdio mode and follows the MCP specification.

## 💡 Example Usage in AI Assistant

Once connected, you can ask the AI assistant:

```
"Use NovaAI to fetch the latest Tesla news and analyze sentiment"
```

The AI will:
1. Call the `intelligence_query` tool
2. NovaAI fetches news from multiple sources
3. Analyzes sentiment using Amazon Nova
4. Returns comprehensive intelligence report
5. AI formats and presents it to you

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  AI Assistant (Claude/ChatGPT/etc)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │ MCP Protocol
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  MCP Server (mcp_server.py)                                 │
│  - list_tools()      → Available capabilities               │
│  - call_tool()       → Execute NovaAI functions             │
│  - list_resources()  → Available data sources               │
│  - read_resource()   → Read history/capabilities            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  NovaAI Backend                                             │
│  ├── Planner Agent (plan_task)                              │
│  ├── Executor Agent (execute_plan)                          │
│  └── Tools                                                  │
│      ├── Multi-source news fetcher                          │
│      ├── Nova AI summarizer                                 │
│      ├── Sentiment analyzer                                 │
│      └── Trend extractor                                    │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Benefits of MCP Integration

1. **Universal Access** - Any MCP-compatible AI can use NovaAI
2. **Natural Language** - Users describe what they want, AI handles the tool calls
3. **Composability** - AI can combine NovaAI with other MCP tools
4. **No UI Needed** - Direct integration into chat interfaces
5. **Standardized** - Works across different AI platforms

## 🔐 Security Notes

- The MCP server runs locally and uses your existing `.env` credentials
- No data is sent to MCP protocol itself - it's just a communication layer
- API keys remain secure in your environment


