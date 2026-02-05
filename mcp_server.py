"""MCP Server for Nova Intelligence Agent.

Exposes NovaAI capabilities as MCP tools for AI assistants.
"""
import asyncio
import json
from typing import Any, Dict, List
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource, ResourceTemplate
from mcp.server.stdio import stdio_server
from dotenv import load_dotenv

# Import NovaAI components
from app.agents.planner_agent import plan_task
from app.agents.executor_agent import execute_plan
from app.tools.multi_fetcher import fetch_news_multi_source
from app.tools.summarizer import summarize_news
from app.tools.sentiment import analyze_sentiment
from app.tools.trends import extract_trends
from app.memory.store import get_recent_plans, get_recent_results

load_dotenv()

# Create MCP server instance
app = Server("nova-intelligence")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available NovaAI tools."""
    return [
        Tool(
            name="fetch_news",
            description="Fetch latest news articles from multiple sources (Tavily, GNews, RSS) on any topic. Returns deduplicated articles with titles, URLs, summaries, and source information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for (e.g., 'Tesla', 'AI', 'India trade deal')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of articles to fetch (default: 10)",
                        "default": 10
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["tavily", "gnews", "rss"]},
                        "description": "Which sources to use (default: all available)",
                        "default": ["tavily", "gnews", "rss"]
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="analyze_sentiment",
            description="Analyze sentiment and market intelligence from news articles. Returns institutional analyst-style narrative with mood, direction, confidence, market bias, risk level, and reasoning. Includes positive/negative signals and emerging themes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "summary": {"type": "string"}
                            }
                        },
                        "description": "List of articles to analyze (with title and summary)"
                    }
                },
                "required": ["articles"]
            }
        ),
        Tool(
            name="summarize_news",
            description="Generate AI-powered executive summary of news articles using Amazon Nova. Returns 2-3 sentence digest of key developments.",
            inputSchema={
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "summary": {"type": "string"}
                            }
                        },
                        "description": "List of articles to summarize (with title and summary)"
                    }
                },
                "required": ["articles"]
            }
        ),
        Tool(
            name="extract_trends",
            description="Extract trending topics and themes from news articles using time-weighted velocity detection. Returns rising/stable/fading topics with scores, story direction, news cycle stage, and AI-powered explanations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "published": {"type": "string"}
                            }
                        },
                        "description": "List of articles to analyze for trends"
                    }
                },
                "required": ["articles"]
            }
        ),
        Tool(
            name="intelligence_query",
            description="Full intelligence pipeline: automatically fetches news, analyzes sentiment, generates summary, and extracts trends for any topic. This is the main entry point - use this for comprehensive intelligence reports.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query (e.g., 'Tesla news with sentiment', 'AI trends', 'India US trade analysis')"
                    },
                    "include_summary": {
                        "type": "boolean",
                        "description": "Include AI summary (default: true)",
                        "default": True
                    },
                    "include_sentiment": {
                        "type": "boolean",
                        "description": "Include sentiment analysis (default: true)",
                        "default": True
                    },
                    "include_trends": {
                        "type": "boolean",
                        "description": "Include trend extraction (default: true)",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_history",
            description="Retrieve recent intelligence queries and results from memory store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent items to retrieve (default: 5)",
                        "default": 5
                    }
                }
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Execute NovaAI tools."""
    
    if name == "fetch_news":
        topic = arguments["topic"]
        limit = arguments.get("limit", 10)
        sources = arguments.get("sources", ["tavily", "gnews", "rss"])
        
        result = await fetch_news_multi_source(
            topic=topic,
            sources=sources,
            limit=limit
        )
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "analyze_sentiment":
        articles = arguments["articles"]
        result = await analyze_sentiment({"articles": articles})
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "summarize_news":
        articles = arguments["articles"]
        result = await summarize_news({"articles": articles})
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "extract_trends":
        articles = arguments["articles"]
        result = await extract_trends({"articles": articles})
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "intelligence_query":
        query = arguments["query"]
        
        # Use the planner to create a task plan
        plan = plan_task(query)
        
        # Optionally filter steps based on user preferences
        if not arguments.get("include_summary", True):
            plan["steps"] = [s for s in plan["steps"] if s["tool"] != "summarizer"]
        if not arguments.get("include_sentiment", True):
            plan["steps"] = [s for s in plan["steps"] if s["tool"] != "sentiment"]
        if not arguments.get("include_trends", True):
            plan["steps"] = [s for s in plan["steps"] if s["tool"] != "trends"]
        
        # Execute the plan
        result = await execute_plan(plan)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "query": query,
                "plan": plan,
                "result": result
            }, indent=2)
        )]
    
    elif name == "get_history":
        limit = arguments.get("limit", 5)
        
        plans = get_recent_plans(limit)
        results = get_recent_results(limit)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "recent_plans": plans,
                "recent_results": results
            }, indent=2)
        )]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available NovaAI resources."""
    return [
        Resource(
            uri="nova://capabilities",
            name="NovaAI Capabilities",
            mimeType="application/json",
            description="Available tools and their descriptions"
        ),
        Resource(
            uri="nova://history/plans",
            name="Task Plans History",
            mimeType="application/json",
            description="Recent task plans generated by the planner"
        ),
        Resource(
            uri="nova://history/results",
            name="Execution Results History",
            mimeType="application/json",
            description="Recent execution results"
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read NovaAI resources."""
    
    if uri == "nova://capabilities":
        tools = await list_tools()
        return json.dumps({
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools
            ]
        }, indent=2)
    
    elif uri == "nova://history/plans":
        plans = get_recent_plans(10)
        return json.dumps(plans, indent=2)
    
    elif uri == "nova://history/results":
        results = get_recent_results(10)
        return json.dumps(results, indent=2)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
