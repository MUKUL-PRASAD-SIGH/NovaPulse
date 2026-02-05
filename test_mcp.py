"""Test script for MCP server.

Run this to verify the MCP server works correctly.
"""
import asyncio
import json
from mcp_server import list_tools, call_tool


async def test_mcp_server():
    """Test MCP server functionality."""
    
    print("🧪 Testing NovaAI MCP Server\n")
    
    # Test 1: List tools
    print("1️⃣ Testing list_tools()...")
    tools = await list_tools()
    print(f"✅ Found {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool.name}: {tool.description[:60]}...")
    print()
    
    # Test 2: Simple news fetch
    print("2️⃣ Testing fetch_news tool...")
    try:
        result = await call_tool("fetch_news", {
            "topic": "AI",
            "limit": 3,
            "sources": ["rss"]  # Use RSS as it doesn't require API key
        })
        print("✅ News fetch successful!")
        data = json.loads(result[0].text)
        if "articles" in data:
            print(f"   Found {len(data.get('articles', []))} articles")
        print()
    except Exception as e:
        print(f"⚠️  News fetch failed (expected if no API keys): {e}\n")
    
    # Test 3: Intelligence query (full pipeline)
    print("3️⃣ Testing intelligence_query tool...")
    try:
        result = await call_tool("intelligence_query", {
            "query": "latest AI news",
            "include_summary": False,  # Disable to avoid AWS calls
            "include_sentiment": False,
            "include_trends": False
        })
        print("✅ Intelligence query successful!")
        data = json.loads(result[0].text)
        print(f"   Query: {data.get('query')}")
        print(f"   Plan steps: {len(data.get('plan', {}).get('steps', []))}")
        print()
    except Exception as e:
        print(f"⚠️  Intelligence query failed: {e}\n")
    
    # Test 4: Get history
    print("4️⃣ Testing get_history tool...")
    try:
        result = await call_tool("get_history", {"limit": 3})
        print("✅ History retrieval successful!")
        data = json.loads(result[0].text)
        print(f"   Recent plans: {len(data.get('recent_plans', []))}")
        print(f"   Recent results: {len(data.get('recent_results', []))}")
        print()
    except Exception as e:
        print(f"⚠️  History retrieval failed: {e}\n")
    
    print("🎉 MCP Server tests complete!")
    print("\n📝 Next steps:")
    print("   1. Add API keys to .env (AWS, Tavily, GNews)")
    print("   2. Configure MCP client (Claude Desktop, etc.)")
    print("   3. Use mcp_config.json for client configuration")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
