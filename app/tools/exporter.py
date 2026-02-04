"""Multi-format data exporter V2 - JSON, Markdown, CSV with Intelligence format."""
import json
import os
import csv
from datetime import datetime
from typing import Dict, Any


OUTPUT_DIR = "output"


def export_data(
    data: Dict[str, Any],
    filename: str = "report",
    format: str = "json"
) -> str:
    """
    Export data to specified format.
    
    Args:
        data: Dict with news, summary, sentiment, trends
        filename: Base filename (without extension)
        format: 'json', 'markdown', or 'csv'
    
    Returns:
        Path to saved file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "markdown":
        return _export_markdown(data, filename, timestamp)
    elif format == "csv":
        return _export_csv(data, filename, timestamp)
    else:
        return _export_json(data, filename, timestamp)


def _export_json(data: Dict, filename: str, timestamp: str) -> str:
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.json"
    output = {
        "generated_at": datetime.now().isoformat(),
        "report_type": "Nova Intelligence Report",
        **data
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return path


def _export_markdown(data: Dict, filename: str, timestamp: str) -> str:
    """Export as professional intelligence-style Markdown."""
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.md"
    lines = [
        "# 🧠 Nova Intelligence Report",
        f"\n*Generated: {timestamp}*\n",
        "---",
    ]
    
    # Summary Section
    if "summary" in data and data["summary"]:
        s = data["summary"]
        lines.append("\n## 📝 Executive Summary\n")
        summary_text = s.get("summary", str(s)) if isinstance(s, dict) else str(s)
        lines.append(summary_text)
        
        if isinstance(s, dict) and s.get("key_points"):
            lines.append("\n**Key Takeaways:**")
            for point in s.get("key_points", []):
                lines.append(f"- {point}")
        lines.append("")
    
    # Sentiment Intelligence V2
    if "sentiment" in data and data["sentiment"]:
        s = data["sentiment"]
        lines.append("\n## 💭 Sentiment Intelligence\n")
        
        # Mood & Direction
        mood = s.get("mood_label", s.get("overall", "N/A"))
        direction = s.get("direction", "stable")
        direction_icon = "📈" if direction == "improving" else "📉" if direction == "deteriorating" else "➡️"
        lines.append(f"**Mood:** {mood}")
        lines.append(f"**Direction:** {direction_icon} {direction.capitalize()}")
        lines.append("")
        
        # Market Indicators
        momentum = s.get("momentum_strength", "moderate")
        bias = s.get("market_bias", "balanced")
        bias_display = "🟢 Risk-On" if bias == "risk_on" else "🔴 Risk-Off" if bias == "risk_off" else "⚪ Balanced"
        lines.append(f"| Indicator | Value |")
        lines.append(f"|-----------|-------|")
        lines.append(f"| Momentum | {momentum.capitalize()} |")
        lines.append(f"| Market Bias | {bias_display} |")
        lines.append(f"| Confidence | {s.get('confidence', 'medium').capitalize()} |")
        lines.append(f"| Risk Level | {s.get('risk_level', 'low').capitalize()} |")
        lines.append("")
        
        # Analyst Reasoning
        if s.get("reasoning"):
            lines.append(f"**Analyst View:** {s.get('reasoning')}")
            lines.append("")
        
        # Signals
        if s.get("positive_signals"):
            lines.append("**✅ Bullish Signals:**")
            for sig in s.get("positive_signals", []):
                lines.append(f"- {sig}")
            lines.append("")
        
        if s.get("negative_signals"):
            lines.append("**⚠️ Risk Signals:**")
            for sig in s.get("negative_signals", []):
                lines.append(f"- {sig}")
            lines.append("")
        
        # Emerging Themes
        if s.get("emerging_themes"):
            themes = ", ".join(s.get("emerging_themes", []))
            lines.append(f"**🔥 Emerging Themes:** {themes}")
            lines.append("")
    
    # Trends Section
    if "trends" in data and data["trends"]:
        t = data["trends"]
        if t.get("trending_topics"):
            lines.append("\n## 📊 Trending Topics\n")
            lines.append("| Topic | Mentions |")
            lines.append("|-------|----------|")
            for topic in t.get("trending_topics", [])[:8]:
                lines.append(f"| {topic.get('topic')} | {topic.get('mentions')} |")
            lines.append("")
    
    # Articles Section
    if "news" in data and data["news"]:
        lines.append("\n## 📰 Source Articles\n")
        for i, item in enumerate(data["news"][:10], 1):
            source = item.get('source', 'unknown').upper()
            lines.append(f"{i}. [{item.get('title', 'No title')}]({item.get('link', '#')}) `{source}`")
        lines.append("")
    
    # Footer
    lines.append("\n---")
    lines.append("*Report generated by Nova Intelligence Agent*")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def _export_csv(data: Dict, filename: str, timestamp: str) -> str:
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.csv"
    news = data.get("news", [{"title": "No data", "link": "", "source": ""}])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "link", "source", "published"], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(news)
    return path
