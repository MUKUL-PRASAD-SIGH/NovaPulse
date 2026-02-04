"""Multi-format data exporter - JSON, Markdown, CSV."""
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
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.md"
    lines = [
        "# Nova Intelligence Report",
        f"\n*Generated: {timestamp}*\n",
    ]
    if "summary" in data and data["summary"]:
        s = data["summary"]
        lines.append("## Summary")
        lines.append(s.get("summary", str(s)) if isinstance(s, dict) else str(s))
        lines.append("")
    if "sentiment" in data and data["sentiment"]:
        s = data["sentiment"]
        lines.append(f"## Sentiment: {s.get('overall', 'N/A')} ({s.get('score', 0):.0%})")
        lines.append("")
    if "news" in data and data["news"]:
        lines.append("## Articles")
        for item in data["news"]:
            lines.append(f"- [{item.get('title', 'No title')}]({item.get('link', '#')})")
        lines.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def _export_csv(data: Dict, filename: str, timestamp: str) -> str:
    path = f"{OUTPUT_DIR}/{filename}_{timestamp}.csv"
    news = data.get("news", [{"title": "No data", "link": "", "source": ""}])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["title", "link", "source"], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(news)
    return path
