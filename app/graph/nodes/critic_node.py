"""Nova v3 — Critic Agent (Quality Gate).

Evaluates the fused intelligence report for quality and completeness.
If the score is below threshold AND retries remain, triggers a retry.

Scoring dimensions:
1. Completeness  (0-25) — how many sections are present
2. Source diversity (0-25) — how many unique sources contributed  
3. Depth          (0-25) — richness of summary/analysis text
4. Consistency    (0-25) — absence of contradictions

Total: 0-100. Threshold: 60 to pass.
"""

import time
from typing import Dict, Any, List

from app.memory.store import log


def _count_sections(report: Dict) -> int:
    """Count how many meaningful sections the report has."""
    sections = 0
    if report.get("news"):
        sections += 1
    if report.get("summary"):
        sections += 1
    if report.get("sentiment"):
        sections += 1
    if report.get("trends"):
        sections += 1
    if report.get("entities"):
        sections += 1
    if report.get("social"):
        sections += 1
    if report.get("research"):
        sections += 1
    if report.get("images"):
        sections += 1
    if report.get("market"):
        sections += 1
    if report.get("scraped_articles"):
        sections += 1
    return sections


def _count_news_sources(report: Dict) -> int:
    """Count unique news source domains."""
    news = report.get("news", [])
    sources = set()
    for article in news:
        source = article.get("source", article.get("source_id", ""))
        if source:
            sources.add(source)
    return len(sources)


def _measure_depth(report: Dict) -> int:
    """Measure depth via word count of key text fields."""
    word_count = 0

    summary = report.get("summary")
    if isinstance(summary, dict):
        text = summary.get("summary", "") or summary.get("text", "")
        word_count += len(str(text).split())
    elif isinstance(summary, str):
        word_count += len(summary.split())

    sentiment = report.get("sentiment")
    if isinstance(sentiment, dict):
        reasoning = sentiment.get("reasoning", "")
        word_count += len(str(reasoning).split())

    return word_count


def _detect_contradictions(report: Dict) -> List[str]:
    """Detect obvious contradictions between sections."""
    contradictions = []

    sentiment = report.get("sentiment", {})
    social = report.get("social", {})

    if isinstance(sentiment, dict) and isinstance(social, dict):
        news_sentiment = sentiment.get("overall", "").lower()
        social_agg = social.get("aggregate", {})
        social_sentiment = social_agg.get("overall_sentiment", "").lower()

        if news_sentiment and social_sentiment:
            # Detect strong disagreement
            positive = {"positive", "bullish", "optimistic"}
            negative = {"negative", "bearish", "pessimistic"}

            if (news_sentiment in positive and social_sentiment in negative) or \
               (news_sentiment in negative and social_sentiment in positive):
                contradictions.append(
                    f"News sentiment ({news_sentiment}) contradicts social sentiment ({social_sentiment})"
                )

    return contradictions


async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate intelligence quality. Returns approval or triggers retry.
    
    Reads:  fused_report, enable_critic, retry_count, max_retries
    Writes: critic_score, critic_feedback, critic_approved, retry_count, execution_trace
    """
    start = time.time()

    # If critic is disabled, auto-approve
    if not state.get("enable_critic", False):
        return {
            "critic_score": 100,
            "critic_feedback": [],
            "critic_approved": True,
            "execution_trace": state.get("execution_trace", []) + [{
                "node": "critic",
                "status": "skipped_auto_approve",
                "duration_ms": 0,
            }],
        }

    report = state.get("fused_report", {})
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    score = 0
    feedback = []

    # === 1. COMPLETENESS (0-25) ===
    sections = _count_sections(report)
    completeness = min(25, sections * 5)
    score += completeness
    if sections < 3:
        feedback.append(f"Low completeness: only {sections} sections (need ≥3)")

    # === 2. SOURCE DIVERSITY (0-25) ===
    sources = _count_news_sources(report)
    diversity = min(25, sources * 8)
    score += diversity
    if sources < 2:
        feedback.append(f"Low source diversity: only {sources} source(s)")

    # === 3. DEPTH (0-25) ===
    word_count = _measure_depth(report)
    depth = min(25, word_count // 4)
    score += depth
    if word_count < 30:
        feedback.append(f"Summary too shallow: only {word_count} words")

    # === 4. CONSISTENCY (0-25) ===
    contradictions = _detect_contradictions(report)
    consistency = max(0, 25 - len(contradictions) * 12)
    score += consistency
    for c in contradictions:
        feedback.append(f"Contradiction: {c}")

    # === DECISION ===
    approved = score >= 60 or retry_count >= max_retries

    duration = round((time.time() - start) * 1000)

    trace_entry = {
        "node": "critic",
        "status": "approved" if approved else "retry",
        "duration_ms": duration,
        "score": score,
        "breakdown": {
            "completeness": completeness,
            "diversity": diversity,
            "depth": depth,
            "consistency": consistency,
        },
        "feedback": feedback,
    }

    log_level = "INFO" if approved else "WARN"
    log(log_level, f"Critic: score={score}/100, approved={approved}, feedback={feedback}")

    result = {
        "critic_score": score,
        "critic_feedback": feedback,
        "critic_approved": approved,
        "execution_trace": state.get("execution_trace", []) + [trace_entry],
    }

    if not approved:
        result["retry_count"] = retry_count + 1

    return result
