"""Nova v3 — Unified Memory Manager.

Single facade that coordinates short-term, long-term, and semantic
memory layers. All graph nodes interact with memory through this
manager — never directly with individual memory stores.

Blueprint Section 6: Three-Layer Memory Architecture.
"""

import time
import json
from typing import Dict, List, Any, Optional

from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.memory.semantic import SemanticMemory
from app.memory.store import save_result, log


class MemoryManager:
    """Unified memory interface for Nova v3.
    
    Three layers:
      1. Short-term: Session-scoped, in-memory, fast
      2. Long-term:  SQLite persistent, survives restarts
      3. Semantic:   Embedding-based similarity (Phase 4)
    """

    def __init__(self):
        self.short_term = ShortTermMemory(max_entries=200, ttl_seconds=3600)
        self.long_term = LongTermMemory(max_entries=500)
        self.semantic = SemanticMemory()

    # ── STORE ──────────────────────────────────────────────────

    def store(self, query: str, result: Dict, depth: str = "standard",
              pipelines: List[str] = None, critic_score: int = 0,
              confidence: float = 0.0, entities: List[Dict] = None,
              duration_ms: int = 0):
        """Store a completed query result across all memory layers."""

        # Layer 1: Short-term (fast lookup)
        self.short_term.store(query, {
            "result": result,
            "depth": depth,
            "critic_score": critic_score,
            "confidence": confidence,
        })

        # Layer 2: Long-term (persistent)
        try:
            self.long_term.store_query(
                query=query,
                result=result,
                depth=depth,
                pipelines=pipelines or [],
                critic_score=critic_score,
                confidence=confidence,
                entities=entities or [],
                duration_ms=duration_ms,
            )
        except Exception as e:
            log("WARN", f"Long-term memory store failed: {e}")

        # Layer 2b: Entity timeline
        if entities:
            try:
                self.long_term.record_entities(entities, source_query=query)
            except Exception as e:
                log("WARN", f"Entity timeline store failed: {e}")

        # Layer 3: Semantic (if available)
        if self.semantic.is_available:
            try:
                doc_id = f"query_{int(time.time()*1000)}"
                text = f"{query}\n{json.dumps(result.get('summary', ''), default=str)}"
                self.semantic.store_embedding(doc_id, text, {
                    "query": query,
                    "critic_score": critic_score,
                })
            except Exception as e:
                log("WARN", f"Semantic memory store failed: {e}")

        # Legacy compatibility: still save to JSON store
        try:
            save_result({
                "query": query,
                "critic_score": critic_score,
                "data": result,
            })
        except Exception:
            pass

    # ── RECALL ─────────────────────────────────────────────────

    def find_similar(self, query: str, max_age_hours: int = 1) -> Optional[Dict]:
        """Find a similar recent result. Checks short-term first, then long-term.
        
        Returns:
            Dict with keys: result, age_hours, source, critic_score
            or None if nothing found.
        """
        # Layer 1: Check short-term (instant, in-memory)
        stm_key = self.short_term.find_similar_key(query)
        if stm_key:
            data = self.short_term.recall(stm_key)
            if data:
                return {
                    "result": data.get("result", {}),
                    "age_hours": 0,  # Session-level
                    "source": "short_term",
                    "critic_score": data.get("critic_score", 0),
                    "confidence": data.get("confidence", 0.0),
                }

        # Layer 2: Check long-term (SQLite)
        try:
            ltm = self.long_term.find_similar(query, max_age_hours=max_age_hours)
            if ltm:
                age_hours = self.long_term._hours_between(
                    ltm.get("created_at", ""),
                    time.strftime("%Y-%m-%dT%H:%M:%S")
                )
                return {
                    "result": json.loads(ltm.get("result_json", "{}")),
                    "age_hours": age_hours,
                    "source": "long_term",
                    "critic_score": ltm.get("critic_score", 0),
                    "confidence": ltm.get("confidence", 0.0),
                }
        except Exception as e:
            log("WARN", f"Long-term memory recall failed: {e}")

        return None

    # ── COMPARE ────────────────────────────────────────────────

    def compare_topic(self, topic: str, days: int = 7) -> Dict[str, Any]:
        """Compare a topic's intelligence over time.
        
        Returns sentiment direction, score evolution, and key changes.
        Used by fusion node for "How has X changed?" type queries.
        """
        return self.long_term.compare_over_time(topic, days=days)

    def get_entity_history(self, entity: str, limit: int = 20) -> List[Dict]:
        """Get timeline of entity appearances."""
        return self.long_term.get_entity_timeline(entity, limit=limit)

    # ── CHANGE DETECTION (Continuous Mode) ─────────────────────

    def detect_changes(self, topic: str, current_result: Dict) -> Dict[str, Any]:
        """Compare current result with the most recent stored result.
        
        Returns change summary for WebSocket notifications.
        """
        past = self.find_similar(topic, max_age_hours=24)

        if not past:
            return {
                "significant": False,
                "reason": "no_prior_data",
                "changes": [],
            }

        past_result = past.get("result", {})
        changes = []

        # Sentiment shift
        current_sentiment = self._get_sentiment_score(current_result)
        past_sentiment = self._get_sentiment_score(past_result)
        sentiment_delta = abs(current_sentiment - past_sentiment)

        if sentiment_delta > 0.2:
            changes.append({
                "type": "sentiment_shift",
                "from": round(past_sentiment, 2),
                "to": round(current_sentiment, 2),
                "delta": round(sentiment_delta, 2),
            })

        # New articles (by URL)
        current_urls = self._get_article_urls(current_result)
        past_urls = self._get_article_urls(past_result)
        new_urls = current_urls - past_urls

        if new_urls:
            changes.append({
                "type": "new_articles",
                "count": len(new_urls),
            })

        # New entities
        current_ents = self._get_entity_names(current_result)
        past_ents = self._get_entity_names(past_result)
        new_ents = current_ents - past_ents

        if new_ents:
            changes.append({
                "type": "new_entities",
                "entities": list(new_ents)[:10],
            })

        significant = len(changes) > 0 and (
            sentiment_delta > 0.2 or len(new_urls) >= 3 or len(new_ents) >= 2
        )

        return {
            "significant": significant,
            "reason": "changes_detected" if significant else "no_significant_changes",
            "changes": changes,
            "compared_to": {
                "age_hours": past.get("age_hours", 0),
                "source": past.get("source", "unknown"),
            },
        }

    # ── RECORD TRENDS ──────────────────────────────────────────

    def record_trends(self, trends: List[Dict]):
        """Snapshot trending topics for change detection."""
        try:
            self.long_term.record_trends(trends)
        except Exception as e:
            log("WARN", f"Trend recording failed: {e}")

    def detect_trend_changes(self, topic: str, hours: int = 24) -> Dict:
        """Check if a topic's trend score has changed."""
        return self.long_term.detect_trend_changes(topic, hours)

    # ── STATS ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get unified memory statistics."""
        return {
            "short_term": {
                "entries": len(self.short_term),
                "session_context": self.short_term.get_session_context(),
            },
            "long_term": self.long_term.get_stats(),
            "semantic": self.semantic.get_stats(),
        }

    # ── INTERNAL HELPERS ───────────────────────────────────────

    @staticmethod
    def _get_sentiment_score(result: Dict) -> float:
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                return 0.5
        data = result.get("data", result)
        sentiment = data.get("sentiment", {})
        if isinstance(sentiment, dict):
            return sentiment.get("score", 0.5)
        return 0.5

    @staticmethod
    def _get_article_urls(result: Dict) -> set:
        data = result.get("data", result)
        articles = data.get("news", data.get("articles", []))
        if isinstance(articles, list):
            return {a.get("url", a.get("link", "")) for a in articles if isinstance(a, dict)}
        return set()

    @staticmethod
    def _get_entity_names(result: Dict) -> set:
        data = result.get("data", result)
        entities = data.get("entities", [])
        if isinstance(entities, list):
            return {e.get("name", e.get("text", "")) for e in entities if isinstance(e, dict)}
        elif isinstance(entities, dict):
            names = set()
            for category in entities.values():
                if isinstance(category, list):
                    names.update(str(e) for e in category)
            return names
        return set()


# ── GLOBAL SINGLETON ───────────────────────────────────────
# All graph nodes import and use this single instance.

memory_manager = MemoryManager()
