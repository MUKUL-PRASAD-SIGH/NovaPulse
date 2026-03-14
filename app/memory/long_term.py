"""Nova v3 — Long-Term Memory (SQLite Persistent).

Stores query history, entity timelines, trend evolution, and
source reliability scores. Survives restarts.

Schema mirrors the blueprint Section 6, storing:
  - Query history with results & scores
  - Entity sighting timeline
  - Trend snapshots for comparison
"""

import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from threading import Lock


DB_DIR = os.path.join(os.path.dirname(__file__), "db")
DB_PATH = os.path.join(DB_DIR, "nova.db")


def _ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)


class LongTermMemory:
    """SQLite-backed persistent memory for Nova v3.
    
    Tables:
      - query_history: Every query + result + score
      - entity_timeline: Entity sightings across queries
      - trend_snapshots: Point-in-time trend captures
    """

    def __init__(self, db_path: str = DB_PATH, max_entries: int = 500):
        _ensure_db_dir()
        self._db_path = db_path
        self._max = max_entries
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS query_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        query_normalized TEXT NOT NULL,
                        depth TEXT DEFAULT 'standard',
                        pipelines TEXT DEFAULT '[]',
                        critic_score INTEGER DEFAULT 0,
                        confidence REAL DEFAULT 0.0,
                        result_json TEXT DEFAULT '{}',
                        entities_json TEXT DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        duration_ms INTEGER DEFAULT 0
                    );

                    CREATE INDEX IF NOT EXISTS idx_query_norm
                        ON query_history(query_normalized);
                    CREATE INDEX IF NOT EXISTS idx_created
                        ON query_history(created_at);

                    CREATE TABLE IF NOT EXISTS entity_timeline (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_name TEXT NOT NULL,
                        entity_type TEXT DEFAULT 'unknown',
                        sentiment REAL DEFAULT 0.5,
                        mention_count INTEGER DEFAULT 1,
                        source_query TEXT,
                        seen_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_entity_name
                        ON entity_timeline(entity_name);

                    CREATE TABLE IF NOT EXISTS trend_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        topic TEXT NOT NULL,
                        score REAL DEFAULT 0.0,
                        velocity TEXT DEFAULT 'stable',
                        related_entities TEXT DEFAULT '[]',
                        snapshot_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_trend_topic
                        ON trend_snapshots(topic);
                """)
                conn.commit()
            finally:
                conn.close()

    # ── QUERY HISTORY ──────────────────────────────────────────

    def store_query(self, query: str, result: Dict, depth: str = "standard",
                    pipelines: List[str] = None,
                    critic_score: int = 0, confidence: float = 0.0,
                    entities: List[Dict] = None, duration_ms: int = 0):
        """Persist a completed query and its result."""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO query_history
                    (query, query_normalized, depth, pipelines, critic_score,
                     confidence, result_json, entities_json, created_at, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query,
                    self._normalize(query),
                    depth,
                    json.dumps(pipelines or []),
                    critic_score,
                    confidence,
                    json.dumps(result, default=str),
                    json.dumps(entities or []),
                    datetime.utcnow().isoformat(),
                    duration_ms,
                ))
                conn.commit()
                self._cleanup(conn)
            finally:
                conn.close()

    def find_similar(self, query: str, max_age_hours: int = 168) -> Optional[Dict]:
        """Find the most recent similar query in history.
        
        Returns None if nothing relevant found.
        max_age_hours: Only look at queries from the last N hours (default: 7 days).
        
        Strategy:
          1. Exact match on normalized text
          2. Keyword overlap using normalized + raw tokens (handles stop-word collapse)
        """
        normalized = self._normalize(query)
        words = normalized.split()
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                # 1. Exact match
                row = conn.execute("""
                    SELECT * FROM query_history
                    WHERE query_normalized = ? AND created_at > ?
                    ORDER BY created_at DESC LIMIT 1
                """, (normalized, cutoff)).fetchone()

                if row:
                    return self._row_to_dict(row)

                # 2. Keyword overlap.
                # Also include raw tokens (pre-stop-word removal) to handle cases
                # where normalisation collapses the query to 0–1 meaningful words
                # e.g. "Tesla news" → "tesla" (only 1 word after removing "news")
                import re
                raw_tokens = [
                    w for w in re.sub(r"[^\w\s]", "", query.lower()).split()
                    if len(w) > 2
                ]
                # Deduplicated merged list, capped at 5 terms
                search_words = list(dict.fromkeys(words + raw_tokens))[:5]

                if search_words:
                    like_clauses = " OR ".join(
                        ["query_normalized LIKE ?"] * len(search_words)
                    )
                    params = [f"%{w}%" for w in search_words] + [cutoff]

                    row = conn.execute(f"""
                        SELECT * FROM query_history
                        WHERE ({like_clauses}) AND created_at > ?
                        ORDER BY created_at DESC LIMIT 1
                    """, params).fetchone()

                    if row:
                        return self._row_to_dict(row)

                return None
            finally:
                conn.close()

    def get_topic_history(self, topic: str, limit: int = 10) -> List[Dict]:
        """Get all past queries related to a topic."""
        normalized = f"%{self._normalize(topic)}%"

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT * FROM query_history
                    WHERE query_normalized LIKE ?
                    ORDER BY created_at DESC LIMIT ?
                """, (normalized, limit)).fetchall()
                return [self._row_to_dict(r) for r in rows]
            finally:
                conn.close()

    def compare_over_time(self, topic: str, days: int = 7) -> Dict[str, Any]:
        """Compare sentiment/trends for a topic over time.
        
        Returns a comparison dict with trend direction and key changes.
        """
        history = self.get_topic_history(topic, limit=20)
        if len(history) < 2:
            return {"comparison_available": False, "entries": len(history)}

        latest = history[0]
        oldest = history[-1]

        latest_result = json.loads(latest.get("result_json", "{}"))
        oldest_result = json.loads(oldest.get("result_json", "{}"))

        # Extract sentiments
        latest_sentiment = self._extract_sentiment(latest_result)
        oldest_sentiment = self._extract_sentiment(oldest_result)

        delta = latest_sentiment - oldest_sentiment

        return {
            "comparison_available": True,
            "entries": len(history),
            "time_span_hours": self._hours_between(oldest["created_at"], latest["created_at"]),
            "latest_score": latest.get("critic_score", 0),
            "oldest_score": oldest.get("critic_score", 0),
            "sentiment_latest": round(latest_sentiment, 3),
            "sentiment_oldest": round(oldest_sentiment, 3),
            "sentiment_delta": round(delta, 3),
            "direction": "improving" if delta > 0.05 else "declining" if delta < -0.05 else "stable",
        }

    # ── ENTITY TIMELINE ────────────────────────────────────────

    def record_entities(self, entities: List[Dict], source_query: str):
        """Record entity sightings from a query result."""
        if not entities:
            return

        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                for ent in entities[:50]:  # Cap at 50 per query
                    name = ent.get("name", ent.get("text", ""))
                    if not name:
                        continue
                    conn.execute("""
                        INSERT INTO entity_timeline
                        (entity_name, entity_type, sentiment, mention_count, source_query, seen_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        name,
                        ent.get("type", ent.get("category", "unknown")),
                        ent.get("sentiment", 0.5),
                        ent.get("mentions", ent.get("count", 1)),
                        source_query,
                        now,
                    ))
                conn.commit()
            finally:
                conn.close()

    def get_entity_timeline(self, entity_name: str, limit: int = 20) -> List[Dict]:
        """Get timeline of an entity's appearances."""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT * FROM entity_timeline
                    WHERE entity_name LIKE ?
                    ORDER BY seen_at DESC LIMIT ?
                """, (f"%{entity_name}%", limit)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    # ── TREND SNAPSHOTS ────────────────────────────────────────

    def record_trends(self, trends: List[Dict]):
        """Snapshot current trending topics."""
        if not trends:
            return

        now = datetime.utcnow().isoformat()
        with self._lock:
            conn = self._get_conn()
            try:
                for trend in trends[:30]:
                    topic = trend.get("topic", trend.get("name", ""))
                    if not topic:
                        continue
                    conn.execute("""
                        INSERT INTO trend_snapshots
                        (topic, score, velocity, related_entities, snapshot_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        topic,
                        trend.get("score", 0.0),
                        trend.get("velocity", "stable"),
                        json.dumps(trend.get("related_entities", [])),
                        now,
                    ))
                conn.commit()
            finally:
                conn.close()

    def detect_trend_changes(self, topic: str, hours: int = 24) -> Dict[str, Any]:
        """Detect if a topic's trend status has changed recently."""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute("""
                    SELECT * FROM trend_snapshots
                    WHERE topic LIKE ? AND snapshot_at > ?
                    ORDER BY snapshot_at DESC LIMIT 10
                """, (f"%{topic}%", cutoff)).fetchall()

                if len(rows) < 2:
                    return {"change_detected": False, "snapshots": len(rows)}

                latest = dict(rows[0])
                oldest = dict(rows[-1])

                score_delta = (latest.get("score", 0) or 0) - (oldest.get("score", 0) or 0)

                return {
                    "change_detected": abs(score_delta) > 5,
                    "snapshots": len(rows),
                    "score_latest": latest.get("score", 0),
                    "score_oldest": oldest.get("score", 0),
                    "score_delta": round(score_delta, 2),
                    "velocity_latest": latest.get("velocity", "stable"),
                    "velocity_oldest": oldest.get("velocity", "stable"),
                }
            finally:
                conn.close()

    # ── UTILITIES ──────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize query text for comparison."""
        import re
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        # Remove common filler words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
                       "for", "to", "with", "and", "or", "but", "about", "latest", "recent",
                       "tell", "me", "give", "show", "what", "news"}
        words = [w for w in text.split() if w not in stop_words]
        return " ".join(words) if words else text

    @staticmethod
    def _row_to_dict(row) -> Dict:
        return dict(row)

    @staticmethod
    def _extract_sentiment(result: Dict) -> float:
        """Pull sentiment score from a result dict."""
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                return 0.5

        if "data" in result:
            result = result["data"]

        sentiment = result.get("sentiment", {})
        if isinstance(sentiment, dict):
            return sentiment.get("score", 0.5)
        return 0.5

    @staticmethod
    def _hours_between(iso1: str, iso2: str) -> float:
        try:
            t1 = datetime.fromisoformat(iso1)
            t2 = datetime.fromisoformat(iso2)
            return round(abs((t2 - t1).total_seconds()) / 3600, 1)
        except Exception:
            return 0.0

    def _cleanup(self, conn):
        """Keep only the latest max_entries rows."""
        count = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
        if count > self._max:
            delete_count = count - self._max
            conn.execute("""
                DELETE FROM query_history
                WHERE id IN (
                    SELECT id FROM query_history ORDER BY created_at ASC LIMIT ?
                )
            """, (delete_count,))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Return memory statistics."""
        with self._lock:
            conn = self._get_conn()
            try:
                qh_count = conn.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
                et_count = conn.execute("SELECT COUNT(*) FROM entity_timeline").fetchone()[0]
                ts_count = conn.execute("SELECT COUNT(*) FROM trend_snapshots").fetchone()[0]
                return {
                    "query_history_entries": qh_count,
                    "entity_sightings": et_count,
                    "trend_snapshots": ts_count,
                    "db_path": self._db_path,
                }
            finally:
                conn.close()
