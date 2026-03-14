"""Nova v3 — Short-Term Memory (Session-Scoped).

Holds ephemeral context for the current session. Volatile — resets on restart.
Think of this as the system's working memory / scratchpad.
"""

import time
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from threading import Lock


class ShortTermMemory:
    """In-memory session store with TTL and LRU eviction.
    
    Usage:
        stm = ShortTermMemory(max_entries=200, ttl_seconds=3600)
        stm.store("query_abc", {"report": ...})
        result = stm.recall("query_abc")
    """

    def __init__(self, max_entries: int = 200, ttl_seconds: int = 3600):
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_seconds
        self._lock = Lock()

    def store(self, key: str, data: Dict[str, Any], metadata: Optional[Dict] = None):
        """Store data with auto-timestamping."""
        with self._lock:
            entry = {
                "data": data,
                "metadata": metadata or {},
                "stored_at": time.time(),
                "access_count": 0,
            }
            # Move to end (most recent)
            if key in self._store:
                del self._store[key]
            self._store[key] = entry

            # Evict oldest if over capacity
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def recall(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data if it exists and hasn't expired."""
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None

            # Check TTL
            age = time.time() - entry["stored_at"]
            if age > self._ttl:
                del self._store[key]
                return None

            # Update access count and move to end
            entry["access_count"] += 1
            self._store.move_to_end(key)
            return entry["data"]

    def find_similar_key(self, query: str) -> Optional[str]:
        """Simple keyword-based lookup for similar past queries.
        
        Returns the key of the most similar recent entry, if any.
        Phase 3 will use embedding-based similarity.
        """
        query_words = set(query.lower().split())
        best_key = None
        best_overlap = 0

        with self._lock:
            for key in reversed(self._store):
                entry = self._store[key]
                age = time.time() - entry["stored_at"]
                if age > self._ttl:
                    continue

                key_words = set(key.lower().split())
                overlap = len(query_words & key_words)

                if overlap > best_overlap and overlap >= 2:
                    best_overlap = overlap
                    best_key = key

        return best_key

    def get_session_context(self) -> Dict[str, Any]:
        """Return a summary of the current session for supervisor context."""
        with self._lock:
            self._evict_expired()
            return {
                "active_entries": len(self._store),
                "recent_queries": list(self._store.keys())[-5:],
                "session_duration_s": round(
                    time.time() - min(
                        (e["stored_at"] for e in self._store.values()),
                        default=time.time()
                    )
                ),
            }

    def clear(self):
        """Wipe all session data."""
        with self._lock:
            self._store.clear()

    def _evict_expired(self):
        """Remove expired entries."""
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v["stored_at"] > self._ttl]
        for k in expired:
            del self._store[k]

    def __len__(self):
        return len(self._store)
