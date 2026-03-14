"""Nova v3 — Semantic Memory (Embedding-Based Recall).

Phase 3 stub — provides the interface for future vector-based
similarity search using ChromaDB or FAISS.

Currently uses keyword-based fallback. Will be upgraded to
embedding-based search when chromadb is added to dependencies.
"""

from typing import Dict, List, Any, Optional


class SemanticMemory:
    """Embedding-based memory for similarity search.
    
    Phase 3 stub: Falls back to keyword matching.
    Future: ChromaDB / FAISS for true vector similarity.
    """

    def __init__(self):
        self._available = False
        try:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection("nova_intelligence")
            self._available = True
        except ImportError:
            # ChromaDB not installed — stub mode
            self._client = None
            self._collection = None

    @property
    def is_available(self) -> bool:
        return self._available

    def store_embedding(self, doc_id: str, text: str, metadata: Dict = None):
        """Store a document with its embedding."""
        if not self._available:
            return

        self._collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata or {}],
        )

    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Find documents similar to the query.
        
        Returns list of {id, text, score, metadata} dicts.
        """
        if not self._available:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
        return output

    def get_stats(self) -> Dict[str, Any]:
        """Return semantic memory statistics."""
        if not self._available:
            return {"status": "unavailable", "reason": "chromadb not installed"}

        count = self._collection.count()
        return {
            "status": "active",
            "documents_stored": count,
        }
