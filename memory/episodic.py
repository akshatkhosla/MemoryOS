"""
memory/episodic.py — Episodic Memory Tier
══════════════════════════════════════════

CONCEPT:
  Episodic memory stores *events* — things that happened in past sessions.
  "On Tuesday you asked me about Azure AD B2C."
  "Three sessions ago you mentioned you were stressed about a deadline."

  Unlike working memory (ephemeral) or semantic memory (dry facts),
  episodic memory captures the *narrative* of past interactions.

  It uses vector embeddings so the LLM can recall memories by meaning,
  not just keyword match. "tell me about auth issues I had" can surface
  a memory that says "user struggled with JWT token expiry in .NET" even
  though no keyword matches perfectly.

IMPLEMENTATION:
  ChromaDB as the vector store. ChromaDB:
  - Runs fully in-process (no separate server needed)
  - Persists to disk automatically
  - Handles embedding storage + cosine similarity search
  - Returns results with relevance scores

  We bring our OWN embeddings (SentenceTransformers) rather than letting
  ChromaDB embed for us. Why? More control, works offline, same model
  used everywhere in MemoryOS for consistency.

MEMORY STRUCTURE:
  Each episodic memory has:
  - id:          UUID string
  - content:     the raw text of the memory
  - embedding:   384-dim vector from all-MiniLM-L6-v2
  - metadata:    {timestamp, importance, source, session_id, entities}
"""

import uuid
import logging
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Persistent vector memory for past conversation events.

    Wraps ChromaDB with our embedding model and provides a clean
    interface for the MCP tools to store and retrieve memories.
    """

    # The embedding model. all-MiniLM-L6-v2:
    # - 384-dimensional vectors
    # - ~80MB download (cached after first use)
    # - 14,000+ tokens/second on CPU
    # - Good semantic understanding for conversational text
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # ChromaDB collection name
    COLLECTION_NAME = "episodic_memories"

    def __init__(self, persist_dir: str = "./data/chroma"):
        """
        Initialize ChromaDB client and load embedding model.

        Args:
            persist_dir: where ChromaDB stores its files on disk.
                         Default: ./data/chroma (relative to cwd)
        """
        logger.info(f"Initializing EpisodicMemory (persist_dir={persist_dir})")

        # ── ChromaDB Setup ────────────────────────────────────────────────
        # PersistentClient: stores data to disk automatically.
        # Every write is immediately durable.
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                # Disable telemetry — we're privacy-conscious here
                anonymized_telemetry=False,
            ),
        )

        # Get or create our collection.
        # IMPORTANT: We use "none" embedding function because we provide
        # our own embeddings. If you let ChromaDB embed, it downloads its
        # own model and you lose control over consistency.
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            # cosine similarity: best for comparing semantic meaning
            # (dot product is faster but requires normalized vectors)
            metadata={"hnsw:space": "cosine"},
        )

        # ── Embedding Model Setup ─────────────────────────────────────────
        # SentenceTransformer downloads the model on first use and caches it.
        # Subsequent loads are instant (uses ~/.cache/huggingface).
        self._embedder = None
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
            self._embedder = SentenceTransformer(self.EMBEDDING_MODEL)
            logger.info("EpisodicMemory ready.")
        except Exception as e:
            logger.warning(
                "EpisodicMemory embedding backend is unavailable. "
                "Install sentence-transformers and compatible PyTorch/NumPy versions if you want episodic recall."
            )
            logger.warning("EpisodicMemory disabled: %s", e)

    # ── Core Operations ──────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        importance: float = 0.5,
        source: str = "conversation",
        session_id: Optional[str] = None,
        entities: Optional[list[str]] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """
        Store a new episodic memory.

        Args:
            content:    the memory text. Can be a conversation excerpt,
                        a summary, or any free-form text.
            importance: 0.0 to 1.0. Higher = prioritized in recall.
                        Used to break ties when two memories are equally similar.
            source:     where this memory came from:
                        "conversation" | "summary" | "manual"
            session_id: which session this memory came from
            entities:   list of entity strings (from extractor.py)
            memory_id:  optional pre-assigned ID (for idempotent writes)

        Returns:
            memory_id: the UUID of the stored memory

        Example:
            mid = episodic.store(
                content="User mentioned they're building a side project called MemoryOS",
                importance=0.8,
                entities=["MemoryOS"]
            )
        """
        memory_id = memory_id or str(uuid.uuid4())

        if self._embedder is None:
            logger.warning(
                "Skipping episodic memory store because embedding backend is unavailable. "
                "Install sentence-transformers and compatible PyTorch/NumPy versions to enable this feature."
            )
            return memory_id

        # Generate embedding for the content
        # encode() returns a numpy array; .tolist() converts to plain Python list
        # ChromaDB needs a list of lists (one embedding per document)
        embedding = self._embedder.encode(content).tolist()

        # Metadata stored alongside the vector.
        # ChromaDB metadata must be: str, int, float, or bool — no lists/dicts.
        # So we serialize entities as a comma-separated string.
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "unix_ts": datetime.utcnow().timestamp(),
            "importance": float(importance),
            "source": source,
            "session_id": session_id or "unknown",
            # ChromaDB can't store lists — serialize to string
            "entities": ",".join(entities) if entities else "",
            "content_length": len(content),
        }

        # ChromaDB add() takes lists (batch API) even for single items
        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

        logger.debug(f"Stored episodic memory {memory_id[:8]}... importance={importance}")
        return memory_id

    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
        since_timestamp: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search: find the most relevant past memories for a query.

        This is the core magic of episodic memory. We embed the query,
        then find the stored memories whose embeddings are closest in
        cosine similarity space.

        Args:
            query:           natural language search query
            top_k:           number of results to return
            min_importance:  filter out memories below this importance score
            since_timestamp: ISO timestamp — only return memories after this date

        Returns:
            List of memory dicts, sorted by relevance score (descending).
            Each dict: {id, content, metadata, relevance_score}

        Example:
            results = episodic.recall("what did the user say about Python?", top_k=3)
            # Returns memories semantically related to Python discussions
        """
        if self._embedder is None:
            logger.warning(
                "Skipping episodic memory recall because embedding backend is unavailable. "
                "Install sentence-transformers and compatible PyTorch/NumPy versions to enable this feature."
            )
            return []

        # Check if collection is empty (ChromaDB throws on empty query)
        count = self._collection.count()
        if count == 0:
            logger.debug("EpisodicMemory is empty — nothing to recall")
            return []

        # Embed the query using the same model used for storage.
        # CRITICAL: You must use the same model for query and document embeddings.
        # Mixing models breaks semantic similarity — they live in different spaces.
        query_embedding = self._embedder.encode(query).tolist()

        # Build ChromaDB where filter if needed
        # ChromaDB uses a dict-based filter syntax
        where_filter = None
        if min_importance > 0.0:
            where_filter = {"importance": {"$gte": min_importance}}

        # Query ChromaDB for top_k most similar memories
        # n_results can't exceed collection size
        n_results = min(top_k, count)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            # include both the text and metadata in results
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists (batch API): results["documents"][0] = list for query 0
        # We only have one query, so we unwrap [0]
        memories = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]

            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity score 0→1 (1 = perfect match)
            # Formula: similarity = 1 - (distance / 2)
            similarity = 1.0 - (distance / 2.0)

            metadata = results["metadatas"][0][i]
            content = results["documents"][0][i]

            # Deserialize entities back to list
            entities_str = metadata.get("entities", "")
            entities = [e.strip() for e in entities_str.split(",") if e.strip()]

            memories.append({
                "id": doc_id,
                "content": content,
                "relevance_score": round(similarity, 4),
                "importance": metadata.get("importance", 0.5),
                "timestamp": metadata.get("timestamp", ""),
                "session_id": metadata.get("session_id", ""),
                "source": metadata.get("source", ""),
                "entities": entities,
            })

        # Sort by a combined score: relevance * (1 + importance)
        # This means a slightly less relevant but high-importance memory
        # can outrank a highly relevant but low-importance one.
        memories.sort(
            key=lambda m: m["relevance_score"] * (1 + m["importance"]),
            reverse=True,
        )

        return memories

    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Returns True if deleted, False if not found.
        """
        try:
            self._collection.delete(ids=[memory_id])
            logger.debug(f"Deleted episodic memory {memory_id[:8]}...")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")
            return False

    def get_by_id(self, memory_id: str) -> Optional[dict]:
        """Retrieve a specific memory by its ID."""
        try:
            result = self._collection.get(
                ids=[memory_id],
                include=["documents", "metadatas"],
            )
            if not result["ids"]:
                return None
            return {
                "id": result["ids"][0],
                "content": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        except Exception:
            return None

    def count(self) -> int:
        """Return total number of stored episodic memories."""
        return self._collection.count()

    def get_all_for_session(self, session_id: str) -> list[dict]:
        """
        Get all memories from a specific session.
        Used by summarise_memories() to consolidate a session's memories.
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            where={"session_id": {"$eq": session_id}},
            include=["documents", "metadatas"],
        )

        memories = []
        for i, doc_id in enumerate(results["ids"]):
            memories.append({
                "id": doc_id,
                "content": results["documents"][i],
                "metadata": results["metadatas"][i],
            })

        return memories

    def get_recent(self, n: int = 10) -> list[dict]:
        """
        Return the N most recently stored memories (by timestamp).
        Useful for 'what have I been thinking about lately?' queries.
        """
        if self._collection.count() == 0:
            return []

        # ChromaDB doesn't support ORDER BY natively.
        # Strategy: get all, sort in Python, return top N.
        # For large collections, you'd want to add a time-based index.
        # For most personal memory use cases, total memories < 10,000 — fine.
        results = self._collection.get(include=["documents", "metadatas"])

        memories = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            memories.append({
                "id": doc_id,
                "content": results["documents"][i],
                "timestamp": meta.get("timestamp", ""),
                "unix_ts": meta.get("unix_ts", 0),
                "importance": meta.get("importance", 0.5),
            })

        # Sort by unix timestamp descending
        memories.sort(key=lambda m: m["unix_ts"], reverse=True)
        return memories[:n]

    def __repr__(self) -> str:
        return f"EpisodicMemory(count={self.count()}, model={self.EMBEDDING_MODEL})"
