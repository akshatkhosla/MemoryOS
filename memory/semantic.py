"""
memory/semantic.py — Semantic Memory Tier
══════════════════════════════════════════

CONCEPT:
  Semantic memory stores *facts* — stable, structured knowledge about the
  world and the user. Unlike episodic memory (narrative events), semantic
  memory is dry and factual:

  "User's name is Akshat"
  "User is based in Hyderabad"
  "User's preferred language is Python"
  "Project DocMind uses ChromaDB"

  Think of it as the LLM's knowledge graph of the user's world.

  These facts are extracted by extractor.py using spaCy NER and
  simple pattern matching on conversation turns.

WHY SQLITE (not ChromaDB)?
  Semantic facts need:
  - Exact lookup: "what is user.name?" → deterministic answer
  - Deduplication: "user lives in Hyderabad" should UPDATE not duplicate
  - Atomic updates: changing a fact should be transactional
  - Simple querying: "show all facts about user"

  Vector similarity is wrong for this — "user.name = Akshat" should
  return EXACTLY "Akshat", not the 5 most similar facts.
  SQLite is perfect: zero-dependency, file-based, ACID-compliant.

SCHEMA:
  Table: facts
  ┌────────────┬────────────┬────────────┬────────────┬────────────────────┐
  │ id (PK)    │ entity     │ attribute  │ value      │ confidence         │
  │ TEXT       │ TEXT       │ TEXT       │ TEXT       │ REAL               │
  ├────────────┼────────────┼────────────┼────────────┼────────────────────┤
  │ uuid       │ "user"     │ "name"     │ "Akshat"   │ 0.99               │
  │ uuid       │ "user"     │ "location" │ "Hyderabad"│ 0.95               │
  │ uuid       │ "project"  │ "name"     │ "DocMind"  │ 0.99               │
  └────────────┴────────────┴────────────┴────────────┴────────────────────┘

  Additional columns: source, created_at, updated_at, times_confirmed
"""

import sqlite3
import uuid
import logging
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


# ── Schema Definition ────────────────────────────────────────────────────────

CREATE_FACTS_TABLE = """
CREATE TABLE IF NOT EXISTS facts (
    id              TEXT PRIMARY KEY,
    entity          TEXT NOT NULL,       -- e.g. "user", "project", "technology"
    attribute       TEXT NOT NULL,       -- e.g. "name", "location", "preference"
    value           TEXT NOT NULL,       -- e.g. "Akshat", "Hyderabad", "Python"
    confidence      REAL DEFAULT 0.5,   -- 0.0 to 1.0, how sure we are
    source          TEXT DEFAULT 'extracted',  -- "extracted" | "manual" | "confirmed"
    times_confirmed INTEGER DEFAULT 1,  -- incremented when same fact seen again
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
"""

# Index for fast entity+attribute lookups (the most common query pattern)
CREATE_ENTITY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_entity_attribute
ON facts (entity, attribute);
"""

# Index for entity-only lookups ("show all facts about user")
CREATE_ENTITY_ONLY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_entity
ON facts (entity);
"""


# ── SemanticMemory Class ─────────────────────────────────────────────────────

class SemanticMemory:
    """
    SQLite-backed structured fact store.

    Provides CRUD operations for entity-attribute-value triples,
    with confidence scoring and deduplication.
    """

    def __init__(self, db_path: str = "./data/semantic.db"):
        """
        Initialize SQLite connection and create tables if they don't exist.

        Args:
            db_path: path to the SQLite database file.
                     Created automatically if it doesn't exist.
        """
        self._db_path = db_path
        logger.info(f"Initializing SemanticMemory (db={db_path})")
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """
        Context manager for database connections.

        Why not keep a persistent connection?
        MCP tools are called asynchronously and SQLite connections
        are not thread-safe by default. Creating a connection per
        operation is safe and fast (SQLite is embedded, no TCP overhead).
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row  # allows dict-like access: row["column"]
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        with self._get_conn() as conn:
            conn.execute(CREATE_FACTS_TABLE)
            conn.execute(CREATE_ENTITY_INDEX)
            conn.execute(CREATE_ENTITY_ONLY_INDEX)
        logger.info("SemanticMemory schema initialized.")

    # ── Core Operations ──────────────────────────────────────────────────────

    def upsert_fact(
        self,
        entity: str,
        attribute: str,
        value: str,
        confidence: float = 0.7,
        source: str = "extracted",
    ) -> tuple[str, bool]:
        """
        Insert a new fact or update an existing one.

        Supports both single-valued and multi-valued attributes:
        - Single-valued (e.g., "user.name"): updates existing value
        - Multi-valued (e.g., "technology.name"): allows multiple values,
          only updates if SAME VALUE already exists

        This allows tech stacks like:
          - technology.name = "Python"
          - technology.name = "ChromaDB"
          - technology.name = "sentence-transformers"
        to all be stored instead of overwriting each other.

        Args:
            entity:     the subject ("user", "project", "technology")
            attribute:  the property ("name", "location", "preference")
            value:      the value ("Akshat", "Hyderabad", "Python")
            confidence: how confident we are in this fact (0.0 to 1.0)
            source:     where this fact came from

        Returns:
            (fact_id, was_updated): True if updated existing fact with same value

        Example:
            id1, _ = semantic.upsert_fact("technology", "name", "Python", 0.70)
            id2, _ = semantic.upsert_fact("technology", "name", "ChromaDB", 0.70)
            # Both facts stored (different values, different IDs)
        """
        entity = entity.lower().strip()
        attribute = attribute.lower().strip()
        value = value.strip()
        now = datetime.utcnow().isoformat()

        with self._get_conn() as conn:
            # Check for exact match: same entity + attribute + value
            # (We update only if the value is identical, not if just attribute matches)
            existing = conn.execute(
                "SELECT id, confidence, times_confirmed FROM facts "
                "WHERE entity = ? AND attribute = ? AND value = ?",
                (entity, attribute, value),
            ).fetchone()

            if existing:
                # ── UPDATE path: same value seen again ──────────────────────
                fact_id = existing["id"]
                old_confidence = existing["confidence"]

                # Confidence update: weighted average of old and new
                new_confidence = (old_confidence * 0.7) + (confidence * 0.3)

                conn.execute(
                    """
                    UPDATE facts
                    SET confidence = ?,
                        times_confirmed = times_confirmed + 1,
                        updated_at = ?,
                        source = ?
                    WHERE id = ?
                    """,
                    (new_confidence, now, source, fact_id),
                )
                logger.debug(
                    f"Updated fact: {entity}.{attribute} = {value!r} "
                    f"(confidence: {old_confidence:.2f} → {new_confidence:.2f})"
                )
                return fact_id, True

            else:
                # ── INSERT path: new entity+attribute+value combination ────
                fact_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO facts
                        (id, entity, attribute, value, confidence, source,
                         times_confirmed, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (fact_id, entity, attribute, value,
                     confidence, source, now, now),
                )
                logger.debug(
                    f"Inserted fact: {entity}.{attribute} = {value!r} "
                    f"(confidence: {confidence:.2f})"
                )
                return fact_id, False

    def get_fact(self, entity: str, attribute: str) -> Optional[dict]:
        """
        Retrieve a specific fact by entity+attribute.

        Returns None if not found.

        Example:
            fact = semantic.get_fact("user", "name")
            # → {"id": "...", "entity": "user", "attribute": "name",
            #    "value": "Akshat", "confidence": 0.99, ...}
        """
        entity = entity.lower().strip()
        attribute = attribute.lower().strip()

        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM facts WHERE entity = ? AND attribute = ?",
                (entity, attribute),
            ).fetchone()

        return dict(row) if row else None

    def get_entity_facts(self, entity: str) -> list[dict]:
        """
        Get all facts about a given entity.

        Example:
            facts = semantic.get_entity_facts("user")
            # → [
            #     {"attribute": "name", "value": "Akshat", ...},
            #     {"attribute": "location", "value": "Hyderabad", ...},
            #   ]
        """
        entity = entity.lower().strip()

        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM facts WHERE entity = ? ORDER BY confidence DESC",
                (entity,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_all_facts(self, min_confidence: float = 0.0) -> list[dict]:
        """
        Return all stored facts, optionally filtered by confidence.

        Used by list_entities() MCP tool.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM facts
                WHERE confidence >= ?
                ORDER BY entity, confidence DESC
                """,
                (min_confidence,),
            ).fetchall()

        return [dict(row) for row in rows]

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by its ID. Returns True if found and deleted."""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            deleted = cursor.rowcount > 0

        if deleted:
            logger.debug(f"Deleted semantic fact {fact_id[:8]}...")
        return deleted

    def delete_entity(self, entity: str) -> int:
        """
        Delete all facts about an entity.

        Returns count of deleted rows.

        Example:
            n = semantic.delete_entity("user")
            # Deletes all user.* facts
        """
        entity = entity.lower().strip()
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM facts WHERE entity = ?", (entity,))
            return cursor.rowcount

    def search_facts(self, query: str) -> list[dict]:
        """
        Simple text search across all fact values and attributes.

        Not semantic — this is a SQL LIKE search. For exact fact lookup
        this is more appropriate than vector similarity.

        Example:
            results = semantic.search_facts("Python")
            # Returns all facts where value or attribute contains "Python"
        """
        pattern = f"%{query}%"
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM facts
                WHERE value LIKE ? OR attribute LIKE ? OR entity LIKE ?
                ORDER BY confidence DESC
                LIMIT 20
                """,
                (pattern, pattern, pattern),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_all_entities(self) -> list[str]:
        """Return list of all distinct entity names."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT entity FROM facts ORDER BY entity"
            ).fetchall()

        return [row["entity"] for row in rows]

    def fact_count(self) -> int:
        """Return total number of stored facts."""
        with self._get_conn() as conn:
            result = conn.execute("SELECT COUNT(*) as cnt FROM facts").fetchone()
        return result["cnt"]

    def get_high_confidence_facts(self, threshold: float = 0.8) -> list[dict]:
        """
        Return all facts above a confidence threshold.
        Used to build the LLM's 'permanent context' — facts it should
        always be aware of regardless of query.
        """
        return self.get_all_facts(min_confidence=threshold)

    def format_as_context(self, facts: Optional[list[dict]] = None) -> str:
        """
        Format facts as a readable string for injection into LLM context.

        Example output:
          Known facts:
          - user.name = Akshat (confidence: 99%)
          - user.location = Hyderabad (confidence: 95%)
          - user.lang_pref = Python (confidence: 87%)
        """
        if facts is None:
            facts = self.get_high_confidence_facts(threshold=0.7)

        if not facts:
            return "No structured facts available."

        lines = ["Known facts about the user and their context:"]
        for fact in facts:
            pct = int(fact["confidence"] * 100)
            lines.append(
                f"  - {fact['entity']}.{fact['attribute']} = {fact['value']!r} "
                f"(confidence: {pct}%, confirmed {fact['times_confirmed']}x)"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SemanticMemory(db={self._db_path!r}, facts={self.fact_count()})"
