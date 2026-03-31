"""
memory/working.py — Working Memory Tier
════════════════════════════════════════

CONCEPT:
  Working memory is the LLM's "RAM" — fast, temporary, and limited.
  It holds the current session's conversation turns, extracted entities,
  and any "hot" context the LLM needs RIGHT NOW.

  Analogy: When you're having a conversation, working memory is what you're
  actively thinking about. You don't need to "remember" it — it's just there.
  When the conversation ends, it evaporates unless you write it down
  (that's the job of flush_to_episodic() in the agent loop).

IMPLEMENTATION:
  Pure Python dict. No DB, no disk. This means:
  - Zero latency reads/writes
  - Lost on process restart (by design — sessions are ephemeral)
  - Size-bounded to prevent unbounded growth
"""

import time
import uuid
from datetime import datetime
from typing import Any, Optional


class WorkingMemory:
    """
    In-session memory store. Lives only as long as the MCP server process.

    Internal structure:
      self._turns:    dict[turn_id, TurnRecord]     ← conversation history
      self._context:  dict[str, Any]                ← arbitrary key-value context
      self._entities: list[str]                     ← entities seen this session
    """

    # Maximum turns to keep before rolling the window
    # (prevents unbounded growth in long sessions)
    MAX_TURNS = 50

    def __init__(self):
        # conversation turns: ordered dict keyed by turn_id
        # Each turn: {id, role, content, timestamp, entities}
        self._turns: dict[str, dict] = {}

        # Ordered list of turn IDs so we can implement rolling window
        self._turn_order: list[str] = []

        # Free-form context: things like "user_mood", "current_topic", etc.
        self._context: dict[str, Any] = {}

        # Entities mentioned this session (deduplicated)
        self._entities: set[str] = set()

        # Session metadata
        self._session_id = str(uuid.uuid4())
        self._session_start = datetime.utcnow().isoformat()

        # Stats for debugging
        self._total_turns_ever = 0

    # ── Turn Management ──────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str, entities: Optional[list[str]] = None) -> str:
        """
        Add a conversation turn to working memory.

        Args:
            role: "user" or "assistant"
            content: the message text
            entities: optional list of extracted entities from this turn

        Returns:
            turn_id: unique ID for this turn (useful for deletion)

        Example:
            turn_id = wm.add_turn("user", "I'm building a Flask app in Python")
            # Internally stores: {role, content, timestamp, entities}
        """
        turn_id = str(uuid.uuid4())
        turn = {
            "id": turn_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            # unix timestamp for sorting/filtering
            "unix_ts": time.time(),
            # entities extracted from this turn (populated by extractor.py)
            "entities": entities or [],
        }

        self._turns[turn_id] = turn
        self._turn_order.append(turn_id)
        self._total_turns_ever += 1

        # Track entities seen this session
        if entities:
            self._entities.update(entities)

        # Rolling window: if we exceed MAX_TURNS, drop the oldest
        # This prevents working memory from becoming a full history log
        # (that's episodic memory's job)
        if len(self._turn_order) > self.MAX_TURNS:
            oldest_id = self._turn_order.pop(0)
            del self._turns[oldest_id]
            # Note: we DON'T delete from _entities because the entity was
            # mentioned this session even if the turn rolled off

        return turn_id

    def get_recent_turns(self, n: int = 10) -> list[dict]:
        """
        Get the N most recent turns as a list, oldest first.

        Used by the agent loop to build the LLM's conversation context.

        Example output:
          [
            {"role": "user", "content": "What's asyncio?", ...},
            {"role": "assistant", "content": "asyncio is...", ...},
          ]
        """
        recent_ids = self._turn_order[-n:]
        return [self._turns[tid] for tid in recent_ids]

    def get_all_turns(self) -> list[dict]:
        """Return all turns in order. Used when flushing to episodic memory."""
        return [self._turns[tid] for tid in self._turn_order]

    def clear_turns(self):
        """Wipe all turns. Called at session end after flushing to episodic."""
        self._turns.clear()
        self._turn_order.clear()

    # ── Context Management ───────────────────────────────────────────────────

    def set_context(self, key: str, value: Any):
        """
        Store arbitrary session context.

        Examples:
            wm.set_context("current_topic", "async Python")
            wm.set_context("user_frustration_level", "high")
            wm.set_context("last_tool_called", "recall")
        """
        self._context[key] = {
            "value": value,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve context value by key."""
        entry = self._context.get(key)
        return entry["value"] if entry else default

    def get_all_context(self) -> dict[str, Any]:
        """Return all context as a flat dict of {key: value}."""
        return {k: v["value"] for k, v in self._context.items()}

    # ── Entity Tracking ──────────────────────────────────────────────────────

    def add_entities(self, entities: list[str]):
        """Add extracted entities to the session entity set."""
        self._entities.update(entities)

    def get_entities(self) -> list[str]:
        """Return all unique entities seen this session."""
        return sorted(self._entities)

    # ── Session Snapshot ─────────────────────────────────────────────────────

    def get_session_summary(self) -> dict:
        """
        Return a summary of the current session state.
        Used by summarise_memories() tool and for debugging.
        """
        return {
            "session_id": self._session_id,
            "session_start": self._session_start,
            "total_turns_ever": self._total_turns_ever,
            "current_turns_in_window": len(self._turn_order),
            "entities_this_session": self.get_entities(),
            "context_keys": list(self._context.keys()),
        }

    def __repr__(self) -> str:
        return (
            f"WorkingMemory(session={self._session_id[:8]}..., "
            f"turns={len(self._turn_order)}, "
            f"entities={len(self._entities)})"
        )
