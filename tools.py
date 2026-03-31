"""
tools.py — MCP Tool Definitions and Handlers
═════════════════════════════════════════════

CONCEPT:
  This is the bridge between the MCP protocol layer and the memory system.

  In DocMind you manually defined tool schemas as dicts and routed
  tool_use blocks with if/elif chains. MCP formalizes this:

  1. Tool SCHEMAS are declared with @mcp.tool() decorators
  2. The MCP SDK auto-generates the JSON Schema from Python type hints
  3. The SDK handles routing — when a client calls "remember", it
     automatically invokes the remember() function below
  4. Return values are auto-serialized to MCP content format

  JSON SCHEMA AUTO-GENERATION:
  Python signature:
      async def remember(content: str, importance: float = 0.5) -> str:

  MCP SDK generates:
      {
        "name": "remember",
        "description": "...",
        "inputSchema": {
          "type": "object",
          "properties": {
            "content": {"type": "string"},
            "importance": {"type": "number", "default": 0.5}
          },
          "required": ["content"]
        }
      }

  The LLM client receives this schema and uses it to decide when/how
  to call the tool — it knows exactly what arguments to provide.
"""

import logging
from typing import Optional
from mcp.server import Server
from mcp.types import TextContent

from memory import MemorySystem
from extractor import Extractor

logger = logging.getLogger(__name__)


def register_tools(mcp: Server, memory: MemorySystem, extractor: Extractor):
    """
    Register all MemoryOS tools with the MCP server.

    Why a function instead of module-level decorators?
    The MCP server instance and memory system need to be initialized first
    (they're created in server.py). This function wires them together.

    Args:
        mcp:       the MCP Server instance (from server.py)
        memory:    the MemorySystem (all three tiers)
        extractor: the Extractor (spaCy + patterns)
    """

    # ════════════════════════════════════════════════════════════════════════
    # TOOL 1: remember()
    # ════════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def remember(
        content: str,
        importance: float = 0.5,
        tier: Optional[str] = None,
    ) -> str:
        """
        Store a new memory. Auto-classifies into the appropriate memory tier.

        Args:
            content:    The content to remember. Can be a fact ("User's name
                        is Akshat"), a conversation excerpt, or any free-form text.
            importance: Importance score from 0.0 (trivial) to 1.0 (critical).
                        If not provided, computed automatically from content.
            tier:       Override auto-classification: "working", "episodic",
                        or "semantic". Leave empty for auto-routing.

        Returns:
            Confirmation message with the memory ID and tier used.
        """
        logger.info(f"remember() called: content={content[:50]!r}... importance={importance}")

        try:
            # ── Step 1: Extract entities from content ────────────────────
            # Even if we're storing this in episodic memory, we want to:
            # 1. Tag the memory with entities for better recall
            # 2. Potentially upsert high-confidence facts to semantic memory
            entities = extractor.extract_entities(content)
            entity_strings = [e["text"] for e in entities]

            # ── Step 2: Compute importance if not provided ───────────────
            # The LLM may not always set importance; compute a default.
            if importance == 0.5:  # default was passed — compute it
                importance = extractor.compute_importance(content, entities)

            # ── Step 3: Determine target tier ────────────────────────────
            if tier is None:
                tier = extractor.classify_memory_tier(content)

            # ── Step 4: Store in appropriate tier ────────────────────────
            memory_id = None
            tier_used = tier

            if tier == "semantic":
                # Try to extract a structured fact from the content
                facts = extractor.extract_facts(content, role="user")
                if facts:
                    # Store the first extracted fact (highest confidence)
                    best_fact = max(facts, key=lambda f: f["confidence"])
                    fact_id, was_updated = memory.semantic.upsert_fact(
                        entity=best_fact["entity"],
                        attribute=best_fact["attribute"],
                        value=best_fact["value"],
                        confidence=best_fact["confidence"],
                        source="remember_tool",
                    )
                    memory_id = fact_id
                    action = "Updated existing" if was_updated else "Stored new"
                    return (
                        f"{action} semantic fact [{fact_id[:8]}]: "
                        f"{best_fact['entity']}.{best_fact['attribute']} = "
                        f"{best_fact['value']!r} (confidence: {best_fact['confidence']:.0%})"
                    )
                else:
                    # Can't extract a structured fact — fall back to episodic
                    tier_used = "episodic"
                    logger.debug("No structured facts found; falling back to episodic tier")

            if tier_used in ("episodic", "working"):
                # Get session ID from working memory
                session_id = memory.working.get_session_summary()["session_id"]

                memory_id = memory.episodic.store(
                    content=content,
                    importance=importance,
                    source="remember_tool",
                    session_id=session_id,
                    entities=entity_strings,
                )

                # Also store extracted facts in semantic tier as a bonus
                facts = extractor.extract_facts(content, role="user")
                for fact in facts:
                    if fact["confidence"] >= 0.75:
                        memory.semantic.upsert_fact(
                            entity=fact["entity"],
                            attribute=fact["attribute"],
                            value=fact["value"],
                            confidence=fact["confidence"],
                            source="remember_tool_bonus",
                        )

                return (
                    f"Stored episodic memory [{memory_id[:8]}] "
                    f"(importance: {importance:.0%}, entities: {entity_strings or 'none'})"
                )

        except Exception as e:
            logger.error(f"remember() failed: {e}", exc_info=True)
            return f"Error storing memory: {str(e)}"

    # ════════════════════════════════════════════════════════════════════════
    # TOOL 2: recall()
    # ════════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def recall(
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
        search_semantic: bool = True,
        search_episodic: bool = True,
    ) -> str:
        """
        Search across all memory tiers for relevant memories.

        This is the primary retrieval tool. The LLM should call this
        at the start of a conversation to load relevant context, and
        whenever the user asks about something that might have been
        discussed before.

        Args:
            query:           Natural language query. E.g. "what do I know about
                             the user's tech preferences?"
            top_k:           Maximum number of results to return (default: 5).
            min_importance:  Filter out memories below this importance (0.0-1.0).
            search_semantic: Include structured facts in results (default: True).
            search_episodic: Include episodic memories in results (default: True).

        Returns:
            Formatted string with ranked memories and their relevance scores.
        """
        logger.info(f"recall() called: query={query!r}, top_k={top_k}")

        results = []

        try:
            # ── Semantic tier: exact + fuzzy SQL search ──────────────────
            if search_semantic:
                semantic_facts = memory.semantic.search_facts(query)
                for fact in semantic_facts[:top_k]:
                    results.append({
                        "tier": "semantic",
                        "content": (
                            f"{fact['entity']}.{fact['attribute']} = {fact['value']!r} "
                            f"(confidence: {fact['confidence']:.0%})"
                        ),
                        "score": fact["confidence"],  # confidence acts as relevance
                        "id": fact["id"],
                        "timestamp": fact.get("updated_at", ""),
                    })

            # ── Episodic tier: vector similarity search ──────────────────
            if search_episodic:
                episodic_memories = memory.episodic.recall(
                    query=query,
                    top_k=top_k,
                    min_importance=min_importance,
                )
                for mem in episodic_memories:
                    results.append({
                        "tier": "episodic",
                        "content": mem["content"],
                        "score": mem["relevance_score"],
                        "id": mem["id"],
                        "timestamp": mem["timestamp"],
                        "entities": mem.get("entities", []),
                    })

            # ── Working memory: check current session context ────────────
            recent_turns = memory.working.get_recent_turns(n=3)
            if recent_turns:
                # Simple keyword check — is the query relevant to recent turns?
                query_words = set(query.lower().split())
                for turn in recent_turns:
                    turn_words = set(turn["content"].lower().split())
                    overlap = query_words & turn_words
                    if len(overlap) >= 2:  # at least 2 words in common
                        results.append({
                            "tier": "working",
                            "content": f"[Current session] {turn['role']}: {turn['content'][:200]}",
                            "score": 0.6,  # lower score since not semantic search
                            "id": turn["id"],
                            "timestamp": turn["timestamp"],
                        })

            if not results:
                return (
                    f"No memories found for query: {query!r}\n"
                    "The memory store may be empty, or try a different query."
                )

            # ── Sort all results by score descending ──────────────────────
            results.sort(key=lambda r: r["score"], reverse=True)
            results = results[:top_k]

            # ── Format output for LLM consumption ────────────────────────
            lines = [f"Found {len(results)} relevant memories for: {query!r}\n"]
            for i, r in enumerate(results, 1):
                tier_icon = {"semantic": "📋", "episodic": "💭", "working": "⚡"}.get(r["tier"], "•")
                lines.append(
                    f"{i}. {tier_icon} [{r['tier'].upper()}] "
                    f"(relevance: {r['score']:.0%})\n"
                    f"   {r['content']}\n"
                    f"   ID: {r['id'][:8]}... | {r['timestamp'][:19]}"
                )
                if r.get("entities"):
                    lines.append(f"   Entities: {', '.join(r['entities'])}")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"recall() failed: {e}", exc_info=True)
            return f"Error during recall: {str(e)}"

    # ════════════════════════════════════════════════════════════════════════
    # TOOL 3: forget()
    # ════════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def forget(memory_id: str) -> str:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The ID of the memory to delete (from recall results).
                       Can be the full UUID or just the first 8 characters.

        Returns:
            Confirmation message.
        """
        logger.info(f"forget() called: memory_id={memory_id!r}")

        # Try episodic first (most common)
        deleted = memory.episodic.delete(memory_id)
        if deleted:
            return f"Deleted episodic memory {memory_id[:8]}..."

        # Try semantic
        deleted = memory.semantic.delete_fact(memory_id)
        if deleted:
            return f"Deleted semantic fact {memory_id[:8]}..."

        return f"Memory {memory_id[:8]}... not found in any tier."

    # ════════════════════════════════════════════════════════════════════════
    # TOOL 4: summarise_memories()
    # ════════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def summarise_memories(max_memories: int = 20) -> str:
        """
        Get a high-level summary of everything stored in memory.

        Use this at the start of a session to quickly understand what the
        LLM knows about the user, or before context window fills up.

        Unlike recall(), this doesn't search by query — it gives an overview
        of ALL memory across all tiers.

        Args:
            max_memories: Maximum episodic memories to include in summary.

        Returns:
            Formatted summary string covering all three memory tiers.
        """
        logger.info("summarise_memories() called")

        try:
            lines = ["# MemoryOS Summary\n"]

            # ── Working memory summary ────────────────────────────────────
            session = memory.working.get_session_summary()
            lines.append("## 📌 Current Session (Working Memory)")
            lines.append(f"- Session ID: {session['session_id'][:8]}...")
            lines.append(f"- Started: {session['session_start']}")
            lines.append(f"- Turns this session: {session['current_turns_in_window']}")
            entities_this_session = session.get("entities_this_session", [])
            if entities_this_session:
                lines.append(f"- Entities mentioned: {', '.join(entities_this_session)}")
            lines.append("")

            # ── Semantic memory summary ───────────────────────────────────
            lines.append("## 📋 Known Facts (Semantic Memory)")
            all_facts = memory.semantic.get_all_facts(min_confidence=0.6)
            if all_facts:
                # Group by entity
                entity_groups: dict[str, list] = {}
                for fact in all_facts:
                    entity_groups.setdefault(fact["entity"], []).append(fact)

                for entity, facts in sorted(entity_groups.items()):
                    lines.append(f"\n### {entity.title()}")
                    for fact in facts:
                        pct = int(fact["confidence"] * 100)
                        lines.append(
                            f"  - {fact['attribute']}: {fact['value']!r} "
                            f"({pct}% confidence, confirmed {fact['times_confirmed']}x)"
                        )
            else:
                lines.append("  (no structured facts stored yet)")
            lines.append("")

            # ── Episodic memory summary ───────────────────────────────────
            lines.append("## 💭 Recent Memories (Episodic Memory)")
            total_episodic = memory.episodic.count()
            lines.append(f"Total episodic memories: {total_episodic}")

            recent = memory.episodic.get_recent(n=max_memories)
            if recent:
                for mem in recent[:max_memories]:
                    ts = mem.get("timestamp", "")[:19]
                    imp = mem.get("importance", 0)
                    content_preview = mem["content"][:100]
                    if len(mem["content"]) > 100:
                        content_preview += "..."
                    lines.append(
                        f"\n- [{ts}] (importance: {imp:.0%})\n  {content_preview}"
                    )
            else:
                lines.append("  (no episodic memories stored yet)")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"summarise_memories() failed: {e}", exc_info=True)
            return f"Error generating summary: {str(e)}"

    # ════════════════════════════════════════════════════════════════════════
    # TOOL 5: list_entities()
    # ════════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def list_entities() -> str:
        """
        Return all known entities and their structured facts from semantic memory.

        Use this to see what MemoryOS knows about the user, their projects,
        and their tech stack. Returns all entities with confidence ≥ 60%.

        Returns:
            Formatted list of all entities and their attributes.
        """
        logger.info("list_entities() called")

        try:
            entities = memory.semantic.get_all_entities()

            if not entities:
                return "No entities in semantic memory yet."

            lines = [f"Known entities ({len(entities)} total):\n"]

            for entity_name in entities:
                facts = memory.semantic.get_entity_facts(entity_name)
                # Filter to reasonably confident facts
                facts = [f for f in facts if f["confidence"] >= 0.6]

                if not facts:
                    continue

                lines.append(f"**{entity_name.upper()}**")
                for fact in facts:
                    pct = int(fact["confidence"] * 100)
                    conf_bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
                    lines.append(
                        f"  {fact['attribute']:20s} = {fact['value']!r:25s} "
                        f"[{conf_bar}] {pct}%"
                    )
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"list_entities() failed: {e}", exc_info=True)
            return f"Error listing entities: {str(e)}"

    # ════════════════════════════════════════════════════════════════════════
    # TOOL 6: store_conversation_turn() — helper called by the agent loop
    # ════════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def store_conversation_turn(
        role: str,
        content: str,
        auto_extract: bool = True,
    ) -> str:
        """
        Store a single conversation turn and optionally extract facts from it.

        This is called by the agent loop (ollama_agent.py) after every
        message exchange. It:
        1. Adds the turn to working memory
        2. Extracts entities and tags the turn
        3. Upserts any high-confidence facts to semantic memory
        4. If importance is high enough, stores to episodic memory too

        Args:
            role:         "user" or "assistant"
            content:      the message text
            auto_extract: whether to run entity extraction (default: True)

        Returns:
            Summary of what was stored.
        """
        logger.debug(f"store_conversation_turn(): role={role}, content={content[:50]!r}")

        entities = []
        facts_stored = 0

        if auto_extract and role == "user":
            # Extract entities for working memory tagging
            entities = extractor.get_entity_strings(content)

            # Extract and store facts in semantic memory
            facts = extractor.extract_facts(content, role=role)
            for fact in facts:
                if fact["confidence"] >= 0.70:
                    memory.semantic.upsert_fact(
                        entity=fact["entity"],
                        attribute=fact["attribute"],
                        value=fact["value"],
                        confidence=fact["confidence"],
                        source="conversation_turn",
                    )
                    facts_stored += 1

        # Add to working memory
        turn_id = memory.working.add_turn(role, content, entities)

        # Decide if this turn is important enough for episodic storage
        importance = extractor.compute_importance(content, [])
        if importance >= 0.6:
            session_id = memory.working.get_session_summary()["session_id"]
            memory.episodic.store(
                content=f"{role.title()}: {content}",
                importance=importance,
                source="conversation_turn",
                session_id=session_id,
                entities=entities,
            )

        return (
            f"Stored turn {turn_id[:8]} (role={role}, "
            f"entities={entities or 'none'}, "
            f"facts_extracted={facts_stored}, "
            f"importance={importance:.2f})"
        )

    logger.info("All MemoryOS tools registered with MCP server.")
