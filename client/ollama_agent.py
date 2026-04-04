"""
client/ollama_agent.py — Ollama ↔ MCP Agent Loop
══════════════════════════════════════════════════

CONCEPT:
  The agent loop is the intelligence coordinator. It:
  1. Takes user input
  2. Builds context (injecting relevant memories from MCP)
  3. Calls Ollama (local LLM) with that context
  4. Parses the response — did Ollama request a tool call?
  5. If yes: calls the MCP server, gets result, feeds back to Ollama
  6. Repeats until Ollama gives a final response

ARCHITECTURE:
  ┌──────────────┐   user message    ┌──────────────────────────┐
  │  chat.py     │ ───────────────>  │  OllamaAgent             │
  │  (UI layer)  │ <───────────────  │  (this file)             │
  └──────────────┘   final response  │                          │
                                     │  1. Build context        │
                                     │  2. Call Ollama          │
                                     │  3. Parse tool requests  │
                                     │  4. Call MCP tools       │
                                     │  5. Feed results back    │
                                     └──────────────────────────┘
                                             │         ▲
                                      LLM calls        │
                                             ▼         │
                                     ┌───────────────────────┐
                                     │  Ollama (local LLM)   │
                                     │  qwen2.5:7b / others  │
                                     └───────────────────────┘
                                             │         ▲
                                      MCP tool calls   │
                                             ▼         │
                                     ┌───────────────────────┐
                                     │  MCP Server           │
                                     │  (server.py process)  │
                                     │  running over stdio   │
                                     └───────────────────────┘

TOOL CALLING WITH OLLAMA:
  Ollama supports the same tool-calling format as OpenAI:
  - You pass tools as JSON schema in the API request
  - Ollama returns tool_calls in the response
  - You execute the tools and append results as "tool" role messages
  - Continue the conversation

  This is almost identical to how DocMind worked with Claude's API,
  but running entirely locally with Ollama!
"""

import json
import logging
import subprocess
import sys
import os
from typing import Optional
import ollama

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ── MCP Tool Schemas ─────────────────────────────────────────────────────────
# These are the tool schemas we pass to Ollama.
# They must match what server.py exposes (the MCP SDK validates this).
# Ollama uses these schemas to decide when to call tools and with what args.

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store a new memory in the persistent memory system. "
                "Call this when the user shares important information about themselves, "
                "their projects, preferences, or anything worth remembering for future sessions. "
                "Auto-classifies into working/episodic/semantic tiers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember. Be specific and include context.",
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance from 0.0 (trivial) to 1.0 (critical). Default 0.5.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search memory for information relevant to the current query. "
                "Call this when the user asks about something you might have discussed before, "
                "or when you need context about the user to give a better answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what to search for.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entities",
            "description": (
                "List all known facts about the user and their context "
                "from structured memory. Use at session start to load context."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_memories",
            "description": (
                "Get a full summary of all stored memories across all tiers. "
                "Use when the user asks 'what do you remember about me?' or similar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_memories": {
                        "type": "integer",
                        "description": "Max episodic memories to include (default 20).",
                    },
                },
            },
        },
    },
    # store_conversation_turn is intentionally NOT exposed to the LLM.
    # It is an internal tool called directly by the agent loop in chat().
    # Exposing it caused the LLM to call it incorrectly or with missing args.
]


# ── MCP Client ───────────────────────────────────────────────────────────────

class MCPClient:
    """
    Simple MCP client that communicates with server.py over stdio.

    For our use case, we use a direct approach: we call the MCP server's
    Python functions directly by importing them — this avoids the complexity
    of managing a subprocess for the client demo.

    For production/Claude Desktop integration, you'd use the full stdio
    subprocess approach. This inline version makes the demo easy to run.
    """

    def __init__(self):
        # Import memory system directly (same process as agent for simplicity)
        # In a real deployment, these would be subprocess + stdio pipes
        from memory import MemorySystem
        from extractor import Extractor
        self._memory = MemorySystem(data_dir="./data")
        self._extractor = Extractor()

        # Import tool functions directly
        # We create lightweight wrappers that call the same logic as tools.py
        self._setup_direct_tools()

    def _setup_direct_tools(self):
        """
        Set up direct (non-subprocess) tool wrappers.

        In this demo architecture, the agent and memory system run in the
        same Python process. The "MCP protocol" layer is abstracted.
        The server.py provides the real stdio-based MCP server for
        external clients (Claude Desktop, Cursor, etc.).
        """
        from memory import MemorySystem
        from extractor import Extractor

        # These are the same operations as tools.py, called directly
        self._mem = self._memory
        self._ext = self._extractor

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """
        Call a MemoryOS tool and return its result as a string.

        In the full MCP implementation this sends a JSON-RPC request
        over stdio to server.py. Here we call the logic directly.
        """
        logger.debug(f"MCP tool call: {tool_name}({arguments})")

        try:
            if tool_name == "remember":
                return await self._remember(**arguments)
            elif tool_name == "recall":
                return await self._recall(**arguments)
            elif tool_name == "list_entities":
                return await self._list_entities()
            elif tool_name == "summarise_memories":
                return await self._summarise_memories(**arguments)
            elif tool_name == "store_conversation_turn":
                # Guard: LLM sometimes calls this without required args.
                # Silently skip rather than crash — the turn is already in
                # working memory from the explicit add_turn() call in chat().
                if "role" not in arguments or "content" not in arguments:
                    return "Skipped: missing role or content argument."
                return await self._store_turn(**arguments)
            elif tool_name == "forget":
                return await self._forget(**arguments)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Tool call failed: {e}", exc_info=True)
            return f"Tool error: {str(e)}"

    async def _remember(self, content: str, importance=0.5, tier=None) -> str:
        # The LLM sometimes passes importance as a string ("0.8") instead of float.
        # Always coerce to float defensively — this prevents the :.0% format crash.
        try:
            importance = float(importance)
        except (TypeError, ValueError):
            importance = 0.5

        entities = self._ext.extract_entities(content)
        entity_strings = [e["text"] for e in entities]

        # Only auto-compute importance if the caller left it at the default
        if importance == 0.5:
            importance = self._ext.compute_importance(content, entities)

        tier = tier or self._ext.classify_memory_tier(content)

        # Always try to extract and store structured facts into semantic memory,
        # regardless of which tier the content is routed to.
        # This is the key fix for "/entities showing empty" — facts were only being
        # extracted on the episodic path, so semantic-tier content never got stored.
        facts_stored = []
        for fact in self._ext.extract_facts(content, role="user"):
            if fact["confidence"] >= 0.75:
                self._mem.semantic.upsert_fact(
                    fact["entity"], fact["attribute"], fact["value"], fact["confidence"]
                )
                facts_stored.append(f"{fact['entity']}.{fact['attribute']}={fact['value']!r}")

        if tier == "semantic" and facts_stored:
            return f"Stored semantic facts: {', '.join(facts_stored)}"

        # Store to episodic memory
        session_id = self._mem.working.get_session_summary()["session_id"]
        mid = self._mem.episodic.store(
            content, importance, session_id=session_id, entities=entity_strings
        )
        facts_note = f", facts: {', '.join(facts_stored)}" if facts_stored else ""
        return f"Stored episodic memory [{mid[:8]}] (importance: {importance:.0%}{facts_note})"

    async def _recall(self, query: str, top_k=5, min_importance=0.0) -> str:
        # Coerce LLM-supplied args — may arrive as strings
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = 5
        try:
            min_importance = float(min_importance)
        except (TypeError, ValueError):
            min_importance = 0.0
        results = []
        for fact in self._mem.semantic.search_facts(query)[:top_k]:
            results.append({"tier": "semantic", "content": f"{fact['entity']}.{fact['attribute']} = {fact['value']!r}", "score": fact["confidence"]})
        for mem in self._mem.episodic.recall(query, top_k, min_importance):
            # Format episodic memories to make clear these are EXACT stored content
            # Strip "User said: " prefix to show just what was stored
            content = mem["content"]
            if content.startswith("User said: "):
                content = content[len("User said: "):]
            results.append({"tier": "episodic", "content": content, "score": mem["relevance_score"]})
        if not results:
            return f"No memories found for: {query!r}"
        results.sort(key=lambda r: r["score"], reverse=True)
        lines = [f"Found {len(results[:top_k])} memories for: {query!r}\n(These are EXACT stored memories — do not elaborate or infer details)"]
        for i, r in enumerate(results[:top_k], 1):
            lines.append(f"{i}. [{r['tier'].upper()}] ({r['score']:.0%}) {r['content']}")
        return "\n".join(lines)

    async def _list_entities(self) -> str:
        entities = self._mem.semantic.get_all_entities()
        if not entities:
            return "No entities in semantic memory."
        
        # Format as natural language facts with confidence scores
        # This encourages the LLM to respond conversationally while showing certainty
        lines = ["Here's what I know about you from past conversations:"]
        
        for entity in entities:
            facts = self._mem.semantic.get_entity_facts(entity)
            
            # Build natural language for this entity
            if entity == "user":
                user_facts = []
                for f in facts:
                    attr = f['attribute']
                    val = f['value']
                    conf = int(f['confidence'] * 100)
                    
                    # Map attributes to natural language with confidence
                    if attr == "name":
                        user_facts.append(f"Your name is {val} ({conf}%)")
                    elif attr == "location":
                        user_facts.append(f"You're based in {val} ({conf}%)")
                    elif attr == "role":
                        user_facts.append(f"You work as a {val} ({conf}%)")
                    elif attr == "experience_years":
                        user_facts.append(f"You have {val} years of experience ({conf}%)")
                    elif attr == "tech_preference":
                        user_facts.append(f"You prefer using {val} ({conf}%)")
                    elif attr == "tool_usage":
                        user_facts.append(f"You use {val} ({conf}%)")
                    else:
                        user_facts.append(f"{attr}: {val} ({conf}%)")
                
                if user_facts:
                    lines.append("• " + "\n• ".join(user_facts))
            
            elif entity == "project":
                project_facts = []
                for f in facts:
                    attr = f['attribute']
                    val = f['value']
                    conf = int(f['confidence'] * 100)
                    if attr == "name":
                        project_facts.append(f"Project: {val} ({conf}%)")
                    else:
                        project_facts.append(f"{attr}: {val} ({conf}%)")
                if project_facts:
                    lines.append("Projects:\n  • " + "\n  • ".join(project_facts))
            
            elif entity == "technology":
                tech_facts = []
                for f in facts:
                    val = f['value']
                    conf = int(f['confidence'] * 100)
                    tech_facts.append(f"{val} ({conf}%)")
                if tech_facts:
                    lines.append(f"Technologies you mentioned: {', '.join(tech_facts)}")
        
        return "\n".join(lines)

    async def _summarise_memories(self, max_memories: int = 20) -> str:
        status = self._mem.get_status()
        lines = ["=== MemoryOS Summary ==="]
        lines.append(f"Session: {status['working']['session_id'][:8]}...")
        lines.append(f"Episodic memories: {status['episodic']['total_memories']}")
        lines.append(f"Semantic facts: {status['semantic']['total_facts']}")
        lines.append(f"Known entities: {', '.join(status['semantic']['entities']) or 'none'}")
        recent = self._mem.episodic.get_recent(max_memories)
        if recent:
            lines.append("\nRecent memories:")
            for m in recent[:5]:
                lines.append(f"  - {m['content'][:80]}...")
        return "\n".join(lines)

    async def _store_turn(self, role: str, content: str, auto_extract: bool = True) -> str:
        entities = []
        facts_stored = []

        # Only extract facts from USER messages — never assistant responses.
        if auto_extract and role == "user" and len(content.strip()) > 10:
            entities = self._ext.get_entity_strings(content)
            facts = self._ext.extract_facts(content, role="user")

            for fact in facts:
                # Threshold 0.70 — matches what our patterns actually output (0.75)
                # and spaCy NER outputs (0.75-0.85).
                # The previous threshold of 0.80 was silently dropping ALL pattern-based
                # facts because patterns output 0.75, which is below 0.80.
                if fact["confidence"] >= 0.70:
                    self._mem.semantic.upsert_fact(
                        fact["entity"], fact["attribute"], fact["value"],
                        fact["confidence"], source="auto_extraction"
                    )
                    facts_stored.append(f"{fact['entity']}.{fact['attribute']}={fact['value']!r}")

        tid = self._mem.working.add_turn(role, content, entities)

        if role == "user":
            importance = self._ext.compute_importance(content, [])
            # Store to episodic memory if importance >= 0.50
            # This ensures even moderately important content is remembered across sessions.
            # With compute_importance's new technical problem boost (0.35), even a single-line
            # bug report will score 0.85 and definitely get stored.
            if importance >= 0.50:
                session_id = self._mem.working.get_session_summary()["session_id"]
                self._mem.episodic.store(
                    f"User said: {content}",
                    importance,
                    session_id=session_id,
                    entities=entities,
                )
            return (
                f"Stored user turn {tid[:8]} "
                f"(importance={importance:.2f}, "
                f"facts={facts_stored if facts_stored else 'none'})"
            )

        return f"Stored assistant turn {tid[:8]} in working memory only"

    async def _forget(self, memory_id: str) -> str:
        if self._mem.episodic.delete(memory_id):
            return f"Deleted episodic memory {memory_id[:8]}"
        if self._mem.semantic.delete_fact(memory_id):
            return f"Deleted semantic fact {memory_id[:8]}"
        return f"Memory {memory_id[:8]} not found."


# ── Ollama Agent ─────────────────────────────────────────────────────────────

class OllamaAgent:
    """
    Agent loop: takes user input, coordinates Ollama + MCP tools.

    This implements the "ReAct" pattern (Reason + Act):
    - Ollama REASONS about what to do
    - Calls memory ACTIONS (MCP tools) as needed
    - Loops until it has enough context for a final response
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant with persistent memory.
You have access to a memory system that stores real information from past conversations.

MANDATORY RULES — follow these without exception:

1. NEVER answer questions about past conversations, problems, bugs, or events from
   your own knowledge or imagination. ALWAYS call recall() first.

2. When a user asks ANYTHING about:
   - what they told you before
   - past bugs, errors, or technical problems
   - previous conversations or sessions
   - what you remember about them
   You MUST call recall() with a relevant query BEFORE forming any response.

3. After calling recall():
   - If results are found: answer ONLY from what the results say, word for word.
   - If no results are found: say "I don't have any memory of that. Could you remind me?"
   - NEVER fill gaps with guesses, assumptions, or plausible-sounding details.
   - NEVER say "as per our previous conversation" if you have not called recall() first.

4. Do NOT fabricate details. If recall() returns "ChromaDB error", say exactly that.
   Do not add context like "race condition" or "API integration" unless the memory says so.

5. When the user asks "what do you know about me?":
   - At session start, facts are automatically loaded and injected into your context.
   - Read those facts carefully and respond CONVERSATIONALLY, not by repeating the raw data.
   - Instead of: "location: 'Hyderabad' (80%) name: 'Akshat' (75%)"
   - Say: "You're Akshat, a software engineer based in Hyderabad."

6. If facts are loaded but incomplete, acknowledge what you know and ask for missing details.
   Example: "I know you're in Hyderabad, but could you remind me of your exact role?"

7. At the start of each session, facts are loaded automatically. Call remember() when 
   the user shares important NEW facts, bugs, preferences, or events.

The user can verify everything you say against stored memory. Fabricated answers
will be caught immediately. When in doubt, call recall() and report exactly what it returns."""

    MAX_TOOL_ITERATIONS = 5  # prevent infinite tool loops

    def __init__(self, model: str = "qwen2.5:7b"):
        """
        Args:
            model: Ollama model name. Must be pulled first: ollama pull qwen2.5:7b
                   Other options: llama3.2, mistral, llama3.1, gemma2
        """
        self._model = model
        self._mcp = MCPClient()
        self._conversation_history: list[dict] = []

        logger.info(f"OllamaAgent initialized (model={model})")

    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        This is the main agent loop:
        1. Add user message to history
        2. Proactively recall relevant memories (don't rely on LLM to call tools)
        3. Call Ollama with full history + tools
        4. If Ollama requests tools: execute them, add results, loop
        5. Return final text response

        Args:
            user_message: the user's input text

        Returns:
            The agent's final response string
        """
        # Add user message to conversation history
        self._conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # Store this turn in memory (async, fire-and-forget for responsiveness)
        await self._mcp.call_tool("store_conversation_turn", {
            "role": "user",
            "content": user_message,
        })

        # ── Proactive recall ─────────────────────────────────────────────
        # Don't rely on the LLM to call recall() — 7B models frequently skip
        # tool calls and hallucinate answers. We always fetch relevant memories
        # and inject them into context so the LLM has them available.
        recall_result = await self._mcp.call_tool("recall", {
            "query": user_message,
            "top_k": 5,
        })
        if recall_result and "No memories found" not in recall_result:
            self._conversation_history.append({
                "role": "assistant",
                "content": "Let me check my memory for relevant information.",
                "tool_calls": [{
                    "function": {
                        "name": "recall",
                        "arguments": {"query": user_message, "top_k": 5},
                    }
                }],
            })
            self._conversation_history.append({
                "role": "tool",
                "content": (
                    recall_result
                    + "\n\nREMINDER: Only use information from the above memories. "
                    "Do not add details that are not present in these results."
                ),
            })

        # Agent loop: keep calling Ollama until we get a final response
        iterations = 0
        while iterations < self.MAX_TOOL_ITERATIONS:
            iterations += 1

            # ── Call Ollama ──────────────────────────────────────────────
            # ollama.chat() is the Python client's chat endpoint.
            # It's synchronous by default — we call it directly here.
            # For a production async setup, you'd use asyncio.run_in_executor().
            response = ollama.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    *self._conversation_history,
                ],
                tools=MEMORY_TOOLS,
                options={
                    "num_ctx": 32768,   # 32k context window
                },
            )

            message = response["message"]

            # ── Check for tool calls ──────────────────────────────────────
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # No tool calls — this is the final response
                final_response = message["content"]

                # Add assistant response to history
                self._conversation_history.append({
                    "role": "assistant",
                    "content": final_response,
                })

                # Store assistant turn in WORKING MEMORY ONLY (no episodic, no extraction).
                # The assistant's words are NOT facts about the user — storing them to
                # episodic/semantic memory causes false extractions like treating the
                # assistant's own statements as user preferences.
                await self._mcp.call_tool("store_conversation_turn", {
                    "role": "assistant",
                    "content": final_response,
                    "auto_extract": False,
                })

                return final_response

            # ── Execute tool calls ────────────────────────────────────────
            # Ollama requested one or more tool calls.
            # We execute each one and add results to conversation history.

            # Add the assistant's tool-calling message to history
            self._conversation_history.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            })

            # Execute each tool call
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

                # Parse arguments if they're a JSON string
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}

                logger.info(f"Executing tool: {tool_name}({arguments})")

                # Call MCP tool
                tool_result = await self._mcp.call_tool(tool_name, arguments)

                logger.info(f"Tool result: {tool_result[:100]}...")

                # Add tool result to conversation history
                # Ollama expects tool results as role="tool" messages
                self._conversation_history.append({
                    "role": "tool",
                    "content": tool_result,
                })

            # Loop: call Ollama again with tool results in context

        # If we hit MAX_TOOL_ITERATIONS, return what we have
        logger.warning(f"Hit MAX_TOOL_ITERATIONS ({self.MAX_TOOL_ITERATIONS})")
        return "I hit my tool call limit. Here's what I found so far."

    def reset_conversation(self):
        """Clear conversation history (but NOT persistent memory)."""
        self._conversation_history.clear()
        logger.info("Conversation history cleared.")

    async def initialize_session(self) -> str:
        """
        Load memory context at session start and inject it into conversation history.

        This is the critical fix for hallucination. Previously, loaded context was
        only shown in the terminal banner — the LLM never saw it in its message history.
        Now we inject it as the first message in conversation_history so the LLM
        actually has the memory content available when generating responses.

        Two-step load:
        1. list_entities() — structured facts (name, location, role, project)
        2. recall() — recent episodic memories (bugs, discussions, events)

        Returns the context string for display in the terminal banner.
        """
        logger.info("Initializing session — loading memory context")

        # Step 1: load structured facts
        facts = await self._mcp.call_tool("list_entities", {})

        # Step 2: load recent episodic memories broadly
        # Use a broad query so we surface bugs, discussions, and any past events
        recent_memories = await self._mcp.call_tool("recall", {
            "query": "problems bugs errors technical issues discussions preferences",
            "top_k": 5,
        })

        # Build the context string
        context_parts = []
        has_facts = facts and "No entities" not in facts
        has_memories = recent_memories and "No memories found" not in recent_memories

        if has_facts:
            context_parts.append("=== Known Facts About You ===")
            context_parts.append(facts)

        if has_memories:
            context_parts.append("\n=== Recent Memories From Past Sessions ===")
            context_parts.append(recent_memories)

        if not context_parts:
            return "No previous memories found. Starting fresh."

        context_str = "\n".join(context_parts)

        # Inject context as a proper assistant→tool message pair.
        # An orphan role="tool" message (without a preceding assistant tool_call)
        # violates Ollama's expected message format and may be ignored by the model.
        # By adding a synthetic assistant tool_call first, the tool result is properly
        # associated and the model sees the memory content reliably.
        self._conversation_history.append({
            "role": "assistant",
            "content": "Let me load your information from past sessions.",
            "tool_calls": [{
                "function": {
                    "name": "list_entities",
                    "arguments": {},
                }
            }],
        })
        self._conversation_history.append({
            "role": "tool",
            "content": (
                "MEMORY CONTEXT LOADED AT SESSION START:\n"
                + context_str
                + "\n\nIMPORTANT: The above is your ONLY source of truth about past "
                "conversations. Do not add, infer, or elaborate beyond what is written above."
            ),
        })

        return context_str