# MemoryOS 🧠

**A persistent memory layer for LLMs, built as an MCP (Model Context Protocol) server.**

Fully free and open source — runs 100% locally with Ollama.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORYOS ARCHITECTURE                        │
│                                                                  │
│  ┌──────────────┐   chat.py            ┌──────────────────────┐│
│  │  You (human) │ ──────────────────>  │   OllamaAgent        ││
│  │              │ <──────────────────  │   (agent loop)       ││
│  └──────────────┘   response          └──────────────────────┘│
│                                                 │               │
│                             ┌───────────────────┤               │
│                             │                   │               │
│                             ▼                   ▼               │
│                    ┌─────────────┐   ┌──────────────────────┐  │
│                    │   Ollama    │   │  MemoryOS MCP Server  │  │
│                    │  (local LLM)│   │  (server.py)          │  │
│                    │  llama3.2   │   │                        │  │
│                    └─────────────┘   │  ┌─────────────────┐  │  │
│                                      │  │ Working Memory  │  │  │
│                                      │  │ (Python dict)   │  │  │
│                                      │  ├─────────────────┤  │  │
│                                      │  │ Episodic Memory │  │  │
│                                      │  │ (ChromaDB)      │  │  │
│                                      │  ├─────────────────┤  │  │
│                                      │  │ Semantic Memory │  │  │
│                                      │  │ (SQLite)        │  │  │
│                                      │  └─────────────────┘  │  │
│                                      └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Tiers

| Tier | Storage | Scope | Search | Use Case |
|------|---------|-------|--------|----------|
| **Working** | Python dict | Session only | In-memory | Current conversation turns |
| **Episodic** | ChromaDB vectors | Persistent | Semantic similarity | Past events & narratives |
| **Semantic** | SQLite facts | Persistent | SQL exact/fuzzy | Structured facts about user |

## MCP Tools

| Tool | Description |
|------|-------------|
| `remember(content, importance)` | Store a memory, auto-classify tier |
| `recall(query, top_k)` | Semantic search across all tiers |
| `forget(memory_id)` | Delete a specific memory |
| `summarise_memories()` | Overview of all stored memory |
| `list_entities()` | All structured facts |
| `store_conversation_turn(role, content)` | Auto-store + extract facts from a turn |

---

## Setup

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (choose one):
ollama pull llama3.2    # recommended — fast, good tool use
ollama pull mistral     # good alternative
```

### 2. Clone and install dependencies
```bash
git clone <repo>
cd memoryos
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the chat client
```bash
# Start chatting (automatically uses Ollama + MemoryOS):
python client/chat.py

# Use a different model:
python client/chat.py --model mistral
```

### 4. (Optional) Run the MCP server standalone
For use with Claude Desktop, Cursor, or other MCP clients:
```bash
python server.py
```

Configure in Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "memoryos": {
      "command": "python",
      "args": ["/absolute/path/to/memoryos/server.py"]
    }
  }
}
```

---

## How MCP Works (vs. Raw API Tool Use)

If you've used Claude's API tool use (like in DocMind), here's the key difference:

**DocMind approach** — you manually:
1. Define tool schemas as dicts in each API call
2. Parse `tool_use` blocks in responses
3. Route to Python functions with `if tool_name == "search": ...`
4. Format results and send back

**MCP approach** — the protocol handles it:
1. Server declares tools once via `@mcp.tool()` decorators
2. Any MCP client calls `tools/list` to discover them automatically
3. The SDK routes `tools/call` requests to your handlers automatically
4. Your handlers just return strings — no serialization needed

The result: your MemoryOS tools work with Claude Desktop, Cursor, your custom chat.py, or any future MCP client — without changing server.py.

---

## Project Structure

```
memoryos/
├── server.py           ← MCP server entry point (stdio transport)
├── tools.py            ← @mcp.tool() definitions and handlers
├── extractor.py        ← spaCy NER + regex fact extraction
├── requirements.txt
├── README.md
├── memory/
│   ├── __init__.py     ← MemorySystem (unified interface)
│   ├── working.py      ← In-session dict memory
│   ├── episodic.py     ← ChromaDB vector memory
│   └── semantic.py     ← SQLite structured facts
├── client/
│   ├── chat.py         ← Rich CLI chat interface
│   └── ollama_agent.py ← Ollama ↔ MCP agent loop
└── data/               ← Created automatically
    ├── chroma/         ← ChromaDB files
    ├── semantic.db     ← SQLite database
    └── memoryos.log    ← Server logs
```

---

## Chat Commands

| Command | Description |
|---------|-------------|
| `/memory` | Show all stored memories |
| `/entities` | Show all known facts |
| `/recall <query>` | Search memory |
| `/forget <id>` | Delete a memory |
| `/reset` | Clear conversation history |
| `/status` | Memory system stats |
| `/help` | Show help |
| `/quit` | Exit |

---

## Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| MCP protocol | `mcp` (official SDK) | JSON-RPC server over stdio |
| Local LLM | Ollama + llama3.2 | Language model inference |
| Vector memory | ChromaDB | Episodic memory store |
| Embeddings | SentenceTransformers | Semantic similarity |
| Structured facts | SQLite | Semantic memory store |
| Entity extraction | spaCy en_core_web_sm | NER + fact extraction |
| Terminal UI | Rich | Chat interface |

All free. All local. No API keys.
