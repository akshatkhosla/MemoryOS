# MemoryOS 🧠
### A Persistent Memory Layer for LLMs, Built as an MCP Server

MemoryOS gives local LLMs a memory system that persists across sessions. Every conversation is stored, searchable, and loaded back automatically the next time you start a chat. It runs entirely on your machine with no API keys, no cloud services, and no costs.

```
You: "Hi, I'm Akshat. I'm a software engineer building MemoryOS in Python."
     [session ends]

     [new session, days later]

You: "What do you know about me?"
AI:  "You're Akshat, a software engineer based in Hyderabad.
      You're building a project called MemoryOS using Python,
      ChromaDB, and SQLite."
```

That's the problem this solves. Standard LLMs are stateless. Every session starts from zero. MemoryOS fixes that.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [How It Works: Core Concepts](#how-it-works)
   - [MCP: Model Context Protocol](#mcp-model-context-protocol)
   - [Three Memory Tiers](#three-memory-tiers)
   - [Entity Extraction](#entity-extraction)
   - [Confidence Scores](#confidence-scores)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Tech Stack](#tech-stack)
6. [Setup and Installation](#setup-and-installation)
7. [Running MemoryOS](#running-memoryos)
8. [Chat Commands](#chat-commands)
9. [MCP Tools Reference](#mcp-tools-reference)
10. [Connecting to Claude Desktop](#connecting-to-claude-desktop)
11. [Maintenance and Debugging](#maintenance-and-debugging)

---

## The Problem

Every time you start a new conversation with a local LLM, it has zero memory of you. You repeat yourself constantly:

- "My name is Akshat" (said in every session)
- "I'm a software engineer" (again)
- "I prefer Python" (again)
- "The bug I was debugging last week was..." and the LLM has no idea

This is not a model intelligence problem. It is a memory architecture problem. LLMs have no persistent storage by default. Their context window disappears when the process exits.

MemoryOS solves this by sitting between you and the LLM as a memory layer. It automatically stores what you say, extracts structured facts, and injects relevant context back at the start of every new session.

---

## How It Works

### MCP: Model Context Protocol

MCP is an open standard protocol (by Anthropic, MIT licensed) that lets any application expose tools to any LLM client over a simple JSON-RPC interface. Think of it as a USB standard for LLM tools: build once, connect anywhere.

**How MCP differs from raw LLM tool use:**

With raw tool use, you hardcode tool schemas into every API call, manually parse tool call blocks in responses, and route them yourself with if/elif chains. Everything is tightly coupled to one LLM provider.

With MCP, the server declares its tools once. Any MCP-compatible client (Claude Desktop, Cursor, your own chat app) connects and discovers them automatically. The protocol handles routing. The LLM decides when to call tools based on the schemas it receives.

**The MCP handshake sequence:**

```
CLIENT                                    SERVER (server.py)
  │                                            │
  │──── initialize ──────────────────────────> │
  │     {protocolVersion, clientInfo}          │
  │                                            │
  │<─── initialized ─────────────────────────  │
  │     {capabilities: {tools: {}}}            │
  │                                            │
  │──── tools/list ─────────────────────────>  │
  │                                            │
  │<─── [{name, description, inputSchema}] ──  │
  │                                            │
  │──── tools/call {name:"recall", args} ───>  │
  │                                            │
  │<─── {content: [{type:"text", text:"..."}]} │
```

**Transport: why stdio**

MemoryOS uses stdio transport. The MCP client spawns `server.py` as a subprocess and communicates through stdin/stdout pipes using JSON-RPC 2.0 messages. No ports, no network stack, no auth. Just two processes talking through pipes. This is why `server.py` never prints to stdout (that would corrupt the JSON-RPC stream). All logging goes to a file and stderr.

---

### Three Memory Tiers

MemoryOS uses three different storage backends, each optimised for a different type of memory:

```
┌──────────────────────────────────────────────────────────────────┐
│                      THREE MEMORY TIERS                          │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  TIER 1: WORKING MEMORY                                   │   │
│  │  Storage: Python dict (in-process)                        │   │
│  │  Scope:   Current session only, wiped on exit             │   │
│  │  Speed:   Instant                                         │   │
│  │  Use:     Active conversation turns, session context      │   │
│  │                                                           │   │
│  │  {"turn_1": "User said: I'm Akshat",                      │   │
│  │   "turn_2": "Assistant: Nice to meet you..."}             │   │
│  └───────────────────────────────────────────────────────────┘   │
│                         │ flush on session end                   │
│                         ▼                                        │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  TIER 2: EPISODIC MEMORY                                  │   │
│  │  Storage: ChromaDB (vector database, persists to disk)    │   │
│  │  Scope:   Permanent across sessions                       │   │
│  │  Search:  Semantic similarity (embedding vectors)         │   │
│  │  Use:     Past events, conversations, narrative context   │   │
│  │                                                           │   │
│  │  "Last week user was debugging a ChromaDB error"          │   │
│  │  "User mentioned they prefer concise answers"             │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  TIER 3: SEMANTIC MEMORY                                  │   │
│  │  Storage: SQLite (relational DB, persists to disk)        │   │
│  │  Scope:   Permanent across sessions                       │   │
│  │  Search:  Exact and fuzzy SQL queries                     │   │
│  │  Use:     Structured facts: name, location, role, etc.    │   │
│  │                                                           │   │
│  │  entity  | attribute   | value             | confidence   │   │
│  │  --------+-------------+-------------------+------------- │   │
│  │  user    | name        | Akshat Khosla     | 0.85         │   │
│  │  user    | location    | Hyderabad         | 0.80         │   │
│  │  user    | role        | software engineer | 0.75         │   │
│  │  project | name        | MemoryOS          | 0.75         │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

**Why three tiers and not one?**

Each storage type excels at a different access pattern:

| Need | Best Storage | Why |
|------|-------------|-----|
| "What did I say 3 messages ago?" | Working (dict) | Zero latency, exact lookup |
| "What were we discussing about auth last month?" | Episodic (ChromaDB) | Semantic search finds it without exact keywords |
| "What is the user's name?" | Semantic (SQLite) | Deterministic exact lookup, always correct |

Vector similarity is the wrong tool for "what is user.name?" because you want a deterministic answer, not the 5 most similar facts. Conversely, SQL is useless for "what did we discuss about authentication?" because you don't know the exact words stored. The three-tier design means each query type hits the right storage engine.

---

### Entity Extraction

When you send a message, MemoryOS runs it through an extraction pipeline before storing it. The goal is to pull out structured facts automatically without you having to explicitly tell the system what to remember.

Two strategies run in parallel:

**1. spaCy NER (Named Entity Recognition)**

spaCy's `en_core_web_sm` model is a trained neural pipeline that identifies named entities in text and classifies them:

```
"Hi, I'm Akshat and I'm based in Hyderabad"
         ↓ spaCy NER
PERSON: "Akshat"    → user.name     = "Akshat"     (85% confidence)
GPE:    "Hyderabad" → user.location = "Hyderabad"  (80% confidence)
```

NER labels that MemoryOS maps to facts:

| spaCy Label | Maps To | Example |
|-------------|---------|---------|
| PERSON | user.name | "Akshat Khosla" |
| GPE (geopolitical entity) | user.location | "Hyderabad", "India" |
| ORG | organization.name | "Microsoft", "Anthropic" |
| PRODUCT | technology.name | "ChromaDB", "Azure" |

**2. Regex Pattern Rules**

spaCy detects what is an entity but not always what it means to the user. Regex patterns catch relationship statements that NER misses:

```
"I'm a software engineer"       → user.role           = "software engineer"
"I prefer Python"               → user.tech_preference = "Python"
"I'm building MemoryOS"         → project.name         = "MemoryOS"
"I use VS Code for development" → user.tool_usage       = "VS Code"
"I have 2 years of experience"  → user.experience_years = "2"
```

Both strategies run on every user message, results are deduplicated (highest confidence wins per entity+attribute pair), and only facts above 70% confidence threshold are written to semantic memory automatically.

---

### Confidence Scores

Every stored fact carries a confidence score (0.0 to 1.0) representing how certain the system is about that fact. These are the percentages shown in `/entities` output.

**Where scores come from:**

| Source | Score | Rationale |
|--------|-------|-----------|
| spaCy PERSON detection | 85% | Neural model, reliable for proper names |
| spaCy GPE detection | 80% | Geographic entities are usually unambiguous |
| spaCy ORG detection | 75% | Organisations are sometimes misidentified |
| Regex pattern match | 75% | Rule-based, reliable but less contextual |
| LLM explicit remember() call | Auto-computed | Based on content importance heuristics |

**How scores update over time:**

When the same fact is seen again in a new session, MemoryOS runs a weighted average rather than overwriting:

```python
new_confidence = (old_confidence × 0.7) + (new_confidence × 0.3)
```

This means repeated observations gradually increase confidence without letting a single noisy observation tank a well-established fact. The `confirmed Nx` counter shown alongside the score is often the more meaningful signal. A fact confirmed 5x at 75% is more trustworthy than one seen once at 85%.

**What scores control:**

- Facts below 60% are hidden from `list_entities` output
- Recall results rank semantic facts by confidence score
- The auto-extraction threshold is 70%. Facts below this are not written automatically

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                       MEMORYOS ARCHITECTURE                            │
│                                                                        │
│  ┌─────────────────┐    messages     ┌──────────────────────────────┐  │
│  │       You       │ ─────────────>  │         chat.py  (CLI)       │  │
│  │   (terminal)    │ <─────────────  │     Rich terminal UI         │  │
│  └─────────────────┘   responses     └──────────────────────────────┘  │
│                                                   │                    │
│                                    ┌──────────────┘                    │
│                                    │                                   │ 
│                                    ▼                                   │
│                       ┌────────────────────────────────────────────┐   │
│                       │         OllamaAgent  (agent loop)          │   │
│                       │                                            │   │
│                       │  1. Store user turn → extract facts        │   │
│                       │  2. Call Ollama with history + tools       │   │
│                       │  3. Handle tool calls → MCPClient          │   │
│                       │  4. Return final response                  │   │
│                       └────────────────────────────────────────────┘   │
│                              │                    │                    │
│                              ▼                    ▼                    │
│                   ┌─────────────────┐  ┌──────────────────────────┐    │
│                   │     Ollama      │  │      MCPClient           │    │
│                   │   (local LLM)   │  │  (direct memory access)  │    │
│                   │   qwen2.5:7b    │  └──────────────────────────┘    │
│                   └─────────────────┘             │                    │
│                                                    ▼                   │
│                                    ┌───────────────────────────────┐   │
│                                    │        MemorySystem           │   │
│                                    │                               │   │
│                                    │  ┌──────────────────────────┐ │   │
│                                    │  │  Working Memory (dict)   │ │   │
│                                    │  ├──────────────────────────┤ │   │
│                                    │  │  Episodic (ChromaDB)     │ │   │
│                                    │  │  ./data/chroma/          │ │   │
│                                    │  ├──────────────────────────┤ │   │
│                                    │  │  Semantic (SQLite)       │ │   │
│                                    │  │  ./data/semantic.db      │ │   │
│                                    │  └──────────────────────────┘ │   │
│                                    │             ▲                 │   │
│                                    │    ┌────────┘                 │   │
│                                    │    │  Extractor               │   │
│                                    │    │  spaCy + regex           │   │
│                                    └────┴──────────────────────────┘   │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  server.py: MCP Server (stdio/JSON-RPC transport)                │  │
│  │  Exposes the same memory tools to external MCP clients:          │  │
│  │  Claude Desktop, Cursor, any MCP-compatible application          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

**Message flow for a single user turn:**

```
1.  You type a message in the terminal
2.  chat.py passes it to OllamaAgent.chat()
3.  Agent calls _store_turn() on the user message
       → spaCy + regex extracts facts
       → facts above 70% confidence written to SQLite
       → turn added to working memory (dict)
       → if importance >= 0.55, stored to ChromaDB too
4.  Agent calls Ollama with:
       system prompt + full conversation history + tool schemas
       options: {num_ctx: 32768}
5.  Ollama responds with either:
       (a) Plain text → display to user, done
       (b) Tool call → go to step 6
6.  MCPClient executes the tool call (recall / remember / list_entities)
7.  Tool result appended to conversation as role="tool" message
8.  Back to step 4. Ollama generates the next response with tool result in context
9.  Final text response displayed in the Rich terminal panel
```

---

## Project Structure

```
memoryos/
│
├── server.py              ← MCP server entry point (stdio/JSON-RPC)
├── tools.py               ← @mcp.tool() definitions and handlers
├── extractor.py           ← spaCy NER + regex fact extraction pipeline
├── inspect_memory.py      ← CLI utility to inspect/wipe stored memories
├── requirements.txt       ← All dependencies with version pins
├── README.md
│
├── memory/
│   ├── __init__.py        ← MemorySystem: unified interface to all tiers
│   ├── working.py         ← Tier 1: in-session Python dict memory
│   ├── episodic.py        ← Tier 2: ChromaDB vector memory
│   └── semantic.py        ← Tier 3: SQLite structured fact storage
│
├── client/
│   ├── chat.py            ← Rich CLI chat interface with /commands
│   └── ollama_agent.py    ← Agent loop: Ollama + memory tool coordination
│
└── data/                  ← Created automatically on first run
    ├── chroma/            ← ChromaDB persistent vector files
    ├── semantic.db        ← SQLite database file
    └── memoryos.log       ← Server and extraction logs
```

**Key files explained:**

**`server.py`**: The MCP server process. External MCP clients (Claude Desktop, Cursor) spawn this as a subprocess and communicate over stdio. It initialises all three memory tiers and registers tools via the MCP SDK. You do not run this directly when using the chat client.

**`tools.py`**: Defines the 6 MCP tools using `@mcp.tool()` decorators. The MCP SDK auto-generates JSON schemas from Python type hints and handles all routing with no manual if/elif dispatch needed.

**`extractor.py`**: The NLP pipeline. Runs on every user message automatically. Combines spaCy NER for entity detection with regex patterns for relationship extraction. Includes a blocklist of tech terms to prevent false PERSON matches (e.g. "Python" being stored as a name).

**`ollama_agent.py`**: The agent loop. Manages conversation history, calls Ollama with the correct context window, handles the tool call/response cycle, and calls `_store_turn()` on every message. Also coerces LLM-supplied arguments to correct types (the LLM sometimes passes numbers as strings).

**`inspect_memory.py`**: Standalone debugging utility. Run directly from the terminal to see everything stored, delete individual bad facts, or wipe all memory for a fresh start.

---

## Tech Stack

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| MCP protocol | `mcp` (official SDK) | ≥1.0.0 | JSON-RPC server over stdio |
| Local LLM | Ollama + qwen2.5:7b | latest | Language model inference |
| Vector memory | ChromaDB | ≥0.4.0 | Episodic memory vector store |
| Embeddings | SentenceTransformers | ≥3.0.0 | Semantic similarity computation |
| Embedding model | all-MiniLM-L6-v2 | built-in | 384-dim vectors, ~80MB on disk |
| Structured facts | SQLite | stdlib | Semantic memory relational store |
| Entity extraction | spaCy en_core_web_sm | ≥3.7.0 | NER pipeline |
| Deep learning runtime | PyTorch | ≥2.4.0 | Required by SentenceTransformers |
| Terminal UI | Rich | ≥13.0.0 | Chat interface panels and formatting |

All free. All local. No API keys required.

---

## Setup and Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- 8GB+ RAM (16GB recommended for qwen2.5:7b with 32k context)
- ~5GB free disk space (model weights + ChromaDB + dependencies)

### Step 1: Install Ollama and pull a model

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model
ollama pull qwen2.5:7b

# Start the Ollama server (keep this running in a separate terminal)
ollama serve
```

> **Why qwen2.5:7b?**
> It has the best tool-calling reliability among local models at this size. It consistently passes arguments with correct types, proactively calls `remember()` when you share information, and handles multi-turn tool sequences correctly. Models like llama3.2 often respond conversationally instead of calling tools, and pass numeric arguments as strings which causes runtime errors.

### Step 2: Clone the repository

```bash
git clone <repo-url>
cd memoryos
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### Step 4: Install Python dependencies

```bash
pip install -r requirements.txt
```

If you see `PyTorch >= 2.4 is required but found 2.2.x`, run this instead:

```bash
pip uninstall torch sentence-transformers transformers -y
pip install "torch>=2.4.0"
pip install "transformers>=4.41.0,<4.48.0"
pip install "sentence-transformers>=3.0.0"
```

### Step 5: Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### Step 6: Verify the embedding model

```bash
python -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
print('Shape:', m.encode('test').shape)
print('Episodic memory: ready')
"
# Expected: Shape: (384,)
```

If this fails, episodic memory (ChromaDB vector search) will be disabled and only semantic facts will persist across sessions.

---

## Running MemoryOS

### Start the chat client

```bash
# Recommended model
python client/chat.py --model qwen2.5:7b

# If RAM is limited
python client/chat.py --model qwen2.5:3b
```

On first launch with no prior sessions:
```
No previous memories found. Starting fresh.

You: Hi, my name is Akshat Khosla...
```

On subsequent launches:
```
📋 Loaded from Memory
  USER:
    name: 'Akshat Khosla' (85%)
    location: 'Hyderabad' (80%)
    role: 'software engineer' (75%)
  PROJECT:
    name: 'MemoryOS' (75%)

You:
```

### Set the Context Window (Important)

The default Ollama context window of 4k tokens is too small for MemoryOS. Tool schemas alone consume ~400 tokens, and memory recall results add hundreds more. At 4k you hit the limit after 5-6 exchanges and Ollama silently truncates conversation history.

Open `client/ollama_agent.py` and add `options` to the `ollama.chat()` call:

```python
response = ollama.chat(
    model=self._model,
    messages=[
        {"role": "system", "content": self.SYSTEM_PROMPT},
        *self._conversation_history,
    ],
    tools=MEMORY_TOOLS,
    options={"num_ctx": 32768},    # add this line
)
```

Or bake it into a custom Ollama model:

```bash
ollama show qwen2.5:7b --modelfile > Modelfile
# Add to Modelfile:  PARAMETER num_ctx 32768
ollama create memoryos-brain -f Modelfile
python client/chat.py --model memoryos-brain
```

---

## Chat Commands

| Command | Description |
|---------|-------------|
| `/entities` | All structured facts in semantic memory with confidence scores |
| `/memory` | Full summary across all three tiers |
| `/recall <query>` | Semantic search: find memories related to any topic |
| `/forget <id>` | Delete a specific memory by ID (IDs shown in /memory output) |
| `/reset` | Clear current conversation history (persistent memory is kept) |
| `/status` | Quick stats: memory counts, session turn count |
| `/help` | Show all commands |
| `/quit` | Exit. All memories save automatically |

---

## MCP Tools Reference

These are the tools MemoryOS exposes. The LLM calls them automatically during conversation. You can also trigger most of them via the slash commands above.

### `remember(content, importance?)`
Store a new memory. Auto-classifies into the appropriate tier based on content structure.
- `content`: text to remember. Can be a fact, event, preference, or anything.
- `importance`: 0.0 to 1.0. Auto-computed from content if omitted.

### `recall(query, top_k?, min_importance?)`
Semantic search across all memory tiers. Returns results ranked by relevance × importance.
- `query`: natural language search query
- `top_k`: maximum results to return (default: 5)

### `forget(memory_id)`
Delete a specific memory by its ID. IDs are the short strings shown in `/memory` and `/recall` output (e.g. `a3f2c1b8`).

### `summarise_memories(max_memories?)`
Full overview of stored memory: session state, all semantic facts, and recent episodic memories.

### `list_entities()`
All structured facts from semantic memory grouped by entity, filtered to facts with confidence ≥ 60%.

### `store_conversation_turn(role, content)` *(internal)*
Called automatically by the agent loop after every user message. Runs the extraction pipeline and stores the turn. Not exposed to the LLM.

---

## Connecting to Claude Desktop

`server.py` is a fully functional MCP server that any MCP-compatible client can connect to. To use MemoryOS with Claude Desktop:

**Step 1: Get your absolute project path**

```bash
cd /path/to/memoryos && pwd
# e.g. /Users/akshat/Projects/memoryos
```

**Step 2: Edit the Claude Desktop config**

```bash
# macOS
open ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Windows
# %APPDATA%\Claude\claude_desktop_config.json
```

**Step 3: Add the MemoryOS server**

```json
{
  "mcpServers": {
    "memoryos": {
      "command": "python",
      "args": ["/Users/akshat/Projects/memoryos/server.py"]
    }
  }
}
```

**Step 4: Restart Claude Desktop**

MemoryOS will appear in Claude's available tools. Claude can now call `remember`, `recall`, `list_entities`, and `forget`. Your memories persist across every Claude Desktop conversation, stored locally in `./data/`.

---

## Maintenance and Debugging

### Inspect what is stored

```bash
python inspect_memory.py
```

Output shows all semantic facts with confidence scores and sources, the 20 most recent episodic memories with importance scores, and total counts per tier.

### Delete a specific wrong fact

```bash
python inspect_memory.py --delete-fact user.name
python inspect_memory.py --delete-fact project.name
python inspect_memory.py --delete-fact user.tech_preference
```

### Wipe all memories and start fresh

```bash
python inspect_memory.py --wipe
# Prompts for confirmation
```

### Wipe only one tier

```bash
python inspect_memory.py --wipe-semantic    # SQLite facts only
python inspect_memory.py --wipe-episodic    # ChromaDB vectors only
```

### Follow the logs

```bash
tail -f data/memoryos.log
```

### Common issues and fixes

**spaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**PyTorch version conflict (episodic memory disabled):**
```bash
pip uninstall torch sentence-transformers transformers -y
pip install "torch>=2.4.0"
pip install "transformers>=4.41.0,<4.48.0"
pip install "sentence-transformers>=3.0.0"
```

**Episodic memory count stays at 0:**
Run the embedding verification command from Step 6 of setup. It must print `Shape: (384,)`.

**LLM not calling memory tools (just replying conversationally):**
Use `qwen2.5:7b`. Set context window to 32k (see above). Smaller models and shorter context windows both reduce tool-calling reliability.

**Wrong facts stored (e.g. `user.name = 'Python'`):**
```bash
python inspect_memory.py --delete-fact user.name
```
Then re-state clearly: *"My name is Akshat Khosla"*.

---

## Known Limitations

**Entity extraction is heuristic-based.** The spaCy and regex pipeline works well for clear first-person statements but can misfire on unusual phrasing or long complex sentences. If a wrong fact gets stored, use `inspect_memory.py --delete-fact` to remove it and re-state the correct information.

**No contradiction detection.** If you say "I'm based in Mumbai" after previously storing "I'm based in Hyderabad", MemoryOS updates the value but does not flag the contradiction or reduce confidence on conflicting facts.

**Episodic memory grows unboundedly.** Over many months of use, ChromaDB will accumulate memories without pruning. A future improvement would be automatic consolidation, where the LLM periodically summarises old low-importance memories into compact high-importance ones.

**Tool-calling quality depends on the model.** Models smaller than 7B parameters will sometimes ignore memory tools and respond conversationally. `qwen2.5:7b` is the minimum recommended size for consistent tool use.

**Context window management is manual.** MemoryOS does not yet automatically trim conversation history when approaching the context limit. For very long sessions, use `/reset` to clear conversation history while keeping all persistent memories intact.

---