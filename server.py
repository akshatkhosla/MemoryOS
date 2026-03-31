"""
server.py — MemoryOS MCP Server Entry Point
════════════════════════════════════════════

This is the process that clients (Claude Desktop, your chat.py, Cursor, etc.)
spawn and communicate with over stdio using JSON-RPC 2.0.

HOW TO RUN:
  python server.py

  But you don't usually run this directly — the MCP client spawns it:
  In chat.py:
      process = subprocess.Popen(["python", "server.py"], stdin=PIPE, stdout=PIPE)
  Or in Claude Desktop's config:
      {"command": "python", "args": ["path/to/server.py"]}

MCP SERVER LIFECYCLE:
  1. Process starts
  2. MCP SDK reads from stdin, writes to stdout (stdio transport)
  3. Client sends: initialize → server responds with capabilities
  4. Client sends: tools/list → server responds with tool schemas
  5. Client calls tools as needed
  6. Process exits when stdin closes (client disconnected)

WHY STDIO (not HTTP)?
  - No port conflicts, no firewall issues
  - The client fully controls the server lifetime
  - Works everywhere Python works — no network stack needed
  - Perfect for local personal tools like MemoryOS

JSON-RPC 2.0 MESSAGE FORMAT (what flows over stdio):
  Request:  {"jsonrpc":"2.0","id":1,"method":"tools/call",
              "params":{"name":"remember","arguments":{"content":"..."}}}
  Response: {"jsonrpc":"2.0","id":1,
              "result":{"content":[{"type":"text","text":"Stored..."}]}}

  The MCP SDK handles all of this serialization — you never see raw JSON.
"""

import asyncio
import logging
import sys
import os

# Add project root to Python path (so imports work when spawned as subprocess)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MCP SDK imports
# mcp.server.Server: the core server class
# mcp.server.stdio: the stdio transport handler
# mcp.types: type definitions for MCP protocol objects
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Our modules
from memory import MemorySystem
from extractor import Extractor
from tools import register_tools

# ── Logging Configuration ────────────────────────────────────────────────────
# IMPORTANT: Do NOT log to stdout in an MCP server!
# stdout is used for JSON-RPC messages. Any non-JSON output will break
# the protocol. Use a file for logs, or stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        # Log to file — stdout is sacred for JSON-RPC
        logging.FileHandler("./data/memoryos.log"),
        # Also log to stderr (visible in terminal but doesn't break stdio)
        logging.StreamHandler(sys.stderr),
    ],
)
logger = logging.getLogger("memoryos.server")

# ── Server Identity ──────────────────────────────────────────────────────────
# This information is sent to the client during the initialize handshake.
# The client uses it to identify which server it's talking to.
SERVER_NAME = "MemoryOS"
SERVER_VERSION = "1.0.0"


async def main():
    """
    Initialize and run the MemoryOS MCP server.

    async because:
    1. The MCP SDK is fully async (built on asyncio)
    2. All our tool handlers are async functions
    3. stdio I/O is non-blocking with asyncio
    """
    logger.info("=" * 60)
    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
    logger.info("=" * 60)

    # ── Ensure data directory exists ──────────────────────────────────────
    os.makedirs("./data", exist_ok=True)

    # ── Initialize MCP Server ─────────────────────────────────────────────
    # Server(name) creates the server and sets its identity for the handshake.
    # The name is what shows up in Claude Desktop's connected servers list.
    mcp = Server(SERVER_NAME)

    # ── Initialize Memory System ─────────────────────────────────────────
    # This starts ChromaDB, loads embeddings model, and opens SQLite.
    # May take 2-5 seconds on first run (model download).
    logger.info("Initializing memory system...")
    memory = MemorySystem(data_dir="./data")
    logger.info(f"Memory system ready: {memory.get_status()}")

    # ── Initialize Extractor ─────────────────────────────────────────────
    # Loads spaCy model. Fast after first download.
    logger.info("Initializing entity extractor...")
    extractor = Extractor()
    logger.info("Extractor ready.")

    # ── Register Tools ───────────────────────────────────────────────────
    # This wires up all @mcp.tool() handlers defined in tools.py
    register_tools(mcp, memory, extractor)
    logger.info("Tools registered.")

    # ── Start stdio Transport ─────────────────────────────────────────────
    #
    # stdio_server() is a context manager that:
    # 1. Wraps sys.stdin/sys.stdout as async streams
    # 2. Starts the JSON-RPC read loop on stdin
    # 3. Handles the initialize handshake automatically
    # 4. Routes incoming tool calls to our handlers
    # 5. Serializes responses and writes to stdout
    #
    # mcp.run() is the main event loop. It runs until stdin closes
    # (i.e., until the client disconnects or kills the process).
    #
    # InitializationOptions: tells the SDK our server name and version
    # for the handshake response.

    logger.info("Server ready. Waiting for client connections on stdio...")
    logger.info("(Tip: start chat.py in another terminal to connect)")

    from mcp.server.models import InitializationOptions
    import mcp.types as types

    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=mcp.get_capabilities(
                    # Declare which MCP features we support.
                    # notification_options: we don't push notifications to client
                    # experimental_capabilities: no experimental features
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )

    logger.info("Server shutting down. Goodbye.")


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # asyncio.run() creates an event loop, runs main(), then cleans up.
    # This is the standard way to run async Python programs.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt — shutting down.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
