"""
client/chat.py — MemoryOS CLI Chat Interface
═════════════════════════════════════════════

A rich terminal chat client that connects to OllamaAgent,
which in turn uses MemoryOS for persistent memory.

RUN:
    python client/chat.py

    Or specify a model:
    python client/chat.py --model mistral

SPECIAL COMMANDS:
    /memory   — show all stored memories (summarise_memories)
    /entities — show all known facts (list_entities)
    /recall X — search memory for X
    /forget X — delete memory with ID X
    /reset    — clear conversation (keeps persistent memory)
    /help     — show commands
    /quit     — exit
    /bye      — exit (alias for /quit)
"""

import asyncio
import argparse
import logging
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import print as rprint

from client.ollama_agent import OllamaAgent

# Set up logging (stderr only — don't pollute the chat UI)
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings+ in chat UI
    handlers=[logging.StreamHandler(sys.stderr)],
)

console = Console()


# ── Banner ───────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ███╗   ███╗███████╗███╗   ███╗ ██████╗ ██████╗ ██╗     ║
║   ████╗ ████║██╔════╝████╗ ████║██╔═══██╗██╔══██╗╚██╗    ║
║   ██╔████╔██║█████╗  ██╔████╔██║██║   ██║██████╔╝ ██║    ║
║   ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██║   ██║██╔══██╗ ██║    ║
║   ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║╚██████╔╝██║  ██║ ██║    ║
║   ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═╝    ║
║                                                          ║
║         Persistent Memory Layer for LLMs                 ║
║         Free • Local • Open Source                       ║
╚══════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
[bold cyan]MemoryOS Commands:[/bold cyan]

  [green]/memory[/green]        Show all stored memories
  [green]/entities[/green]      Show all known facts
  [green]/recall <query>[/green] Search memory for something
  [green]/forget <id>[/green]   Delete a memory by ID
  [green]/reset[/green]         Clear conversation (keeps memory)
  [green]/status[/green]        Show memory system status
  [green]/help[/green]          Show this help
  [green]/quit[/green]          Exit
  [green]/bye[/green]           Exit (alias for /quit)

[dim]Tip: Just chat normally — the agent will automatically
store and recall memories as needed.[/dim]
"""


# ── Formatting Helpers ────────────────────────────────────────────────────────

def print_user(text: str):
    console.print(
        Panel(
            text,
            title="[bold blue]You[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def print_assistant(text: str):
    # Render assistant response as Markdown (handles code blocks, lists, etc.)
    console.print(
        Panel(
            Markdown(text),
            title="[bold green]MemoryOS Agent[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )


def print_system(text: str):
    console.print(f"[dim italic]{text}[/dim italic]")


def print_tool_activity(tool_name: str, result_preview: str):
    """Show that a tool was called — gives transparency into agent behavior."""
    console.print(
        f"  [dim]🔧 {tool_name}: {result_preview[:80]}{'...' if len(result_preview) > 80 else ''}[/dim]"
    )


# ── Main Chat Loop ────────────────────────────────────────────────────────────

async def run_chat(model: str = "qwen2.5:7b"):
    """
    Main async chat loop.

    Args:
        model: Ollama model to use (must be pulled first)
    """
    console.print(BANNER, style="cyan")
    console.print(
        f"[dim]Using model: [bold]{model}[/bold] | "
        f"Type /help for commands | /quit or /bye to exit[/dim]\n"
    )

    # ── Initialize agent ──────────────────────────────────────────────────
    print_system("Initializing MemoryOS...")
    try:
        agent = OllamaAgent(model=model)
    except Exception as e:
        console.print(f"[red bold]Failed to initialize agent: {e}[/red bold]")
        console.print("\n[yellow]Checklist:[/yellow]")
        console.print("  1. Is Ollama running?  →  ollama serve")
        console.print(f"  2. Is model pulled?    →  ollama pull {model}")
        console.print("  3. Dependencies?       →  pip install -r requirements.txt")
        console.print("  4. spaCy model?        →  python -m spacy download en_core_web_sm")
        return

    # ── Load session context ──────────────────────────────────────────────
    print_system("Loading memory context...")
    try:
        context_summary = await agent.initialize_session()
        if context_summary and "No entities" not in context_summary:
            console.print(
                Panel(
                    context_summary,
                    title="[yellow]📋 Loaded from Memory[/yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
        else:
            print_system("No previous memories found. Starting fresh.")
    except Exception as e:
        print_system(f"(Memory context load failed: {e})")

    console.print()

    # ── Main input loop ───────────────────────────────────────────────────
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! Your memories are saved.[/dim]")
            break

        if not user_input:
            continue

        # ── Handle special commands ───────────────────────────────────────
        if user_input.startswith("/"):
            await handle_command(user_input, agent)
            continue

        # ── Normal chat message ───────────────────────────────────────────
        console.print()

        # Show "thinking" indicator
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            try:
                response = await agent.chat(user_input)
            except Exception as e:
                console.print(f"[red]Agent error: {e}[/red]")
                logger_ref = logging.getLogger("chat")
                logger_ref.error(f"Agent error: {e}", exc_info=True)
                continue

        print_assistant(response)
        console.print()


async def handle_command(command: str, agent: OllamaAgent):
    """Handle /slash commands."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/quit" or cmd == "/exit" or cmd == "/q" or cmd == "/bye":
        console.print("[dim]Goodbye! Your memories are saved.[/dim]")
        sys.exit(0)

    elif cmd == "/help":
        console.print(HELP_TEXT)

    elif cmd == "/memory":
        print_system("Loading memory summary...")
        result = await agent._mcp.call_tool("summarise_memories", {"max_memories": 20})
        console.print(
            Panel(result, title="[yellow]Memory Summary[/yellow]", border_style="yellow")
        )

    elif cmd == "/entities":
        print_system("Loading entity facts...")
        result = await agent._mcp.call_tool("list_entities", {})
        console.print(
            Panel(result, title="[yellow]Known Facts[/yellow]", border_style="yellow")
        )

    elif cmd == "/recall":
        if not arg:
            console.print("[red]Usage: /recall <query>[/red]")
            return
        print_system(f"Searching memory for: {arg!r}")
        result = await agent._mcp.call_tool("recall", {"query": arg, "top_k": 5})
        console.print(
            Panel(result, title="[yellow]Recall Results[/yellow]", border_style="yellow")
        )

    elif cmd == "/forget":
        if not arg:
            console.print("[red]Usage: /forget <memory_id>[/red]")
            return
        result = await agent._mcp.call_tool("forget", {"memory_id": arg})
        console.print(f"[dim]{result}[/dim]")

    elif cmd == "/reset":
        agent.reset_conversation()
        console.print("[dim]Conversation history cleared. Persistent memory intact.[/dim]")

    elif cmd == "/status":
        status = agent._mcp._mem.get_status()
        lines = [
            f"Episodic memories: {status['episodic']['total_memories']}",
            f"Semantic facts:    {status['semantic']['total_facts']}",
            f"Known entities:    {', '.join(status['semantic']['entities']) or 'none'}",
            f"Session turns:     {status['working']['current_turns_in_window']}",
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title="[yellow]Memory Status[/yellow]",
                border_style="yellow",
            )
        )

    else:
        console.print(f"[red]Unknown command: {cmd}. Type /help for help.[/red]")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemoryOS Chat Client")
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Ollama model to use (default: qwen2.5:7b). Must be pulled first.",
    )
    args = parser.parse_args()

    asyncio.run(run_chat(model=args.model))
