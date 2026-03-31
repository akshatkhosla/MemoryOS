"""
memory/__init__.py
══════════════════

Exposes the three memory tiers as a unified MemorySystem.

Usage (in server.py):
    from memory import MemorySystem
    memory = MemorySystem()

    # Each tier is accessible as an attribute:
    memory.working.add_turn("user", "hello")
    memory.episodic.store("User said hello", importance=0.3)
    memory.semantic.upsert_fact("user", "name", "Akshat")
"""

from .working import WorkingMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
import os
import logging

logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Unified interface to all three memory tiers.

    Instantiate once in server.py and pass to tools.py.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        chroma_subdir: str = "chroma",
        sqlite_filename: str = "semantic.db",
    ):
        """
        Initialize all three memory tiers.

        Args:
            data_dir:        root directory for all persistent data
            chroma_subdir:   subdirectory inside data_dir for ChromaDB files
            sqlite_filename: SQLite DB filename inside data_dir
        """
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        chroma_path = os.path.join(data_dir, chroma_subdir)
        sqlite_path = os.path.join(data_dir, sqlite_filename)

        logger.info(f"MemorySystem initializing (data_dir={data_dir})")

        # Tier 1: Working Memory (in-process, no files)
        self.working = WorkingMemory()

        # Tier 2: Episodic Memory (ChromaDB, persists to chroma_path)
        self.episodic = EpisodicMemory(persist_dir=chroma_path)

        # Tier 3: Semantic Memory (SQLite, persists to sqlite_path)
        self.semantic = SemanticMemory(db_path=sqlite_path)

        logger.info(
            f"MemorySystem ready. "
            f"episodic_count={self.episodic.count()}, "
            f"semantic_count={self.semantic.fact_count()}"
        )

    def get_status(self) -> dict:
        """Return a status summary of all memory tiers."""
        session_summary = self.working.get_session_summary()
        return {
            "working": session_summary,
            "episodic": {
                "total_memories": self.episodic.count(),
            },
            "semantic": {
                "total_facts": self.semantic.fact_count(),
                "entities": self.semantic.get_all_entities(),
            },
        }

    def __repr__(self) -> str:
        return (
            f"MemorySystem("
            f"working={self.working}, "
            f"episodic={self.episodic}, "
            f"semantic={self.semantic})"
        )
