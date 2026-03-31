#!/usr/bin/env python3
"""
inspect_memory.py — Memory Inspection & Repair Utility
Run from MemoryOS project root.

Usage:
    python inspect_memory.py              # show all stored data
    python inspect_memory.py --wipe       # wipe ALL memories
    python inspect_memory.py --wipe-semantic
    python inspect_memory.py --wipe-episodic
    python inspect_memory.py --delete-fact user.name
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import MemorySystem


def show_all(mem):
    print("\n" + "="*60)
    print("SEMANTIC MEMORY (SQLite facts)")
    print("="*60)
    facts = mem.semantic.get_all_facts()
    if not facts:
        print("  (empty)")
    else:
        for f in facts:
            flag = "  ⚠️  LOW CONFIDENCE" if f["confidence"] < 0.70 else ""
            print(
                f"  [{f['id'][:8]}] {f['entity']}.{f['attribute']} = "
                f"{f['value']!r:25s} conf={f['confidence']:.2f} "
                f"confirmed={f['times_confirmed']}x source={f['source']}{flag}"
            )

    print("\n" + "="*60)
    print("EPISODIC MEMORY (ChromaDB — recent 20)")
    print("="*60)
    recent = mem.episodic.get_recent(20)
    if not recent:
        print("  (empty)")
    else:
        for m in recent:
            print(f"  [{m['id'][:8]}] imp={m.get('importance', 0):.2f} | {m['content'][:80]}")

    print(f"\nTOTALS: {mem.episodic.count()} episodic | {mem.semantic.fact_count()} semantic facts\n")


def wipe_semantic(mem):
    import sqlite3
    conn = sqlite3.connect(mem.semantic._db_path)
    count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    conn.execute("DELETE FROM facts")
    conn.commit()
    conn.close()
    print(f"✓ Wiped {count} semantic facts.")


def wipe_episodic(mem):
    mem.episodic._client.delete_collection(mem.episodic.COLLECTION_NAME)
    mem.episodic._collection = mem.episodic._client.get_or_create_collection(
        name=mem.episodic.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print("✓ Wiped all episodic memories.")


def delete_fact(mem, entity_attr):
    parts = entity_attr.split(".", 1)
    if len(parts) != 2:
        print("Format: entity.attribute  e.g.  user.name")
        return
    fact = mem.semantic.get_fact(parts[0], parts[1])
    if not fact:
        print(f"No fact found for {entity_attr}")
        return
    mem.semantic.delete_fact(fact["id"])
    print(f"✓ Deleted: {entity_attr} = {fact['value']!r}")


def main():
    parser = argparse.ArgumentParser(description="MemoryOS Inspection Utility")
    parser.add_argument("--wipe", action="store_true", help="Wipe ALL memories")
    parser.add_argument("--wipe-semantic", action="store_true")
    parser.add_argument("--wipe-episodic", action="store_true")
    parser.add_argument("--delete-fact", metavar="entity.attribute")
    args = parser.parse_args()

    mem = MemorySystem(data_dir="./data")

    if args.wipe:
        confirm = input("⚠️  Wipe ALL memories? Type 'yes' to confirm: ")
        if confirm.strip().lower() == "yes":
            wipe_semantic(mem)
            wipe_episodic(mem)
            print("✓ All memories wiped. Fresh start.")
        else:
            print("Cancelled.")
    elif args.wipe_semantic:
        wipe_semantic(mem)
    elif args.wipe_episodic:
        wipe_episodic(mem)
    elif args.delete_fact:
        delete_fact(mem, args.delete_fact)
    else:
        show_all(mem)


if __name__ == "__main__":
    main()