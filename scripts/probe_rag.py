"""Show exactly what the model is asked when RAG is on, and what it answers.

    python scripts/probe_rag.py "your question"
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm
from rag import engine as rag

QUESTION = " ".join(sys.argv[1:]) or "How do I add documents to the knowledge base?"


async def main() -> int:
    retrieved = await rag.retrieve(QUESTION)
    print(f"retrieved {len(retrieved)} chunk(s)\n")
    for i, (doc, meta) in enumerate(retrieved, 1):
        print(f"--- chunk {i} ({meta.get('source')}) ---")
        print(doc[:400])
        print()

    block = rag.build_context_block(retrieved)
    messages = llm.build_messages(QUESTION, [], block)

    print("=" * 70)
    print("MESSAGES SENT")
    print("=" * 70)
    for m in messages:
        print(f"\n[{m['role']}]\n{m['content'][:900]}")

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    async for token in llm.stream_reply(messages):
        print(token, end="", flush=True)
    print("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
