"""Compare time-to-first-token across chat models.

TTFT is the single biggest term in a voice agent's perceived latency: nothing can
be spoken until the first words exist. Reasoning models think before they emit,
which is fine for a chat box and fatal for a conversation.

    python scripts/bench_llm.py [model ...]
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import llm

DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-nano"]
QUESTION = "In one short sentence, what is latency?"


async def measure(model: str) -> dict:
    original = config.OPENAI_MODEL
    config.OPENAI_MODEL = model
    try:
        messages = llm.build_messages(QUESTION, [])
        started = time.perf_counter()
        first = None
        pieces = []
        async for token in llm.stream_reply(messages):
            if first is None:
                first = time.perf_counter()
            pieces.append(token)
        return {
            "model": model,
            "ttft_ms": int((first - started) * 1000) if first else None,
            "total_ms": int((time.perf_counter() - started) * 1000),
            "text": "".join(pieces).strip(),
        }
    except Exception as exc:
        return {"model": model, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        config.OPENAI_MODEL = original


RUNS = 3


async def main() -> int:
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"question: {QUESTION!r}")

    # The first request of the process pays DNS + TLS + client construction. Without
    # a warm-up, whichever model happens to go first looks the slowest.
    print("warming up the connection pool...")
    await measure(models[0])

    print(f"\n{'model':<16} {'TTFT (best of ' + str(RUNS) + ')':>18} {'median':>9} {'total':>9}   reply")
    print("-" * 105)

    rows = []
    for model in models:
        samples = []
        last = {}
        for _ in range(RUNS):
            last = await measure(model)
            if "error" in last:
                break
            samples.append(last["ttft_ms"])

        if not samples:
            print(f"{model:<16} {'--':>18} {'--':>9} {'--':>9}   {last.get('error', '?')[:50]}")
            continue

        samples.sort()
        row = {
            "model": model,
            "best": samples[0],
            "median": samples[len(samples) // 2],
            "total": last["total_ms"],
            "text": last["text"],
        }
        rows.append(row)
        print(
            f"{model:<16} {str(row['best']) + 'ms':>18} {str(row['median']) + 'ms':>9} "
            f"{str(row['total']) + 'ms':>9}   {row['text'][:48]}"
        )

    if rows:
        best = min(rows, key=lambda r: r["median"])
        print(f"\nfastest to first token: {best['model']} — {best['median']}ms median, {best['best']}ms best")
        print(f"Currently configured: OPENAI_MODEL={config.OPENAI_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
