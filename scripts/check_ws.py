"""Drive the server's websocket headlessly, the way the browser does.

Verifies the wire protocol, both speaking modes, streaming overlap, and the
interrupt path. Start the server first, then:

    python scripts/check_ws.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

import config

URL = f"ws://127.0.0.1:{config.PORT}/ws/selftest"

failures: list[str] = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(label)


def rule(title):
    print(f"\n=== {title} " + "=" * max(0, 56 - len(title)))


class Collector:
    def __init__(self):
        self.events: list[dict] = []
        self.audio_bytes = 0
        self.audio_chunks = 0
        self.first_audio_at: float | None = None
        self.first_token_at: float | None = None
        self.last_token_at: float | None = None
        self.tokens: list[str] = []

    def types(self):
        return [e["type"] for e in self.events]

    def find(self, kind):
        return next((e for e in self.events if e["type"] == kind), None)

    def text(self):
        return "".join(self.tokens).strip()


async def drain(ws, collector, until, timeout=90.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.perf_counter())
        except asyncio.TimeoutError:
            return False
        if isinstance(msg, bytes):
            if collector.first_audio_at is None:
                collector.first_audio_at = time.perf_counter()
            collector.audio_bytes += len(msg)
            collector.audio_chunks += 1
            continue
        event = json.loads(msg)
        collector.events.append(event)
        if event["type"] == "token":
            now = time.perf_counter()
            if collector.first_token_at is None:
                collector.first_token_at = now
            collector.last_token_at = now
            collector.tokens.append(event["text"])
        if event["type"] == "error":
            print(f"    server error: {event['error']}")
        if event["type"] == until:
            return True
    return False


async def send_turn(ws, text, verbatim=False, timeout=90.0):
    collector = Collector()
    started = time.perf_counter()
    await ws.send(json.dumps({"type": "text", "text": text, "verbatim": verbatim}))
    ok = await drain(ws, collector, until="timing", timeout=timeout)
    return collector, started, ok


async def main() -> int:
    try:
        ws = await websockets.connect(URL, max_size=None)
    except Exception as exc:
        print(f"Could not connect to {URL}: {type(exc).__name__}: {exc}")
        print("Start the server first:  python server.py")
        return 1

    async with ws:
        rule("Handshake")
        ready = json.loads(await ws.recv())
        check("ready is first frame", ready["type"] == "ready", ready["type"])
        check("advertises playback rate",
              ready.get("playback_sample_rate") == config.FISH_SAMPLE_RATE,
              str(ready.get("playback_sample_rate")))
        check("no missing keys", not ready.get("missing_keys"), str(ready.get("missing_keys")))
        print(f"  {ready['llm_model']} / tts {ready['tts_model']} / "
              f"{ready['kb_chunks']} kb chunks / voice {ready.get('voice') or 'stock'}")

        # --- reply mode ------------------------------------------------------
        rule("Reply mode (type -> AI answers -> spoken)")
        c, started, ok = await send_turn(ws, "Say hello in one short sentence.")
        check("turn completed", ok)
        check("echoed what was typed", c.find("transcript") is not None)
        check("streamed tokens", len(c.tokens) > 0, f"{len(c.tokens)} tokens")
        check("produced audio", c.audio_bytes > 0,
              f"{c.audio_bytes:,} bytes in {c.audio_chunks} chunks")
        check("audio_start and audio_end both sent",
              "audio_start" in c.types() and "audio_end" in c.types())
        print(f"  reply: {c.text()[:70]!r}")
        t = c.find("timing")
        if t:
            print(f"  timing: rag={t['rag_ms']}ms ttft={t['ttft_ms']}ms "
                  f"ttfa={t['ttfa_ms']}ms total={t['total_ms']}ms "
                  f"speech={t['audio_seconds']}s")

        # --- does streaming actually pay? ------------------------------------
        rule("Streaming vs. waiting for the full text")
        # The architecture's whole claim is that piping partial text into the TTS
        # socket beats generating everything first and only then synthesising.
        # Measure both and compare, rather than asserting it.
        question = ("Name six different fruits. Give one short sentence about each. "
                    "Take your time and be thorough.")
        c, started, ok = await send_turn(ws, question)
        check("turn completed", ok)
        t = c.find("timing")
        streaming_ttfa = t["ttfa_ms"] if t else None
        text_done_ms = int((c.last_token_at - started) * 1000) if c.last_token_at else None
        reply_text = c.text()
        check("multiple audio chunks (streamed, not one blob)", c.audio_chunks > 1,
              f"{c.audio_chunks} chunks")

        # Speak that exact reply verbatim: with the text already in hand, its ttfa
        # is the synthesiser's own warm-up cost, isolated from the LLM.
        v, _, ok = await send_turn(ws, reply_text, verbatim=True)
        check("baseline measured", ok)
        synth_ttfa = v.find("timing")["ttfa_ms"] if v.find("timing") else None

        if streaming_ttfa and text_done_ms and synth_ttfa:
            sequential = text_done_ms + synth_ttfa
            saved = sequential - streaming_ttfa
            print(f"  streaming        : first audio at {streaming_ttfa}ms")
            print(f"  generate-then-say: {text_done_ms}ms writing + {synth_ttfa}ms synthesis"
                  f" = {sequential}ms")
            print(f"  saving           : {saved}ms ({saved / sequential * 100:.0f}% faster to first sound)")
            check("streaming beats generate-then-speak", streaming_ttfa < sequential,
                  f"{streaming_ttfa}ms vs {sequential}ms")
            # Worth knowing which side dominates, since it decides where to optimise.
            print(f"  note: synthesis warm-up ({synth_ttfa}ms) "
                  f"{'exceeds' if synth_ttfa > text_done_ms else 'is under'} "
                  f"generation time ({text_done_ms}ms) — "
                  f"{'the mouth' if synth_ttfa > text_done_ms else 'the brain'} is the bottleneck")

        # --- verbatim mode ---------------------------------------------------
        rule("Verbatim mode (no AI)")
        phrase = "This sentence is spoken exactly as written."
        c, started, ok = await send_turn(ws, phrase, verbatim=True)
        check("turn completed", ok)
        check("echoed text is unchanged", c.text() == phrase, repr(c.text()))
        check("produced audio", c.audio_bytes > 0, f"{c.audio_bytes:,} bytes")
        check("skipped retrieval", c.find("context") is None)
        t = c.find("timing")
        if t:
            check("no LLM was involved", t["rag_ms"] == 0, f"rag_ms={t['rag_ms']}")
            print(f"  timing: ttfa={t['ttfa_ms']}ms total={t['total_ms']}ms "
                  f"speech={t['audio_seconds']}s")

        rule("RAG grounding")
        # Fresh history: an unrelated preceding turn drags the answer off topic.
        await ws.send(json.dumps({"type": "reset"}))
        await ws.recv()
        c, started, ok = await send_turn(ws, "How do I add documents to the knowledge base?")
        check("turn completed", ok)
        ctx = c.find("context")
        check("retrieved from the knowledge base", ctx is not None and ctx["count"] > 0,
              str(ctx))
        if ctx:
            print(f"  sources: {ctx['sources']}")
        answer = c.text().lower()
        check(
            "answered from the retrieved context",
            any(w in answer for w in ("drop", "folder", "ingest", "index", "file")),
            repr(c.text()[:90]),
        )
        print(f"  reply: {c.text()[:110]!r}")

        # --- interrupt --------------------------------------------------------
        rule("Interrupt")
        await ws.send(json.dumps({"type": "text", "text": "Count slowly from one to thirty."}))
        got_audio = False
        deadline = time.perf_counter() + 45
        while time.perf_counter() < deadline:
            msg = await asyncio.wait_for(ws.recv(), timeout=45)
            if isinstance(msg, bytes):
                got_audio = True
                break
        check("agent started speaking", got_audio)

        await ws.send(json.dumps({"type": "interrupt"}))
        interrupted = False
        deadline = time.perf_counter() + 25
        while time.perf_counter() < deadline:
            msg = await asyncio.wait_for(ws.recv(), timeout=25)
            if isinstance(msg, bytes):
                continue
            if json.loads(msg)["type"] == "interrupted":
                interrupted = True
                break
        check("server acknowledged the interrupt", interrupted)

        c, started, ok = await send_turn(ws, "Are you still there? Answer in one word.")
        check("session survives an interrupt", ok and c.audio_bytes > 0,
              f"{c.audio_bytes:,} bytes")
        print(f"  reply: {c.text()[:60]!r}")

        # --- guards -----------------------------------------------------------
        rule("Guards")
        await ws.send(json.dumps({"type": "text", "text": "   "}))
        await ws.send(json.dumps({"type": "reset"}))
        msg = json.loads(await ws.recv())
        check("blank input ignored, reset still handled",
              msg.get("info") == "history_cleared", str(msg))

        await ws.send("not json at all")
        msg = json.loads(await ws.recv())
        check("malformed frame returns an error, not a crash",
              msg["type"] == "error", str(msg)[:60])

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("All websocket checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
