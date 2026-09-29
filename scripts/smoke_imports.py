"""Import every module and exercise the pure-logic pieces without any network.

    python scripts/smoke_imports.py
"""

import asyncio
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import llm
import pipeline
import server
import tts_stream
from rag import engine, ingest

failures = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(label)


print("\n[imports]")
check("all modules import", True)

print("\n[config]")
check("port", config.PORT == 8010, f"{config.PORT}")
check("pcm format", config.FISH_AUDIO_FORMAT == "pcm")
check("no leftover ASR config", not hasattr(config, "STT_PROVIDER"))
print(f"  info  missing keys: {config.missing_keys() or 'none'}")
print(f"  info  llm: {config.OPENAI_MODEL}")

print("\n[tts_stream.resolve_model]")
original = config.FISH_TTS_MODEL
# The SDK's Model Literal is a hint only; the value is passed through as a header,
# so ids newer than this SDK build must survive untouched.
for candidate in ("s2.1-pro-free", "s2.1-pro", "s2-pro", "s1"):
    config.FISH_TTS_MODEL = candidate
    check(f"passes {candidate} through", tts_stream.resolve_model() == candidate)
config.FISH_TTS_MODEL = ""
check("empty falls back to s2-pro", tts_stream.resolve_model() == "s2-pro")
config.FISH_TTS_MODEL = original
check("TTSConfig builds", tts_stream.tts_config().format == "pcm")
print(f"  info  configured model: {tts_stream.resolve_model()}")

print("\n[llm.buffer_for_tts]")


async def collect(tokens, **kw):
    async def gen():
        for t in tokens:
            yield t

    return [s async for s in llm.buffer_for_tts(gen(), **kw)]


sentence = "The quick brown fox jumps over the lazy dog. It was a bright cold day in April. "
# Fed one character at a time -- the worst case for sub-word token boundaries.
segments = asyncio.run(collect(list(sentence)))
check("segments produced", len(segments) >= 3, f"{len(segments)} segments")
check("first segment is short", len(segments[0].split()) <= llm.FIRST_FLUSH_WORDS + 1, repr(segments[0]))
check("nothing lost", "".join(segments).split() == sentence.split())
# A mid-word split makes the synthesiser pronounce fragments, so every word in
# every segment must be a real word from the source.
source_words = set(sentence.split())
check("no mid-word splits", all(w in source_words for s in segments for w in s.split()))
for s in segments:
    print(f"    -> {s!r}")

seen: list[str] = []


async def record(token: str) -> None:
    seen.append(token)


tokens = ["Hel", "lo ", "there ", "friend ", "of ", "mine. "]
asyncio.run(collect(tokens, on_token=record))
check("on_token sees every delta", seen == tokens, f"{len(seen)} deltas")

check("empty stream yields nothing", asyncio.run(collect([])) == [])
check("single word still spoken", asyncio.run(collect(["Hi."])) == ["Hi."])

print("\n[llm abbreviation guard]")
segs = asyncio.run(collect(list("Dr. Smith arrived at 3.5 past noon and waited. Then he left. ")))
check("no split mid-abbreviation", not any(s.strip() in {"Dr.", "3.5"} for s in segs))
for s in segs:
    print(f"    -> {s!r}")

print("\n[rag chunking]")
text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(5))
chunks = ingest.chunk(text, 800, 100)
check("chunks produced", len(chunks) > 1, f"{len(chunks)} chunks")
check("respects size", all(len(c) <= 900 for c in chunks), f"max {max(len(c) for c in chunks)}")
check("splits oversized paragraph", len(ingest.chunk("x" * 5000, 800, 100)) > 1)

print("\n[rag engine, offline]")
check("empty query short-circuits", asyncio.run(engine.retrieve("")) == [])
check("context block of []", engine.build_context_block([]) == "")
check("context block formats source",
      "[Source: a.md]" in engine.build_context_block([("body", {"source": "a.md"})]))

print("\n[pipeline verbatim mode]")


async def verbatim_probe():
    sent, audio = [], []

    async def send_json(p):
        sent.append(p)

    async def send_audio(b):
        audio.append(b)

    # Speak without touching the network: stub the synthesiser.
    original = tts_stream.synthesize

    async def fake(segments):
        async for s in segments:
            yield s.encode()

    tts_stream.synthesize = fake
    pipeline.tts_stream.synthesize = fake
    try:
        return await pipeline.respond("Hello world", [], send_json, send_audio,
                                      speak_verbatim=True), sent, audio
    finally:
        tts_stream.synthesize = original
        pipeline.tts_stream.synthesize = original


reply, sent, audio = asyncio.run(verbatim_probe())
check("verbatim returns the input", reply == "Hello world", repr(reply))
check("verbatim skips retrieval", not any(e["type"] == "context" for e in sent))
check("verbatim reports rag_ms 0",
      next(e for e in sent if e["type"] == "timing")["rag_ms"] == 0)
check("verbatim emitted audio", b"".join(audio) == b"Hello world")
check("event order",
      [e["type"] for e in sent][:2] == ["token", "audio_start"],
      str([e["type"] for e in sent]))

print("\n[server routes]")
routes = {getattr(r, "path", None) for r in server.app.routes}
check("/health", "/health" in routes)
check("/voices", "/voices" in routes)
check("/ws/{session_id}", "/ws/{session_id}" in routes)
check("/ws/order/{session_id}", "/ws/order/{session_id}" in routes)
check("/menu", "/menu" in routes)

print("\n[static assets]")
for name in ("index.html", "app.js", "audio-playback.worklet.js", "order.html",
             "order.js", "audio-recorder.worklet.js"):
    check(name, (config.STATIC_DIR / name).exists())

app_js = (config.STATIC_DIR / "app.js").read_text(encoding="utf-8")
order_js = (config.STATIC_DIR / "order.js").read_text(encoding="utf-8")
# The assistant page has no microphone; the mic belongs to the ordering page
# alone, which is why this is scoped to app.js rather than the whole directory.
check("assistant page has no microphone", "getUserMedia" not in app_js)
check("verbatim toggle wired", "verbatim" in app_js)
check("ordering page takes its rates from the server",
      "playback_sample_rate" in order_js and "mic_sample_rate" in order_js)
check("ordering page renders the cart", "renderCart" in order_js)
check("ordering page captures the mic", "getUserMedia" in order_js)
# Dragging off the button or releasing outside it would otherwise leave the
# microphone recording with nothing to stop it.
check("push-to-talk cannot get stuck on",
      all(h in order_js for h in ("pointerleave", "window.addEventListener('pointerup'",
                                  "'blur'")))

print("\n[speech to text]")
import stt  # noqa: E402

header = stt.wav_bytes(b"\x00\x00" * 100, 16000)
check("wav header is 44 bytes", len(header) - 200 == 44)
check("wav declares RIFF/WAVE", header[:4] == b"RIFF" and header[8:12] == b"WAVE")
check("wav declares the configured rate",
      int.from_bytes(header[24:28], "little") == 16000)
check("a tap is not sent for transcription",
      asyncio.run(stt.transcribe(b"\x00" * 100)) == "")
check("duration maths", abs(stt.duration_seconds(b"\x00" * 32000, 16000) - 1.0) < 0.001)

print("\n[ordering tools]")
import tools  # noqa: E402

names = {t["function"]["name"] for t in tools.SCHEMAS}
check("every schema has a handler", names == set(tools.HANDLERS), str(names ^ set(tools.HANDLERS)))
check("schemas are well formed", all(
    t["type"] == "function"
    and t["function"]["parameters"]["type"] == "object"
    and isinstance(t["function"]["parameters"]["properties"], dict)
    and set(t["function"]["parameters"]["required"])
    <= set(t["function"]["parameters"]["properties"])
    for t in tools.SCHEMAS
))
check("place_order demands a confirmed total",
      "confirmed_total" in next(t for t in tools.SCHEMAS
                                if t["function"]["name"] == "place_order"
                                )["function"]["parameters"]["required"])
check("mutating tools are all real tools", tools.MUTATING <= set(tools.HANDLERS))

print("\n[tool-call streaming]")


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Call:
    def __init__(self, index, id_, name, arguments):
        self.index, self.id = index, id_
        self.function = type("F", (), {"name": name, "arguments": arguments})()


# Tool arguments arrive split across deltas and keyed by index; only the first
# delta of a call carries the id and the name.
async def _fake_stream():
    for delta in (
        _Delta(content="One moment"),
        _Delta(tool_calls=[_Call(0, "call_1", "search_menu", '{"que')]),
        _Delta(tool_calls=[_Call(0, None, None, 'ry": "kebab"}')]),
    ):
        yield type("E", (), {"choices": [type("C", (), {"delta": delta})()]})()


async def _collect_tool_stream():
    import llm as llm_mod
    original = llm_mod._client
    llm_mod._client = lambda: type("X", (), {"chat": type("Y", (), {"completions": type(
        "Z", (), {"create": staticmethod(lambda **kw: _awaitable(_fake_stream()))})()})()})()
    try:
        return [e async for e in llm_mod.stream_with_tools([], [])]
    finally:
        llm_mod._client = original


async def _awaitable(value):
    return value


events = asyncio.run(_collect_tool_stream())
check("text deltas pass through", events[0] == {"type": "text", "delta": "One moment"})
check("split tool arguments are reassembled",
      events[-1]["calls"] == [{"id": "call_1", "name": "search_menu",
                               "arguments": '{"query": "kebab"}'}],
      str(events[-1]["calls"]))

print("\n[order payload]")
from speedoy import cart as cart_mod  # noqa: E402

payload_src = inspect.getsource(cart_mod.CartLine.to_payload)
check("template id stays empty (a real id 500s the API)",
      '"TemplateSelectedPrimaryOptionItemId": ""' in payload_src)

import speedoy.order as order_mod  # noqa: E402

order_src = inspect.getsource(order_mod.build_payload)
check("platfrom keeps the API's spelling", '"platfrom"' in order_src)
check("address keeps SpecialIntructions",
      '"SpecialIntructions"' in inspect.getsource(order_mod.build_address))

print()
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("All offline checks passed.")
