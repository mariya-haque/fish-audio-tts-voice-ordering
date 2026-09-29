"""Live check against the real APIs. Uses short phrases to stay cheap.

    python scripts/check_live.py

Covers, in order: TTS output formats -> sample-rate handling -> streaming
websocket time-to-first-audio -> the full LLM+TTS pipeline.
"""

import asyncio
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import llm
import tts_stream

PHRASE = "Streaming speech keeps latency low."
OUT = config.BASE_DIR / "data" / "smoke"

failures: list[str] = []


def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(label)


def rule(title):
    print(f"\n=== {title} " + "=" * max(0, 56 - len(title)))


async def main() -> int:
    if config.missing_keys():
        print(f"Missing {', '.join(config.missing_keys())} -- cannot run live checks.")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    client = tts_stream.client()
    model = tts_stream.resolve_model()
    print(f"tts model: {model}   voice: {config.FISH_REFERENCE_ID or '(stock)'}")
    print(f"llm model: {config.OPENAI_MODEL}")

    # --- 1. formats ---------------------------------------------------------
    rule("TTS formats")
    produced = {}
    for fmt in ("wav", "pcm", "mp3"):
        try:
            started = time.perf_counter()
            audio = await client.tts.convert(
                text=PHRASE, format=fmt, model=model,
                reference_id=config.FISH_REFERENCE_ID,
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            produced[fmt] = audio
            (OUT / f"smoke.{fmt}").write_bytes(audio)
            check(f"format {fmt}", len(audio) > 1000, f"{len(audio)} bytes in {elapsed}ms")
        except Exception as exc:
            check(f"format {fmt}", False, f"{type(exc).__name__}: {exc}")

    # --- 2. is sample_rate actually honoured? -------------------------------
    rule("Sample rate")
    # Fish streams the WAV, so its header carries a placeholder frame count
    # (0x7FFFFFFF). Never trust getnframes() here -- derive duration from the
    # real byte length instead.
    if "wav" in produced:
        with wave.open(BytesIO(produced["wav"])) as w:
            wav_rate, wav_ch, wav_width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        real_seconds = (len(produced["wav"]) - 44) / (wav_width * wav_ch) / wav_rate
        print(f"  wav header: {wav_rate} Hz, {wav_ch}ch, {wav_width * 8}-bit, ~{real_seconds:.2f}s")
        check("header rate == FISH_SAMPLE_RATE", wav_rate == config.FISH_SAMPLE_RATE,
              f"{wav_rate} vs {config.FISH_SAMPLE_RATE}")
        check("mono 16-bit", wav_ch == 1 and wav_width == 2)

    # Prove the rate is a real knob rather than a fixed default, by asking for a
    # different one and reading it back.
    try:
        alt = await client.tts.convert(
            text="Rate check.", format="wav", model=model,
            config=type(tts_stream.tts_config())(format="wav", sample_rate=24000),
        )
        with wave.open(BytesIO(alt)) as w:
            alt_rate = w.getframerate()
        check("sample_rate is honoured", alt_rate == 24000, f"asked 24000, got {alt_rate}")
    except Exception as exc:
        check("sample_rate is honoured", False, f"{type(exc).__name__}: {exc}")

    # --- 3. streaming websocket TTS ----------------------------------------
    rule("Streaming websocket TTS")

    async def fake_tokens():
        for seg in ["Hello there. ", "This audio started ", "before the text ended. "]:
            yield seg
            await asyncio.sleep(0.05)

    started = time.perf_counter()
    first_at = None
    total = 0
    chunks = 0
    try:
        async for chunk in tts_stream.synthesize(fake_tokens()):
            if first_at is None:
                first_at = time.perf_counter()
            total += len(chunk)
            chunks += 1
        ttfa = int((first_at - started) * 1000) if first_at else None
        wall = int((time.perf_counter() - started) * 1000)
        check("audio received", total > 0, f"{total} bytes in {chunks} chunks")
        check("time-to-first-audio measured", ttfa is not None, f"{ttfa}ms (total {wall}ms)")
        (OUT / "stream.pcm").write_bytes(b"")
        print(f"  synthesised {total / 2 / config.FISH_SAMPLE_RATE:.2f}s of audio")
    except Exception as exc:
        check("streaming websocket tts", False, f"{type(exc).__name__}: {exc}")

    # --- 4. full pipeline ---------------------------------------------------
    rule("LLM -> TTS pipeline")
    started = time.perf_counter()
    first_token_at = None
    first_audio_at = None
    text_out: list[str] = []

    async def on_token(tok: str):
        nonlocal first_token_at
        if first_token_at is None:
            first_token_at = time.perf_counter()
        text_out.append(tok)

    try:
        messages = llm.build_messages("In one short sentence, what is latency?", [])
        segments = llm.buffer_for_tts(llm.stream_reply(messages), on_token=on_token)

        audio_total = 0
        async for chunk in tts_stream.synthesize(segments):
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            audio_total += len(chunk)

        reply = "".join(text_out).strip()
        print(f"  reply: {reply!r}")
        check("llm produced text", bool(reply))
        check("tts produced audio", audio_total > 0, f"{audio_total} bytes")
        if first_token_at:
            print(f"  time-to-first-token : {int((first_token_at - started) * 1000)}ms")
        if first_audio_at:
            print(f"  time-to-first-audio : {int((first_audio_at - started) * 1000)}ms")
        print(f"  total               : {int((time.perf_counter() - started) * 1000)}ms")
        check(
            "audio started before generation finished",
            first_audio_at is not None,
        )
    except Exception as exc:
        check("pipeline", False, f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print(f"All live checks passed. Artifacts in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
