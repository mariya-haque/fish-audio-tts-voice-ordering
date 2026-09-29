"""One turn: text in, speech out.

Two modes:
  reply    (default)  retrieve -> LLM -> speak the answer as it is generated
  verbatim            skip the brain entirely and speak exactly what was typed

Kept free of FastAPI types so it can be driven by the websocket handler, a test,
or a script. The caller supplies two async sinks:

    send_json(dict)   -- control/text events for the UI
    send_audio(bytes) -- raw PCM chunks for playback

Cancelling the task running respond() stops the reply mid-sentence: the async
generators unwind, the Fish websocket closes, and no further audio is emitted.
"""

import logging
import time
from typing import AsyncIterator, Awaitable, Callable

import config
import llm
import tts_stream
from rag import engine as rag

logger = logging.getLogger(__name__)

SendJSON = Callable[[dict], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


async def _one(text: str) -> AsyncIterator[str]:
    """A single-segment stream, for speaking text that already exists in full."""
    yield text


async def respond(
    question: str,
    history: list[dict],
    send_json: SendJSON,
    send_audio: SendAudio,
    speak_verbatim: bool = False,
) -> str:
    """Run one turn end to end. Returns the reply text (or the verbatim input).

    Emits, in order: `context` (if any) -> `token`* -> `audio_start` ->
    audio chunks -> `audio_end` -> `timing`.
    """
    turn_start = time.perf_counter()

    collected: list[str] = []
    first_token_at: float | None = None

    async def on_token(token: str) -> None:
        nonlocal first_token_at
        if first_token_at is None:
            first_token_at = time.perf_counter()
        collected.append(token)
        await send_json({"type": "token", "text": token})

    rag_ms = 0

    if speak_verbatim:
        # Echo before synthesis starts, so the UI shows the text then the speaking
        # state -- not the other way round.
        await on_token(question)
        segments = _one(question)
    else:
        # --- retrieve --------------------------------------------------------
        rag_start = time.perf_counter()
        retrieved = await rag.retrieve(question)
        rag_ms = _ms(rag_start)

        context_block = rag.build_context_block(retrieved) if retrieved else ""
        if retrieved:
            sources = sorted({(meta or {}).get("source", "unknown") for _, meta in retrieved})
            await send_json({"type": "context", "sources": sources, "count": len(retrieved)})

        # --- think -----------------------------------------------------------
        messages = llm.build_messages(question, history, context_block)
        segments = llm.buffer_for_tts(llm.stream_reply(messages), on_token=on_token)

    # --- speak --------------------------------------------------------------
    first_audio_at: float | None = None
    audio_bytes = 0

    await send_json(
        {
            "type": "audio_start",
            "sample_rate": config.FISH_SAMPLE_RATE,
            "format": config.FISH_AUDIO_FORMAT,
            "verbatim": speak_verbatim,
        }
    )

    try:
        async for chunk in tts_stream.synthesize(segments):
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            audio_bytes += len(chunk)
            await send_audio(chunk)
    finally:
        # Runs on cancellation too, so the client always leaves "speaking" state.
        await send_json({"type": "audio_end", "bytes": audio_bytes})

    reply = "".join(collected).strip()
    ttft = int((first_token_at - turn_start) * 1000) if first_token_at else None
    ttfa = int((first_audio_at - turn_start) * 1000) if first_audio_at else None

    await send_json(
        {
            "type": "timing",
            "rag_ms": rag_ms,
            "ttft_ms": ttft,
            "ttfa_ms": ttfa,
            "total_ms": _ms(turn_start),
            "audio_bytes": audio_bytes,
            "audio_seconds": round(audio_bytes / 2 / config.FISH_SAMPLE_RATE, 2),
        }
    )

    logger.info(
        "turn done (%s): rag=%dms ttft=%s ttfa=%s total=%dms chars=%d",
        "verbatim" if speak_verbatim else "reply",
        rag_ms,
        f"{ttft}ms" if ttft is not None else "-",
        f"{ttfa}ms" if ttfa is not None else "-",
        _ms(turn_start),
        len(reply),
    )

    return reply
