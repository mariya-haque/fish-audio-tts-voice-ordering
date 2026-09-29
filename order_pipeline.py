"""One ordering turn: text in, speech out, orders placed.

The sibling of [pipeline.py](pipeline.py), and deliberately shaped like it --
same two async sinks, same streaming-into-Fish trick, same cancellation
behaviour. The difference is that this one can act: the model is given the
ordering tools, and a turn runs

    think -> maybe call tools -> think again -> ... -> speak

looping until the model stops asking for tools.

Speech spans the whole loop. The model usually says a short "let me check that"
before a tool call and the real answer after it, and both belong to one
utterance -- so a single Fish synthesis is fed from a queue that stays open
across every pass, rather than opening a new websocket per pass and stitching
the audio together.

Kept free of FastAPI types so a script or a test can drive it. Cancelling the
task stops the reply mid-word: the generators unwind and the Fish socket closes.
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Awaitable, Callable

import config
import llm
import tools as ordering_tools
import tts_stream
from speedoy import menu

logger = logging.getLogger(__name__)

SendJSON = Callable[[dict], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]

# A runaway loop would burn tokens and money in a tight circle. In practice a
# turn uses one or two passes; five is generous.
MAX_TOOL_PASSES = 5


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def build_messages(question: str, history: list[dict], catalog: menu.Catalog) -> list[dict]:
    """System prompt with the live menu baked in, then the conversation.

    The whole menu goes in the system prompt rather than being retrieved,
    because ordering needs the exact set of dishes and prices on every turn, not
    the handful of chunks most similar to the last thing said.
    """
    system = f"{config.ORDER_SYSTEM_PROMPT}\n\n{catalog.compact_text()}"
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(history[-(config.MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": question})
    return messages


async def _queue_stream(queue: asyncio.Queue) -> AsyncIterator[str]:
    """Turn the speech queue into the token stream buffer_for_tts expects."""
    while True:
        piece = await queue.get()
        if piece is None:
            return
        yield piece


async def respond(
    question: str,
    history: list[dict],
    session: ordering_tools.OrderSession,
    send_json: SendJSON,
    send_audio: SendAudio,
) -> str:
    """Run one ordering turn end to end. Returns what the bot said.

    Emits `token`* and `tool`/`cart` events as it goes, wrapped in
    `audio_start` ... `audio_end`.
    """
    turn_start = time.perf_counter()
    catalog = await menu.get_catalog()

    messages = build_messages(question, history, catalog)
    spoken: list[str] = []
    first_token_at: float | None = None
    first_audio_at: float | None = None
    audio_bytes = 0
    tool_names: list[str] = []

    speech: asyncio.Queue = asyncio.Queue()

    async def speak() -> None:
        nonlocal first_audio_at, audio_bytes
        async for chunk in tts_stream.synthesize(llm.buffer_for_tts(_queue_stream(speech))):
            if first_audio_at is None:
                first_audio_at = time.perf_counter()
            audio_bytes += len(chunk)
            await send_audio(chunk)

    await send_json({
        "type": "audio_start",
        "sample_rate": config.FISH_SAMPLE_RATE,
        "format": config.FISH_AUDIO_FORMAT,
    })
    speaker = asyncio.create_task(speak())

    try:
        for pass_number in range(MAX_TOOL_PASSES):
            calls: list[dict] = []
            said_this_pass: list[str] = []

            async for event in llm.stream_with_tools(
                messages, ordering_tools.SCHEMAS, model=config.ORDER_MODEL
            ):
                if event["type"] == "text":
                    delta = event["delta"]
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    said_this_pass.append(delta)
                    spoken.append(delta)
                    await send_json({"type": "token", "text": delta})
                    speech.put_nowait(delta)
                else:
                    calls = event["calls"]

            if not calls:
                break

            # The assistant's own turn has to go back in verbatim, tool calls
            # and all, or the tool results below have nothing to attach to.
            messages.append({
                "role": "assistant",
                "content": "".join(said_this_pass) or None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in calls
                ],
            })

            for call in calls:
                name = call["name"]
                tool_names.append(name)
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                logger.info("tool %s %s", name, call["arguments"][:200])
                await send_json({"type": "tool", "name": name, "arguments": arguments})

                result = await ordering_tools.call_tool(name, arguments, session)

                if name in ordering_tools.MUTATING:
                    await send_json({
                        "type": "cart",
                        "cart": ordering_tools.cart_summary(catalog, session),
                        "order_type": "delivery" if session.customer.is_delivery else "pickup",
                        "still_needed": session.customer.missing(),
                        "order_id": session.placed_order_id,
                    })
                if name == "place_order" and result.get("ok"):
                    await send_json({
                        "type": "order_placed",
                        "order_id": result.get("order_id"),
                        "total": result.get("total"),
                        "currency": result.get("currency"),
                    })

                # `raw` is the API's own response object: useful in the log,
                # noise in the model's context window.
                trimmed = {k: v for k, v in result.items() if k != "raw"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(trimmed, ensure_ascii=False, default=str),
                })
        else:
            logger.warning("tool loop hit %d passes; stopping", MAX_TOOL_PASSES)

        # No more text is coming, so let Fish flush its last partial chunk.
        speech.put_nowait(None)
        await speaker
    except asyncio.CancelledError:
        speaker.cancel()
        raise
    except Exception:
        speaker.cancel()
        raise
    finally:
        # Runs on cancellation too, so the page always leaves the speaking state.
        await send_json({"type": "audio_end", "bytes": audio_bytes})

    reply = "".join(spoken).strip()
    ttft = int((first_token_at - turn_start) * 1000) if first_token_at else None
    ttfa = int((first_audio_at - turn_start) * 1000) if first_audio_at else None

    await send_json({
        "type": "timing",
        "ttft_ms": ttft,
        "ttfa_ms": ttfa,
        "total_ms": _ms(turn_start),
        "audio_bytes": audio_bytes,
        "audio_seconds": round(audio_bytes / 2 / config.FISH_SAMPLE_RATE, 2),
        "tools": tool_names,
    })

    logger.info(
        "order turn: ttft=%s ttfa=%s total=%dms chars=%d tools=%s",
        f"{ttft}ms" if ttft is not None else "-",
        f"{ttfa}ms" if ttfa is not None else "-",
        _ms(turn_start),
        len(reply),
        ",".join(tool_names) or "-",
    )
    return reply
