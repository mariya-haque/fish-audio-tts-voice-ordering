"""The brain: streamed OpenAI completions, chunked for low-latency speech.

Two responsibilities:
  1. stream_reply()   -- yield token deltas as the model produces them
  2. buffer_for_tts() -- regroup those deltas into speakable segments

(2) is the latency-critical part. Fish's guidance is to send 5-10 complete words
per websocket message; sending single tokens wastes round trips and sending whole
paragraphs throws away the entire point of streaming. The first segment is
deliberately flushed early -- it is the only one the user is waiting on.
"""

import logging
import re
from functools import lru_cache
from typing import AsyncIterator, Awaitable, Callable

from openai import AsyncOpenAI

import config

logger = logging.getLogger(__name__)

# First flush: get *something* speaking fast.
FIRST_FLUSH_WORDS = 4
# Steady state: mid-range of Fish's 5-10 word recommendation.
FLUSH_WORDS = 8
# Backstop for text containing no whitespace at all (a URL, a long identifier).
MAX_BUFFER_CHARS = 400

_SENTENCE_END = re.compile(r"[.!?;:](?=\s|$)|[。！？]")
# "Dr." / "3.5" / "e.g." would otherwise trigger a false sentence break.
_ABBREVIATION = re.compile(r"(?:\b[A-Z][a-z]{0,3}|\b\d+|\b[a-z])\.$")


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=config.OPENAI_API_KEY)


def build_messages(
    question: str,
    history: list[dict],
    context_block: str = "",
) -> list[dict]:
    """Assemble the request. Retrieved context rides with the current question
    rather than the system prompt, so it does not pollute later turns."""
    messages: list[dict] = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    messages.extend(history[-(config.MAX_HISTORY_TURNS * 2) :])

    if context_block:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Knowledge base excerpts that may be relevant:\n"
                    f"{context_block}\n\n"
                    f"Question: {question}"
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": question})

    return messages


async def stream_reply(messages: list[dict]) -> AsyncIterator[str]:
    """Yield token deltas from the chat model."""
    kwargs: dict = {
        "model": config.OPENAI_MODEL,
        "messages": messages,
        "stream": True,
    }
    # Omitted entirely unless configured: gpt-5 and the reasoning models return
    # an unsupported_value error for any temperature but their default.
    if config.OPENAI_TEMPERATURE is not None:
        kwargs["temperature"] = config.OPENAI_TEMPERATURE

    stream = await _client().chat.completions.create(**kwargs)
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            yield piece


async def stream_with_tools(
    messages: list[dict],
    tool_schemas: list[dict],
    model: str | None = None,
) -> AsyncIterator[dict]:
    """Stream a turn that may call tools.

    Yields `{"type": "text", "delta": ...}` as prose arrives, then a single
    `{"type": "tool_calls", "calls": [...]}` at the end if the model asked for
    any. `stream_reply` above deliberately reads only `delta.content` and drops
    everything else, which is right for the text-to-speech page but would
    silently swallow every tool call here.

    Tool call arguments arrive split across deltas the same way prose does, and
    are keyed by index rather than id -- only the first delta of each call
    carries the id and the name.
    """
    kwargs: dict = {
        "model": model or config.OPENAI_MODEL,
        "messages": messages,
        "stream": True,
    }
    if tool_schemas:
        kwargs["tools"] = tool_schemas
        kwargs["tool_choice"] = "auto"
    if config.OPENAI_TEMPERATURE is not None:
        kwargs["temperature"] = config.OPENAI_TEMPERATURE

    pending: dict[int, dict] = {}

    stream = await _client().chat.completions.create(**kwargs)
    async for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta

        piece = getattr(delta, "content", None)
        if piece:
            yield {"type": "text", "delta": piece}

        for call in getattr(delta, "tool_calls", None) or []:
            slot = pending.setdefault(
                call.index, {"id": "", "name": "", "arguments": ""}
            )
            if call.id:
                slot["id"] = call.id
            function = getattr(call, "function", None)
            if function is None:
                continue
            if function.name:
                slot["name"] = function.name
            if function.arguments:
                slot["arguments"] += function.arguments

    if pending:
        yield {
            "type": "tool_calls",
            "calls": [pending[i] for i in sorted(pending)],
        }


def _complete_prefix(buffer: str) -> str:
    """The portion of `buffer` that ends on a word boundary.

    Model tokens are sub-word ("Hel" + "lo"), so a flush decision made on the raw
    buffer can cut a word in half and the synthesiser will dutifully pronounce the
    fragment. Only text up to the last whitespace is ever eligible to be sent.
    """
    idx = max(buffer.rfind(" "), buffer.rfind("\n"), buffer.rfind("\t"))
    return "" if idx == -1 else buffer[: idx + 1]


def _flush_point(complete: str, is_first: bool) -> int:
    """Index in `complete` to cut at (exclusive); 0 means "not yet"."""
    stripped = complete.rstrip()
    if not stripped:
        return 0

    words = len(stripped.split())
    target = FIRST_FLUSH_WORDS if is_first else FLUSH_WORDS

    # Prefer breaking at a clause boundary -- it gives the synthesiser a natural
    # prosodic unit instead of an arbitrary N-word window.
    if words >= 3:
        for match in reversed(list(_SENTENCE_END.finditer(stripped))):
            head = stripped[: match.end()]
            if len(head.split()) >= 3 and not _ABBREVIATION.search(head):
                return match.end()

    if words >= target:
        return len(complete)

    if is_first and words >= 2 and stripped.endswith(","):
        return len(stripped)

    return 0


async def buffer_for_tts(
    tokens: AsyncIterator[str],
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> AsyncIterator[str]:
    """Regroup token deltas into speakable segments.

    `on_token` receives every raw delta as it arrives, so the UI can render text
    at full token speed while the mouth consumes coarser chunks.
    """
    buffer = ""
    is_first = True

    async for token in tokens:
        if on_token is not None:
            await on_token(token)

        buffer += token

        complete = _complete_prefix(buffer)
        if not complete:
            # No word boundary yet. Guard against a pathological run with no
            # whitespace so the buffer cannot grow without bound.
            if len(buffer) >= MAX_BUFFER_CHARS:
                yield buffer
                buffer = ""
                is_first = False
            continue

        cut = _flush_point(complete, is_first)
        if cut:
            segment = buffer[:cut].strip()
            buffer = buffer[cut:]
            is_first = False
            if segment:
                logger.debug("tts segment: %r", segment)
                yield segment + " "

    tail = buffer.strip()
    if tail:
        yield tail
