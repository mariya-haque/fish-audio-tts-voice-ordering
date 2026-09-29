"""The mouth: Fish Audio streaming TTS over WebSocket.

Text segments go in as they are produced by the LLM; PCM audio chunks come out
before the model has finished its sentence. That overlap is the whole trick.

Signatures verified against fish-audio-sdk 1.3.0 by scripts/check_sdk.py:

    AsyncFishAudio().tts.stream_websocket(
        text_stream: AsyncIterable[str | TextEvent | FlushEvent],
        *, reference_id, references, format, latency, speed,
           config: TTSConfig, model, ws_options,
    )
"""

import logging
from functools import lru_cache
from typing import AsyncIterator

from fishaudio import AsyncFishAudio, FlushEvent, TTSConfig

import config

logger = logging.getLogger(__name__)

# fishaudio.types.shared.Model is a Literal, but it is only a type hint -- the SDK
# does no runtime validation and passes `model` straight through as an HTTP header
# (resources/tts.py: headers={"model": model}). So model ids newer than this SDK
# build, such as the free tier's s2.1-pro-free, work fine. Anything outside both
# sets below is still passed through, with a warning, since Fish ships new ids
# faster than the SDK's type stubs track them.
SDK_MODELS = {"speech-1.5", "speech-1.6", "s1", "s2-pro"}
KNOWN_NEWER_MODELS = {"s2.1-pro", "s2.1-pro-free"}


@lru_cache(maxsize=1)
def client() -> AsyncFishAudio:
    return AsyncFishAudio(api_key=config.FISH_API_KEY)


def resolve_model() -> str:
    model = config.FISH_TTS_MODEL
    if not model:
        return "s2-pro"
    if model not in SDK_MODELS and model not in KNOWN_NEWER_MODELS:
        logger.warning(
            "FISH_TTS_MODEL=%r is not one this build recognises; sending it anyway. "
            "Known ids: %s",
            model,
            ", ".join(sorted(SDK_MODELS | KNOWN_NEWER_MODELS)),
        )
    return model


def tts_config() -> TTSConfig:
    return TTSConfig(
        format=config.FISH_AUDIO_FORMAT,
        sample_rate=config.FISH_SAMPLE_RATE,
        latency=config.FISH_LATENCY,
        reference_id=config.FISH_REFERENCE_ID,
        # Smaller chunks than the 200 default: the synthesiser emits audio sooner,
        # which is what we are optimising for here.
        chunk_length=100,
        min_chunk_length=50,
        normalize=True,
    )


async def _with_final_flush(segments: AsyncIterator[str]) -> AsyncIterator[object]:
    """Pass segments through, then emit a FlushEvent.

    Without the trailing flush the synthesiser can sit on the last partial chunk
    waiting for more text that is never coming, which clips the end of a reply.
    """
    async for segment in segments:
        if segment:
            yield segment
    yield FlushEvent()


async def synthesize(segments: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Consume text segments, yield raw s16le mono PCM chunks at FISH_SAMPLE_RATE."""
    stream = client().tts.stream_websocket(
        _with_final_flush(segments),
        format=config.FISH_AUDIO_FORMAT,
        latency=config.FISH_LATENCY,
        speed=config.FISH_SPEED,
        reference_id=config.FISH_REFERENCE_ID,
        config=tts_config(),
        model=resolve_model(),
    )

    async for chunk in stream:
        if chunk:
            yield chunk
