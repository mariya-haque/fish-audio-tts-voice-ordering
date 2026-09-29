"""The ears: Fish Audio speech-to-text.

The mirror of [tts_stream.py](tts_stream.py). The browser records raw PCM while
the caller holds the talk button and streams it up as binary frames; this module
wraps the accumulated samples in a WAV container and hands them to Fish.

Fish ASR is a batch endpoint -- a complete clip in, text out -- with no
streaming and no voice activity detection. That is why the page uses
push-to-talk: releasing the button is the end-of-speech signal, which is exact
and costs nothing, where silence detection is a guess that cuts people off
mid-sentence.

ASR is billed separately from TTS on Fish. An account with TTS credit but no API
credit answers HTTP 402, which `SpeechUnavailable` turns into a sentence naming
the actual fix rather than a stack trace.
"""

import io
import logging
import struct
from functools import lru_cache

import config

logger = logging.getLogger(__name__)

# 16 kHz mono is the standard rate for speech recognition; anything higher is
# bytes on the wire that the recogniser discards.
SAMPLE_WIDTH = 2  # s16le
CHANNELS = 1

# Below this, the caller tapped the button rather than said anything. Fish
# charges for a request either way, so there is no reason to send it.
MIN_SPEECH_BYTES = 4000  # ~0.12s at 16 kHz


class SpeechUnavailable(RuntimeError):
    """Transcription could not run. `message` is safe to show or speak."""


@lru_cache(maxsize=1)
def _client():
    from fishaudio import AsyncFishAudio

    return AsyncFishAudio(api_key=config.FISH_API_KEY)


def wav_bytes(pcm: bytes, sample_rate: int | None = None) -> bytes:
    """Wrap raw s16le mono PCM in a 44-byte RIFF header.

    Fish wants a container, not bare samples, and building the header here is
    both smaller and more predictable than shipping an encoder to the browser.
    """
    rate = sample_rate or config.STT_SAMPLE_RATE
    byte_rate = rate * CHANNELS * SAMPLE_WIDTH

    header = b"".join((
        b"RIFF",
        struct.pack("<I", 36 + len(pcm)),
        b"WAVEfmt ",
        struct.pack("<IHHIIHH", 16, 1, CHANNELS, rate, byte_rate,
                    CHANNELS * SAMPLE_WIDTH, SAMPLE_WIDTH * 8),
        b"data",
        struct.pack("<I", len(pcm)),
    ))
    return header + pcm


async def transcribe(pcm: bytes, sample_rate: int | None = None) -> str:
    """Raw PCM in, spoken words out.

    Returns "" when the clip is too short to be speech. Raises
    SpeechUnavailable with a readable sentence on anything else.
    """
    if len(pcm) < MIN_SPEECH_BYTES:
        logger.info("clip too short to transcribe: %d bytes", len(pcm))
        return ""

    if not config.FISH_API_KEY:
        raise SpeechUnavailable("FISH_API_KEY is not set, so speech cannot be transcribed.")

    audio = wav_bytes(pcm, sample_rate)
    kwargs: dict = {"audio": audio, "include_timestamps": False}
    if config.STT_LANGUAGE:
        kwargs["language"] = config.STT_LANGUAGE

    try:
        response = await _client().asr.transcribe(**kwargs)
    except Exception as exc:
        detail = str(exc)
        if "402" in detail or "credit" in detail.lower():
            # The single most likely failure, and the one with a specific fix.
            raise SpeechUnavailable(
                "Fish Audio has no API credit for speech recognition. It is billed "
                "separately from text-to-speech -- add credit at "
                "fish.audio/app/developers."
            ) from exc
        logger.warning("transcription failed: %s", detail[:300])
        raise SpeechUnavailable(f"Speech recognition failed: {type(exc).__name__}.") from exc

    text = (getattr(response, "text", None) or "").strip()
    logger.info("transcribed %.1fs -> %r",
                len(pcm) / (config.STT_SAMPLE_RATE * SAMPLE_WIDTH), text[:120])
    return text


def duration_seconds(pcm: bytes, sample_rate: int | None = None) -> float:
    return len(pcm) / ((sample_rate or config.STT_SAMPLE_RATE) * SAMPLE_WIDTH)
