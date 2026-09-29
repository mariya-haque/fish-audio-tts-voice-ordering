"""Round-trip Fish Audio against itself: speak a sentence, then transcribe it.

Synthesising the test audio with the same account's TTS means there is no
fixture to keep in the repo and the clip is always in the voice the bot
actually uses.

Until API credit is added this reports the 402 clearly and exits non-zero --
that is the expected state, not a bug in the wiring. Everything up to the
network call is exercised regardless.

    python scripts/check_stt.py
    python scripts/check_stt.py --say "Jag vill bestalla en kebabtallrik"
"""

import argparse
import asyncio
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import stt  # noqa: E402
import tts_stream  # noqa: E402

PHRASE = "I would like a Kebabtallrik with rice and two Coca cola please."

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


def resample(pcm: bytes, src: int, dst: int) -> bytes:
    """Crude nearest-sample downsample, good enough to hand a recogniser.

    The browser gets the real thing for free by asking its AudioContext for the
    target rate; this only exists so the check can reuse TTS output.
    """
    if src == dst:
        return pcm
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm[: len(pcm) // 2 * 2])
    step = src / dst
    out = [samples[int(i * step)] for i in range(int(len(samples) / step))]
    return struct.pack(f"<{len(out)}h", *out)


async def one_segment(text: str):
    yield text


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--say", default=PHRASE, help="what to speak and transcribe")
    args = parser.parse_args()

    print("[wav framing]")
    header = stt.wav_bytes(b"\x00\x00" * 8000, 16000)
    check("RIFF header is 44 bytes", len(header) - 16000 == 44)
    check("declares RIFF/WAVE", header[:4] == b"RIFF" and header[8:12] == b"WAVE")
    check("declares the right rate", struct.unpack("<I", header[24:28])[0] == 16000)
    check("mono 16-bit", struct.unpack("<H", header[22:24])[0] == 1
          and struct.unpack("<H", header[34:36])[0] == 16)

    print("\n[short clips are skipped]")
    check("a tap transcribes to nothing", await stt.transcribe(b"\x00" * 100) == "")

    if not config.FISH_API_KEY:
        print("\nFISH_API_KEY is not set; cannot round-trip.")
        return 1

    print(f"\n[speaking]  {args.say!r}")
    spoken = bytearray()
    async for chunk in tts_stream.synthesize(one_segment(args.say)):
        spoken.extend(chunk)
    seconds = len(spoken) / 2 / config.FISH_SAMPLE_RATE
    check("Fish produced audio", len(spoken) > 0, f"{len(spoken)} bytes, {seconds:.1f}s")
    if not spoken:
        return 1

    clip = resample(bytes(spoken), config.FISH_SAMPLE_RATE, config.STT_SAMPLE_RATE)
    print(f"  downsampled {config.FISH_SAMPLE_RATE} -> {config.STT_SAMPLE_RATE} Hz, "
          f"{len(clip)} bytes")

    print("\n[transcribing]")
    try:
        heard = await stt.transcribe(clip)
    except stt.SpeechUnavailable as exc:
        print(f"  {exc}")
        print("\nThe wiring is in place; only the credit is missing. Re-run this "
              "once the balance is topped up.")
        return 1

    print(f"  heard: {heard!r}")
    check("something came back", bool(heard))

    # Word overlap rather than exact match: recognisers punctuate and case
    # differently, and the dish names are Swedish inside an English sentence.
    said_words = {w.strip(".,!?").lower() for w in args.say.split()}
    heard_words = {w.strip(".,!?").lower() for w in heard.split()}
    overlap = len(said_words & heard_words) / max(1, len(said_words))
    check("it matches what was spoken", overlap >= 0.6, f"{overlap:.0%} of words")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("Speech-to-text round trip works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
