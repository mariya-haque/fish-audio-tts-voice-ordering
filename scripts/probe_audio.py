"""Inspect the audio frames the server sends, and save them as a playable WAV.

If this WAV sounds fine, the server and Fish are healthy and the problem is in
the browser. If it is silent or garbled, the problem is upstream.

    python scripts/probe_audio.py            # needs the server running
    python scripts/probe_audio.py "say something else"
"""

import asyncio
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

import config

TEXT = " ".join(sys.argv[1:]) or "Testing one two three. Can you hear this clearly?"


async def main() -> int:
    url = f"ws://127.0.0.1:{config.PORT}/ws/audioprobe"
    try:
        ws = await websockets.connect(url, max_size=None)
    except Exception as exc:
        print(f"Cannot reach {url}: {exc}\nStart the server first: python server.py")
        return 1

    chunks: list[bytes] = []
    async with ws:
        ready = json.loads(await ws.recv())
        rate = ready["playback_sample_rate"]
        print(f"server says playback_sample_rate = {rate}")

        await ws.send(json.dumps({"type": "text", "text": TEXT, "verbatim": True}))

        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=90)
            if isinstance(msg, bytes):
                chunks.append(msg)
                continue
            event = json.loads(msg)
            if event["type"] == "error":
                print(f"server error: {event['error']}")
            if event["type"] == "timing":
                break

    if not chunks:
        print("\nNO AUDIO FRAMES RECEIVED — the problem is server-side, not the browser.")
        return 1

    print(f"\n{len(chunks)} binary frames, {sum(len(c) for c in chunks):,} bytes total")
    print(f"{'#':>4} {'bytes':>9}  {'even?':>6}")
    odd = 0
    for i, c in enumerate(chunks, 1):
        is_even = len(c) % 2 == 0
        if not is_even:
            odd += 1
        if i <= 8 or not is_even:
            print(f"{i:>4} {len(c):>9,}  {'yes' if is_even else 'ODD':>6}")
    if len(chunks) > 8:
        print(f"     ... {len(chunks) - 8} more")

    pcm = b"".join(chunks)
    print(f"\ntotal samples : {len(pcm) // 2:,}")
    print(f"duration      : {len(pcm) / 2 / rate:.2f}s at {rate} Hz")

    if odd:
        print(f"\n!! {odd} frame(s) had an ODD byte count.")
        print("   new Int16Array(buffer) throws on those, which kills the worklet silently.")
    else:
        print("\nAll frames are 16-bit aligned.")

    # Is it actually signal, or digital silence?
    peak = 0
    for i in range(0, min(len(pcm), 400_000) - 1, 2):
        v = int.from_bytes(pcm[i : i + 2], "little", signed=True)
        peak = max(peak, abs(v))
    print(f"peak amplitude: {peak} / 32768  ({peak / 327.68:.1f}% full scale)")
    if peak < 300:
        print("!! This is effectively silence — the server is sending empty audio.")

    out = config.BASE_DIR / "data" / "smoke" / "probe.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    print(f"\nwrote {out}")
    print("Play it. If you hear the voice here, the server is fine and the bug is in the browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
