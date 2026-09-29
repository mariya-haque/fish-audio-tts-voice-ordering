"""Step 0 smoke test.

Answers the questions the Fish Audio docs leave open before any application code
depends on them:

  1. What is the real signature of tts.stream_websocket / tts.stream / asr.transcribe?
  2. Is format="pcm" accepted, and at what sample rate / bit depth?
  3. Are both API keys live?

Run:  python scripts/check_sdk.py
"""

import inspect
import os
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import fishaudio


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(label, obj):
    try:
        print(f"  {label}{inspect.signature(obj)}")
    except (TypeError, ValueError) as exc:
        print(f"  {label}(?) -- {exc}")


rule("PACKAGE")
print("  version:", getattr(fishaudio, "__version__", "?"))
print("  exports:", sorted(n for n in dir(fishaudio) if not n.startswith("_")))
print("  submodules:", [m.name for m in pkgutil.walk_packages(fishaudio.__path__, "fishaudio.")])

rule("CLIENT CONSTRUCTORS")
for name in ("FishAudio", "AsyncFishAudio"):
    cls = getattr(fishaudio, name, None)
    if cls is None:
        print(f"  {name}: MISSING")
        continue
    show(name, cls)
    print(f"    resource clients: {sorted(n for n in dir(cls) if not n.startswith('_'))}")

rule("RESOURCE METHODS")
api_key = os.getenv("FISH_API_KEY")
if not api_key:
    print("  FISH_API_KEY not set -- signatures still print, live calls skipped.\n")

# A placeholder key is enough to build the client and read signatures; nothing is sent.
probe = fishaudio.FishAudio(api_key=api_key or "sk-placeholder-introspection-only")
aprobe = fishaudio.AsyncFishAudio(api_key=api_key or "sk-placeholder-introspection-only")

for res_name in ("tts", "asr", "voices", "account"):
    for label, c in (("sync", probe), ("async", aprobe)):
        res = getattr(c, res_name, None)
        if res is None:
            continue
        print(f"\n  [{label}.{res_name}] -> {type(res).__name__}")
        for meth_name in sorted(n for n in dir(res) if not n.startswith("_")):
            meth = getattr(res, meth_name)
            if callable(meth):
                show(f"    .{meth_name}", meth)

rule("REALTIME RESOURCE (undocumented)")
try:
    from fishaudio.resources import realtime as _rt

    print("  members:", sorted(n for n in dir(_rt) if not n.startswith("_")))
    for name in sorted(n for n in dir(_rt) if not n.startswith("_")):
        obj = getattr(_rt, name)
        if inspect.isclass(obj) and obj.__module__.startswith("fishaudio"):
            print(f"\n  class {name}:")
            for meth_name in sorted(n for n in dir(obj) if not n.startswith("_")):
                meth = getattr(obj, meth_name)
                if callable(meth):
                    show(f"    .{meth_name}", meth)
except Exception as exc:
    print(f"  could not inspect: {type(exc).__name__}: {exc}")

rule("WEBSOCKET OPTIONS / AUDIO STREAM")
for name in ("WebSocketOptions", "AudioStream", "AsyncAudioStream"):
    obj = getattr(fishaudio, name, None)
    if obj is None:
        continue
    print(f"\n  {name}:")
    show("    __init__", obj)
    for meth_name in sorted(n for n in dir(obj) if not n.startswith("_")):
        meth = getattr(obj, meth_name)
        if callable(meth):
            show(f"    .{meth_name}", meth)

rule("CONFIG / REQUEST MODELS")
for name in sorted(n for n in dir(fishaudio) if not n.startswith("_")):
    obj = getattr(fishaudio, name)
    fields = getattr(obj, "model_fields", None)
    if isinstance(fields, dict):
        print(f"\n  {name}:")
        for fname, finfo in fields.items():
            print(f"    {fname}: {finfo.annotation} = {finfo.default!r}")

if not api_key:
    print("\nNo FISH_API_KEY -- skipping live round trip.")
    sys.exit(0)

rule("LIVE ROUND TRIP")
out_dir = Path(__file__).resolve().parent.parent / "data" / "smoke"
out_dir.mkdir(parents=True, exist_ok=True)

phrase = "The quick brown fox jumps over the lazy dog."

for fmt in ("pcm", "mp3", "wav", "opus"):
    try:
        audio = probe.tts.convert(text=phrase, format=fmt)
        path = out_dir / f"smoke.{fmt}"
        path.write_bytes(audio)
        print(f"  format={fmt!r:8} OK  {len(audio):>8} bytes -> {path.name}")
        if fmt == "pcm":
            # s16le mono: infer the rate from byte count vs. expected duration.
            samples = len(audio) // 2
            for rate in (16000, 22050, 24000, 32000, 44100, 48000):
                print(f"      if {rate} Hz -> {samples / rate:.2f}s")
    except Exception as exc:
        print(f"  format={fmt!r:8} FAILED  {type(exc).__name__}: {exc}")

rule("ASR ROUND TRIP")
wav_path = out_dir / "smoke.wav"
mp3_path = out_dir / "smoke.mp3"
src = wav_path if wav_path.exists() else mp3_path
if src.exists():
    try:
        result = probe.asr.transcribe(audio=src.read_bytes(), language="en")
        print(f"  source: {src.name}")
        print(f"  text:     {getattr(result, 'text', result)!r}")
        print(f"  duration: {getattr(result, 'duration', '?')}")
        print(f"  fields:   {sorted(n for n in dir(result) if not n.startswith('_'))}")
    except Exception as exc:
        print(f"  ASR FAILED  {type(exc).__name__}: {exc}")
else:
    print("  no wav/mp3 produced above; skipping")

rule("VOICES")
try:
    voices = probe.voices.list()
    print(f"  type: {type(voices).__name__}")
    items = getattr(voices, "items", None) or getattr(voices, "data", None) or voices
    for v in list(items)[:5]:
        print(f"    {getattr(v, 'id', '?')}  {getattr(v, 'title', getattr(v, 'name', '?'))}")
except Exception as exc:
    print(f"  voices.list FAILED  {type(exc).__name__}: {exc}")

print("\nDone.")
