"""Find a FISH_REFERENCE_ID — the voice the agent speaks with.

`voices.list()` searches the whole public Fish library, not just your own voices,
so you can browse and pick one without cloning anything.

    python scripts/list_voices.py                      # most-used voices
    python scripts/list_voices.py --search "morgan"    # by title
    python scripts/list_voices.py --language en        # by language
    python scripts/list_voices.py --tag male --tag calm
    python scripts/list_voices.py --mine               # only voices you own
    python scripts/list_voices.py --try <id>           # hear one, saved to data/voices/

Paste the id you like into .env as FISH_REFERENCE_ID and restart the server.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import tts_stream

SAMPLE = "Hello. This is how I sound when I read your text aloud."


async def preview(voice_id: str, text: str | None = None) -> int:
    client = tts_stream.client()

    # Confirm the id resolves before spending a synthesis call on it.
    try:
        voice = await client.voices.get(voice_id)
        print(f"found: {getattr(voice, 'title', '?')!r}")
        for field in ("id", "languages", "task_count", "visibility", "author", "description"):
            value = getattr(voice, field, None)
            if value:
                print(f"  {field:<12} {str(value)[:90]}")
    except Exception as exc:
        print(f"voice id {voice_id!r} did not resolve: {type(exc).__name__}: {exc}")
        return 1

    out_dir = config.BASE_DIR / "data" / "voices"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{voice_id}.mp3"

    print(f"\nsynthesising with {voice_id} ...")
    try:
        audio = await client.tts.convert(
            text=text or SAMPLE,
            reference_id=voice_id,
            format="mp3",
            model=tts_stream.resolve_model(),
        )
    except Exception as exc:
        print(f"failed: {type(exc).__name__}: {exc}")
        return 1

    path.write_bytes(audio)
    print(f"saved {len(audio):,} bytes -> {path}")
    print("Open that file to hear it. If you like it, put this in .env:")
    print(f"\n    FISH_REFERENCE_ID={voice_id}\n")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search", help="filter by title")
    parser.add_argument("--tag", action="append", help="filter by tag (repeatable)")
    parser.add_argument("--language", action="append", help="filter by language, e.g. en")
    parser.add_argument("--mine", action="store_true", help="only voices on your account")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--newest", action="store_true", help="sort by created_at")
    parser.add_argument("--try", dest="try_id", help="synthesise a sample with this voice id")
    parser.add_argument("--text", help="text to speak with --try (default: a short sample)")
    args = parser.parse_args()

    if not config.FISH_API_KEY:
        print("FISH_API_KEY is not set in .env")
        return 1

    if args.try_id:
        return await preview(args.try_id, args.text)

    kwargs = {
        "page_size": args.limit,
        "page_number": args.page,
        "self_only": args.mine,
        "sort_by": "created_at" if args.newest else "task_count",
    }
    if args.search:
        kwargs["title"] = args.search
    if args.tag:
        kwargs["tags"] = args.tag
    if args.language:
        kwargs["language"] = args.language

    try:
        result = await tts_stream.client().voices.list(**kwargs)
    except Exception as exc:
        print(f"voices.list failed: {type(exc).__name__}: {exc}")
        return 1

    items = list(getattr(result, "items", []) or [])
    total = getattr(result, "total", len(items))

    if not items:
        where = "your account" if args.mine else "the public library"
        print(f"No voices found in {where} for that filter.")
        if args.mine:
            print("You have not cloned any voices yet. Drop --mine to browse public ones.")
        return 0

    print(f"{total:,} match; showing {len(items)} (page {args.page})\n")
    print(f"{'id':<34} {'languages':<12} {'uses':>9}   title")
    print("-" * 100)
    for v in items:
        vid = getattr(v, "id", None) or getattr(v, "_id", "?")
        title = (getattr(v, "title", None) or "untitled")[:38]
        langs = getattr(v, "languages", None) or []
        uses = getattr(v, "task_count", None)
        print(f"{vid:<34} {','.join(langs)[:12]:<12} "
              f"{(f'{uses:,}' if isinstance(uses, int) else '-'):>9}   {title}")

    print("\nHear one before committing:")
    print(f"    python scripts/list_voices.py --try {getattr(items[0], 'id', '<id>')}")
    print("\nThen set it in .env and restart the server:")
    print("    FISH_REFERENCE_ID=<the id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
