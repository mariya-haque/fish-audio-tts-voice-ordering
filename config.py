"""Central configuration, read once from the environment.

Plain os.getenv + load_dotenv, matching the style of the sibling apps in this repo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


# --- Fish Audio (the mouth, and optionally the ears) ------------------------
FISH_API_KEY = os.getenv("FISH_API_KEY", "").strip()
FISH_REFERENCE_ID = os.getenv("FISH_REFERENCE_ID", "").strip() or None

# NOTE: SDK 1.3.0 accepts only these model ids: speech-1.5, speech-1.6, s1, s2-pro.
# The public docs advertise "s2.1-pro", but this SDK build rejects it.
FISH_TTS_MODEL = os.getenv("FISH_TTS_MODEL", "s2-pro").strip()

# "balanced" ~300ms time-to-first-audio, "normal" ~500ms but higher quality.
FISH_LATENCY = os.getenv("FISH_LATENCY", "balanced").strip()

# Raw s16le mono PCM straight into the browser's AudioWorklet -- no decode step.
FISH_AUDIO_FORMAT = "pcm"
FISH_SAMPLE_RATE = _int("FISH_SAMPLE_RATE", 44100)
FISH_SPEED = float(os.getenv("FISH_SPEED", "1.0"))

# --- Fish Audio speech-to-text ----------------------------------------------
# Billed separately from TTS. An account with TTS credit but no API credit
# answers HTTP 402; see stt.py.
#
# 16 kHz mono is the standard rate for speech recognition -- higher rates are
# just bytes the recogniser throws away. The browser records at this rate.
STT_SAMPLE_RATE = _int("STT_SAMPLE_RATE", 16000)

# ISO-639-1, e.g. "en" or "sv". Blank lets Fish detect it, which is what you
# want when callers switch between English and Swedish mid-order.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "").strip()

# Longest single press of the talk button. A stuck button would otherwise
# stream until the socket died.
STT_MAX_SECONDS = _int("STT_MAX_SECONDS", 60)

# --- OpenAI (the brain, and optionally the ears) ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()

# The gpt-5 / reasoning families reject any temperature other than the default,
# so an unset value means "send no temperature field at all".
_temp_raw = os.getenv("OPENAI_TEMPERATURE", "").strip()
try:
    OPENAI_TEMPERATURE: float | None = float(_temp_raw) if _temp_raw else None
except ValueError:
    OPENAI_TEMPERATURE = None

# --- Brain / RAG ------------------------------------------------------------
RAG_ENABLED = _flag("RAG_ENABLED", True)
RAG_TOP_K = _int("RAG_TOP_K", 4)
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "voice_kb").strip()
CHUNK_SIZE = _int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 100)

MAX_HISTORY_TURNS = _int("MAX_HISTORY_TURNS", 8)

# Spoken output has different rules from screen output: no markdown, no lists,
# no code fences, short sentences. This prompt is the single biggest lever on
# how natural the agent sounds.
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip() or (
    "You are a helpful voice assistant. Your replies are spoken aloud, so write "
    "the way people talk.\n"
    "- Keep answers to two or three short sentences unless asked for more.\n"
    "- Plain prose only. Never use markdown, bullet points, numbered lists, "
    "headings, emoji, or code blocks.\n"
    "- Spell things out as they should be said: 'twenty percent', not '20%'.\n"
    "- If context from the knowledge base is provided, answer from it and say so "
    "naturally. If it does not contain the answer, say you do not have that "
    "information rather than guessing.\n"
    "- Get to the point in the first sentence; the user hears it before the rest."
)

# The ordering bot's persona. Deliberately separate from SYSTEM_PROMPT above,
# which caps replies at two or three sentences -- fine for Q&A, wrong for
# reading a menu aloud. The live menu is appended to this at connect time.
ORDER_SYSTEM_PROMPT = os.getenv("ORDER_SYSTEM_PROMPT", "").strip() or (
    "You are the phone assistant for a restaurant. You take orders by voice.\n"
    "\n"
    "LANGUAGE\n"
    "- Reply in whatever language the caller speaks, and switch if they switch.\n"
    "- Always say dish names exactly as they are printed on the menu, even when "
    "the rest of your sentence is in another language.\n"
    "\n"
    "HOW TO SPEAK\n"
    "- You are being heard, not read. Every word you write is spoken aloud by a "
    "speech synthesiser, so a bullet point or a dash is either read out or comes "
    "out as an odd pause. Plain prose only: no markdown, no bullet points, no "
    "numbered lists, no headings, no emoji. Join items with 'and', not newlines.\n"
    "- Say prices the way people say them, and name the currency once, not after "
    "every item.\n"
    "- Never read out an internal id. Callers hear names and prices only.\n"
    "- When asked for the menu, name the categories first and offer to go into "
    "one. Do not recite all of the items at once; that is unlistenable.\n"
    "- Keep to one question at a time and stay brief between tool calls.\n"
    "\n"
    "TAKING THE ORDER\n"
    "- Look items up with the tools. Never invent a dish, a price, or an option, "
    "and never guess whether something is available.\n"
    "- If a lookup returns more than one possible match, ask which one they meant "
    "instead of picking for them.\n"
    "- If an item has a required choice, such as a size or a variant, ask before "
    "adding it.\n"
    "- Confirm each item briefly as you add it, then ask what else they want.\n"
    "- Never say a total from memory. Any time you are about to state one, call "
    "get_cart first and read the number it returns. Totals go stale: adding an "
    "address can change the delivery fee, and a remembered figure will then be "
    "wrong and the order will be refused.\n"
    "\n"
    "CLOSING THE ORDER\n"
    "- Before placing anything, ask whether it is delivery or pickup, then collect "
    "their name and phone number, and their address if it is delivery.\n"
    "- Then call get_cart and read the order back as one flowing spoken "
    "sentence, never as a list: 'That's a Kebabtallrik with rice, and two Coca "
    "cola, ninety plus thirty, with fifteen for delivery, so one hundred and "
    "thirty-five kronor all together.' Then ask plainly whether to place it.\n"
    "- Only call place_order after the caller has clearly said yes. Pass the exact "
    "total you just read back. If they change anything, start the read-back again.\n"
    "- After it succeeds, tell them the order is placed and give the estimated "
    "time. If it fails, say what went wrong in plain words and offer to retry.\n"
    "\n"
    "Open the call by greeting them and asking what they would like to order."
)

# --- Speedoy ordering API ---------------------------------------------------
# Endpoints and auth taken from SpeedoyCollection.postman_collection2.json.
SPEEDOY_BASE_URL = os.getenv(
    "SPEEDOY_BASE_URL", "https://speedoy-api-dev.azurewebsites.net"
).strip().rstrip("/")

# Sent as a bare `ApiKey` header. The bot-facing endpoints
# (get-full-restaurant-details, get-restaurant-options, place-order-by-phone-bot)
# all use this instead of the per-user GUID tokens the older endpoints want.
SPEEDOY_API_KEY = os.getenv("SPEEDOY_API_KEY", "").strip()

# getOrderdetail is not on the ApiKey scheme -- it answers 403 to the bot's key
# and wants a per-user GUID. Optional, and only used to read placed orders back
# during development; the bot never needs it to take an order.
SPEEDOY_USER_TOKEN = os.getenv("SPEEDOY_USER_TOKEN", "").strip()

SPEEDOY_RESTAURANT_ID = os.getenv("SPEEDOY_RESTAURANT_ID", "5f3e6876bd7f2e2d8492c887").strip()
SPEEDOY_CUSTOMER_ID = os.getenv("SPEEDOY_CUSTOMER_ID", "5efd82a774f06456cc896ebf").strip()

# place-order-by-phone-bot rejects an order with a blank email as "Profile not
# complete". Spelling an address out loud is a miserable way to spend a phone
# call, so callers who do not offer one are booked against this.
SPEEDOY_DEFAULT_EMAIL = os.getenv("SPEEDOY_DEFAULT_EMAIL", "orders@thebotss.com").strip()

# Re-fetch the menu this often. The catalog is held in memory; a restaurant
# toggling an item off should reach the bot without a restart.
# The ordering bot's brain, kept separate from OPENAI_MODEL above.
#
# gpt-4o-mini is fine for the assistant page, which calls no tools, but it is
# not reliable enough to drive thirteen of them: it does arithmetic in its head
# rather than calling get_cart, and wanders off the ordering routine. Measured
# in scripts/bench_llm.py, gpt-4.1-mini also reaches first token *faster*
# (827ms vs 905ms), so this costs nothing in latency.
ORDER_MODEL = os.getenv("ORDER_MODEL", "gpt-4.1-mini").strip()

MENU_TTL_SECONDS = _int("MENU_TTL_SECONDS", 900)

SPEEDOY_TIMEOUT = _int("SPEEDOY_TIMEOUT", 30)

# Fallback coordinates when geocoding the caller's spoken address fails. The
# order payload requires a lat/long; the API does not appear to validate it
# tightly (the working sample pairs a Stockholm address with a Vasteras
# restaurant), but sending nothing is not an option.
SPEEDOY_DEFAULT_LAT = float(os.getenv("SPEEDOY_DEFAULT_LAT", "59.32986991484339"))
SPEEDOY_DEFAULT_LON = float(os.getenv("SPEEDOY_DEFAULT_LON", "18.078129776774492"))

# Nominatim is free and keyless but requires an identifying User-Agent.
GEOCODER_ENABLED = _flag("GEOCODER_ENABLED", True)
GEOCODER_USER_AGENT = os.getenv("GEOCODER_USER_AGENT", "speedoy-voice-bot/1.0").strip()
GEOCODER_COUNTRY_CODES = os.getenv("GEOCODER_COUNTRY_CODES", "se").strip()

# --- Server -----------------------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0").strip()
PORT = _int("PORT", 8010)
STATIC_DIR = BASE_DIR / "static"


def missing_keys() -> list[str]:
    """Names of required credentials that are absent, for a clear startup error."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not FISH_API_KEY:
        missing.append("FISH_API_KEY")
    return missing


def missing_order_keys() -> list[str]:
    """Credentials the ordering bot needs. Kept apart from missing_keys() so the
    Fish Audio text page still starts cleanly without any Speedoy setup."""
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not SPEEDOY_API_KEY:
        missing.append("SPEEDOY_API_KEY")
    if not SPEEDOY_RESTAURANT_ID:
        missing.append("SPEEDOY_RESTAURANT_ID")
    if not FISH_API_KEY:
        missing.append("FISH_API_KEY")
    return missing
