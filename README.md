# Fish Audio Text to Speech, and a Speedoy voice ordering bot

Two independent apps sharing a server.

| | page | socket | what it is |
|---|---|---|---|
| **Text to speech** | `/` | `/ws/{id}` | type, hear it spoken by Fish Audio |
| **Voice ordering** | `/order.html` | `/ws/order/{id}` | type or push-to-talk, it takes your order, reads it back in a Fish voice and places it |

Jump to [Ordering](#ordering) for the second. Both speak through the same Fish
Audio streaming path; only the ordering page opens a microphone (push-to-talk,
transcribed by Fish ASR), and only it talks to the Speedoy API.

---

## Text to speech

Type something; hear it spoken. Two modes:

- **Reply** (default) — an LLM answers your text, and the answer is streamed into
  Fish Audio's TTS websocket *while it is still being written*, so speech starts
  before the sentence is finished. Grounded on a local Chroma knowledge base.
- **Verbatim** — no AI at all. Whatever you type is spoken exactly as written.

```
you type ──► WebSocket (JSON)
                  │
   ┌──────────────▼──────────────┐
   │ RAG    Chroma + embeddings  │  skipped in verbatim mode / empty KB
   ├─────────────────────────────┤
   │ LLM    gpt-4o-mini, stream  │  token deltas
   ├─────────────────────────────┤
   │ buffer 4 words, then 8      │  never splits a word
   ├─────────────────────────────┤
   │ TTS    stream_websocket     │  text in, PCM out, concurrently
   └──────────────┬──────────────┘
                  │
 speakers ◄── AudioWorklet @44.1kHz ◄── WebSocket (binary PCM)
```

## Does the streaming actually help?

Measured on this account by `scripts/check_ws.py`, for a six-item answer:

| approach | time to first sound |
|---|---|
| stream tokens into TTS as they arrive | **2560 ms** |
| generate all text (1958 ms), then synthesise (1504 ms) | 3462 ms |
| **saving** | **902 ms — 26% faster** |

So the overlap is real and worth having. But note the honest caveats:

- **"Sub-300 ms" is Fish's TTS time-to-first-audio in isolation**, not the round
  trip. Measured here on `s2.1-pro-free`, synthesis warm-up alone is ~1000–1500 ms;
  the docs state the free tier carries **no TTFA guarantee**. The paid `s2.1-pro`
  is where the 300 ms figure lives.
- **On short replies streaming buys almost nothing**, because the LLM finishes
  before the synthesiser has warmed up. The gain scales with reply length.
- Every turn emits a `timing` event and the UI prints it, so you read real numbers
  rather than trusting a marketing figure.

## Setup

Requires **Python 3.10+**, a [Fish Audio](https://fish.audio) API key and an
OpenAI API key. The ordering bot additionally needs a Speedoy `ApiKey`.

```bash
cd TTS
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt

copy .env.example .env           # cp on POSIX
# fill in FISH_API_KEY and OPENAI_API_KEY

python scripts/check_sdk.py      # prints the live SDK surface
python scripts/check_live.py     # real API round trip, both keys
python -m rag.ingest             # index whatever is in docs/
python server.py                 # http://localhost:8010
```

## Configuration

All in `.env` — see [.env.example](.env.example). The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `FISH_API_KEY` | *(required)* | Fish Audio key — TTS, and ASR for push-to-talk |
| `OPENAI_API_KEY` | *(required)* | Chat completions and RAG embeddings |
| `FISH_TTS_MODEL` | `s2.1-pro-free` in `.env.example` (code falls back to `s2-pro`) | Free tier. Also `s2.1-pro`, `s2-pro`, `s1`. |
| `FISH_REFERENCE_ID` | *(unset)* | A **voice id**, not an API key. `GET /voices` lists yours. |
| `FISH_LATENCY` | `balanced` | vs `normal` — slower, better quality. |
| `FISH_SAMPLE_RATE` | `44100` | Verified honoured; `24000` halves the bytes on the wire. |
| `OPENAI_MODEL` | `gpt-4o-mini` | **Avoid reasoning models** — see below. |
| `OPENAI_TEMPERATURE` | *(unset)* | Blank sends no temperature; gpt-5 rejects non-default values. |
| `RAG_ENABLED` | `true` | Costs one embedding round trip (~550 ms) per turn. |
| `EMBED_MODEL` / `COLLECTION_NAME` | `text-embedding-3-small` / `voice_kb` | Embeddings model and Chroma collection |
| `STT_LANGUAGE` | *(blank = auto-detect)* | Fish ASR language for push-to-talk |
| `ORDER_MODEL` | `gpt-4.1-mini` | Model for the ordering bot — see [Notes](#notes) |
| `SPEEDOY_API_KEY` | *(required for ordering)* | Sent as the `ApiKey` header |
| `SPEEDOY_BASE_URL` / `SPEEDOY_RESTAURANT_ID` | dev API / demo restaurant | Which Speedoy API and restaurant to order from |
| `SPEEDOY_USER_TOKEN` | *(blank)* | Optional; only for reading placed orders back |
| `HOST` / `PORT` | `0.0.0.0` / `8010` | Where `python server.py` listens |

The rest (Fish sample rate/speed, chunking, history length, prompts, geocoder,
default delivery coordinates) are documented inline in `.env.example`.

### Two findings worth keeping

**The SDK's model list is only a type hint.** `fishaudio.types.shared.Model` is
`Literal['speech-1.5','speech-1.6','s1','s2-pro']`, but nothing validates it at
runtime — `resources/tts.py` passes the value straight through as an HTTP header
(`headers={"model": model}`). So newer ids like `s2.1-pro-free` work fine even
though this SDK build predates them. `tts_stream.resolve_model()` passes anything
through, warning only on unrecognised values.

**Don't use a reasoning model for the brain.** Measured with
`scripts/bench_llm.py` (median time-to-first-token, warm connection):

| model | TTFT |
|---|---|
| gpt-4.1-mini | 827 ms |
| gpt-4o-mini | 905 ms |
| gpt-5-nano | **3347 ms** |

`gpt-5-nano` thinks before it emits. Fine in a chat box, fatal in a conversation.

**Fish ASR needs paid credit.** If you ever add a microphone back, note that
`client.asr.transcribe()` returned HTTP 402 on this account — *"API credit is
managed independently from platform credit"*. The free tier covers TTS only.

## Layout

| Path | Role |
|---|---|
| [server.py](server.py) | FastAPI: both sockets, `/voices`, `/health`, `/menu` |
| [config.py](config.py) | every setting, read once from `.env` |
| [pipeline.py](pipeline.py) | one turn: retrieve → think → speak, with timings |
| [llm.py](llm.py) | streamed completions + the word buffering that sets TTFA |
| [tts_stream.py](tts_stream.py) | Fish `stream_websocket` wrapper |
| [rag/](rag/) | Chroma retrieval and the `docs/` ingest script |
| [order_pipeline.py](order_pipeline.py) | one ordering turn: think → call tools → speak |
| [stt.py](stt.py) | Fish ASR: raw mic PCM in, spoken words out |
| [tools.py](tools.py) | the ordering tools the model can call, and the ordering gate |
| [speedoy/](speedoy/) | Speedoy API client, menu catalog, cart, order payload |
| [static/](static/) | both pages, plus the recorder and playback AudioWorklets |
| [scripts/](scripts/) | SDK introspection, live checks, benchmarks, headless E2E |
| [docs/](docs/) | knowledge-base documents for the text page's RAG |
| `data/` | generated at runtime: `chroma_db/` index, sample voices, smoke-test audio (gitignored) |

`static/audio-playback.worklet.js` came from `../realtime/static/` unchanged — its
Int16→Float32 queue with a 20 ms fade-in is exactly what is needed here.

## Scripts

| Script | What it answers |
|---|---|
| `check_sdk.py` | What does the installed SDK actually expose? Run after any SDK bump. |
| `smoke_imports.py` | Offline: does everything import, and is the buffering correct? |
| `check_live.py` | Do both APIs work, and is `sample_rate` honoured? |
| `check_ws.py` | Full end-to-end over the websocket, headless. Needs the server running. |
| `bench_llm.py` | Which chat model reaches first token fastest? |
| `probe_rag.py` | What exactly is the model being asked, and what does it answer? |
| `check_stt.py` | Speaks a sentence with Fish TTS, then transcribes it with Fish ASR. |
| `check_speedoy.py` | Read-only: does the menu load, and do the lookups resolve? |
| `check_order_flow.py` | The ordering tools in text. `--place` sends a real order. |
| `check_order_ws.py` | The ordering bot end to end over `/ws/order`. Needs the server. `--order` places a real order. |
| `list_voices.py` | Browse Fish voices to pick a `FISH_REFERENCE_ID`; `--try <id>` saves a sample. |
| `probe_audio.py` | Saves the server's audio frames as a WAV, to tell a server fault from a browser one. |

## WebSocket protocol

`/ws/{session_id}` — the text-to-speech page:

| Direction | Frame | Payload |
|---|---|---|
| C→S | text | `text` (with optional `verbatim: true`), `interrupt`, `reset` |
| S→C | binary | s16le mono PCM @ `FISH_SAMPLE_RATE` |
| S→C | text | `ready`, `transcript`, `context`, `token`, `audio_start`, `audio_end`, `interrupted`, `timing`, `info`, `error` |

Sending a new message, or `interrupt`, cancels the in-flight turn: the async
generators unwind, the Fish socket closes, and the client posts `stop` to the
playback worklet so audio cuts immediately.

`/ws/order/{session_id}` — the ordering bot:

| Direction | Frame | Payload |
|---|---|---|
| C→S | text | `text`, `speech_start`, `speech_end`, `interrupt`, `reset` |
| C→S | binary | s16le mono PCM @ `STT_SAMPLE_RATE`, only between `speech_start` and `speech_end` (talk button held) |
| S→C | binary | s16le mono PCM @ `FISH_SAMPLE_RATE` |
| S→C | text | everything the text page sends, plus `transcribing`, `tool`, `cart`, `order_placed` |

`speech_start` cancels whatever is still being spoken, the same way a typed
message does; on `speech_end` the clip is transcribed and handled as a typed turn.

## Tuning latency

In rough order of impact:

1. **Move off the free TTS tier** — synthesis warm-up is the largest single term
   at ~1000–1500 ms, and `s2.1-pro-free` explicitly has no TTFA guarantee.
2. `FIRST_FLUSH_WORDS` in [llm.py](llm.py) — how many words before the first
   sound. Lower is snappier; too low sounds chopped.
3. `RAG_ENABLED=false` — removes an embedding round trip (~550 ms) per turn.
4. `OPENAI_MODEL` — see the table above.

---

# Ordering

Type an order; hear it back in your own Fish voice; it gets placed against the
Speedoy API. The same text-in, speech-out shape as the page above — the
difference is that this one can *act*.

```
you type ──► /ws/order/{id}
                  │
   ┌──────────────▼──────────────┐
   │ LLM    gpt-4o-mini + tools  │◄──┐  the menu rides in the system prompt
   ├─────────────────────────────┤   │
   │ tools  search, cart, order  │───┘  loop until the model stops asking
   │            │                │
   │            └──► Speedoy: menu · cart · place-order-by-phone-bot
   ├─────────────────────────────┤
   │ buffer 4 words, then 8      │  shared with the assistant page
   ├─────────────────────────────┤
   │ TTS    Fish stream_websocket│  FISH_REFERENCE_ID — your voice
   └──────────────┬──────────────┘
                  │
 speakers ◄── AudioWorklet @44.1kHz ◄── binary PCM
```

The whole menu — 18 categories, 135 orderable dishes — goes into the system
prompt as an 11 KB digest. That is deliberate rather than lazy: ordering needs
the exact set of dishes and prices on *every* turn, not the four chunks that
looked most similar to the last thing said, so RAG is the wrong tool here.
Chroma stays wired up for the assistant page and is untouched by this path.

**Speech spans the whole tool loop.** The model typically says a short "let me
check that" before a tool call and the real answer after it, and both belong to
one utterance — so a single Fish synthesis is fed from a queue that stays open
across every pass, rather than opening a websocket per pass and stitching the
audio together.

Open `http://localhost:8010/order.html` and type *"what have you got?"*, or
**hold the talk button** and say it. The cart panel fills in as the bot works,
which is how you catch a misread dish before the read-back rather than after.

## Speaking to it

**Fish ASR needs API credit that TTS credit does not cover.** Without it every
transcription returns HTTP 402 — the wiring is complete and
`scripts/check_stt.py` proves it end to end, but the button will report
*"Fish Audio has no API credit for speech recognition"* until the balance at
fish.audio/app/developers is topped up. Nothing else needs changing when it is.

**Push-to-talk, not silence detection.** Fish ASR is a batch endpoint: a
complete clip in, text out, with no streaming and no voice activity detection.
Releasing the button is an exact answer to *"have they finished talking?"*
where silence detection is a guess that cuts people off mid-sentence. Hold the
button, or hold the spacebar when the text box is not focused.

The mic records at `STT_SAMPLE_RATE` (16 kHz — higher rates are bytes the
recogniser discards) by asking the browser's AudioContext for that rate, so
Chrome resamples the 48 kHz microphone for free. The server accumulates the
frames, wraps them in a 44-byte RIFF header, and posts the clip.

Leaving `STT_LANGUAGE` blank lets Fish detect the language, which is what you
want when a caller switches between English and Swedish mid-order.

## Two rules that make it safe

**Names in, ids out.** The model never sees a Speedoy id. It says
"kebabtallrik" and [tools.py](tools.py) resolves it against the live catalog. A
model cannot hallucinate an id it is never shown, so a wrong dish surfaces as
*"which one did you mean?"* instead of a wrong order. Ambiguous matches come
back as a candidate list; the tool refuses rather than picking.

**The total is a gate, not a request.** `place_order` requires a
`confirmed_total` that matches the cart to the öre. The only way for the model
to know that number is to call `get_cart` — so reading the price back is
structurally necessary, not merely something the prompt asks for. It also
catches a cart that changed after the read-back. Add the checks on name, phone
and address, and an order cannot be placed that the caller has not heard priced.

## What the live API taught us

The Postman collection is a starting point, not a specification. Everything
here was established against the running dev API, and several points contradict
what the collection implies.

**The sample order is stale.** `Lamb Pelmeni`, its category, and its option ids
are all absent from the restaurant's current menu. Copy the *shape* from the
collection; take the *data* from `get-full-restaurant-details`.

**Options attach two different ways, and the awkward one dominates.** Nine
orderable dishes use a plain `OptionIds` list. Eighty-six — essentially all the
real food — use an `OptionTemplateId`, where a template carries one required
`PrimaryOption` (the Variant: Standard, Familj, Barn…) that has its own price,
plus a flattened cross-product table of `SecondryOptions`: one row per
(primary item × secondary group × secondary item) with an overridden price. So
extra tomato costs a different amount on a family pizza than on a child's, and
which extras exist depends on the size. That table is why the options response
is six megabytes for one restaurant.

**`TemplateSelectedPrimaryOptionItemId` must be left empty.** Setting it — even
to the correct, live option item id — makes `place-order-by-phone-bot` answer
HTTP 500 *"Object reference not set to an instance of an object"*, because the
server then walks the `OptionTemplate` object the payload sends as null. All
four combinations were tried against the live endpoint; the required choice goes
in the ordinary `Options[]` array, keyed by the template's primary group id, and
the server records it correctly there.

**A blank email is refused** as *"Profile not complete"* — with HTTP 200 and
`Success: false`, so the status code tells you nothing. Callers are not asked to
spell an address out loud; `SPEEDOY_DEFAULT_EMAIL` stands in.

**Delivery is charged by distance, and the retired tiers are still in the
payload.** `FixAmounts` holds two sets: three `Status: 2` rows (3/6/9 kr) and
three live `Status: 1` rows (5/10/15 kr). Reading the wrong set undercharges.
The tiers are upper bounds and anything past the last one pays the last one's
price, so an address 90 km from a Västerås restaurant pays 15, not 5. Quoting
this wrong is the expensive mistake — the caller agrees to one number and the
server bills another — so with no address yet the bot assumes the **top** tier
and corrects downward once it knows. Confirmed end to end: we quote 135, the
server banks 135.

**Two auth schemes, and the bot only has one.** The bot-facing endpoints take a
static `ApiKey` header. `getOrderdetail` and `fetch-menu-item-details` reject it
with 403 and want a per-user GUID in `token` plus a matching Bearer — a phone
bot has no user session. So reading an order back is development-only, behind
the optional `SPEEDOY_USER_TOKEN`. The bot never needs it to take an order.

**Two spellings are the API's, not typos.** `platfrom` at the top level and
`SpecialIntructions` inside `Address`. Correcting either means the server never
sees the value.

## Verifying it

```bash
python scripts/check_speedoy.py          # read-only: menu, lookups, digest size
python scripts/check_speedoy.py --menu   # print what the model is told
python scripts/check_order_flow.py       # the tools in text, stops before the POST
python scripts/check_order_flow.py --place   # ...and actually orders

python server.py
python scripts/check_order_ws.py         # the whole bot over its websocket
python scripts/check_order_ws.py --order # ...through to a placed order
```

Orders are **live**; there is no dry-run mode. `check_order_flow.py` and
`check_order_ws.py` therefore need an explicit flag before they will send one.

## Notes

- **The ordering bot needs a better brain than the assistant.** `ORDER_MODEL`
  is set separately from `OPENAI_MODEL` because `gpt-4o-mini` is not reliable
  enough to drive thirteen tools: it states totals from memory instead of
  calling `get_cart` — and gets them wrong, quoting 145 for a 135 order — and
  wanders off the ordering routine mid-turn. `gpt-4.1-mini` holds the flow, and
  per `bench_llm.py` it reaches first token *faster*, so it costs nothing.
  The `confirmed_total` gate caught every bad total, which is what it is for,
  but a gate that fires constantly is a bad experience.
- **`stream_reply` would have swallowed every tool call.** It reads only
  `delta.content`, which is right for the assistant page but drops
  `delta.tool_calls` silently. [llm.py](llm.py) gained `stream_with_tools` for
  this. Tool arguments arrive split across deltas and keyed by *index*, not id —
  only the first delta of each call carries the id and the name, so they have to
  be reassembled by index.
- **The assistant's tool-call turn must go back into the messages verbatim.**
  Append the `assistant` message with its `tool_calls` intact before the `tool`
  results, or the results have nothing to attach to and the request is rejected.
- **The tool loop is bounded.** `MAX_TOOL_PASSES = 5` in
  [order_pipeline.py](order_pipeline.py). A model that loops burns tokens in a
  tight circle; in practice a turn uses one or two passes.
- **Language follows the caller.** Reply in whatever is written, but say dish
  names exactly as the menu prints them. Ask in Swedish and it answers in
  Swedish, prices in kronor, dish names unchanged.
- **Latency is dominated by Fish's warm-up**, the same ~1000–1500 ms as the
  assistant page, plus a round trip per tool pass. Everything under
  [Tuning latency](#tuning-latency) applies here too.
