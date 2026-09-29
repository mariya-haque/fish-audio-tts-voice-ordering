"""FastAPI host for two text-to-speech apps.

Both work the same way -- you type, an LLM answers, and the answer is spoken by
Fish Audio while it is still being written. Text goes up as JSON, audio comes
back as binary frames.

  /ws/{id}        the general assistant, grounded on the Chroma knowledge base
  /ws/order/{id}  the ordering bot, which can browse the menu and place orders

  client -> server   text     text | interrupt | reset
  server -> client   binary   raw s16le PCM at FISH_SAMPLE_RATE
  server -> client   text     ready | transcript | context | token | audio_start |
                              audio_end | interrupted | timing | error | info
                              plus, on the ordering socket: tool | cart |
                              order_placed
"""

import asyncio
import contextlib
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import order_pipeline
import pipeline
import stt
import tools as ordering_tools
import tts_stream
from rag import engine as rag
from speedoy import client as speedoy_client
from speedoy import menu as speedoy_menu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("server")

@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    """Fetch the menu before the first caller, not during their greeting."""
    if config.missing_order_keys():
        logger.info("Speedoy not configured; the ordering bot is unavailable.")
    else:
        try:
            catalog = await speedoy_menu.get_catalog()
            logger.info("Menu ready: %s, %d items", catalog.name, len(catalog.items))
        except Exception as exc:
            logger.warning("Could not preload the menu: %s", exc)
    yield
    await speedoy_client.aclose()


app = FastAPI(title="Fish Audio Text-to-Speech Agent", lifespan=lifespan)


class Session:
    """Per-connection state: rolling history and the in-flight turn."""

    def __init__(self, websocket: WebSocket, session_id: str):
        self.ws = websocket
        self.id = session_id
        self.history: list[dict] = []
        self.turn: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    # --- sinks handed to the pipeline ---------------------------------------
    async def send_json(self, payload: dict) -> None:
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.ws.send_text(json.dumps(payload))

    async def send_audio(self, chunk: bytes) -> None:
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.ws.send_bytes(chunk)

    # --- turn lifecycle ------------------------------------------------------
    async def cancel_turn(self, notify: bool = True) -> bool:
        """Stop any in-flight response, so a new message cuts off the old reply."""
        task = self.turn
        if task is None or task.done():
            return False

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        self.turn = None

        if notify:
            await self.send_json({"type": "interrupted"})
        logger.info("[%s] turn interrupted", self.id)
        return True

    def start_turn(self, question: str, speak_verbatim: bool = False) -> None:
        if not speak_verbatim:
            self.history.append({"role": "user", "content": question})

        async def run() -> None:
            try:
                reply = await pipeline.respond(
                    question,
                    self.history[:-1] if not speak_verbatim else [],
                    self.send_json,
                    self.send_audio,
                    speak_verbatim=speak_verbatim,
                )
                if reply and not speak_verbatim:
                    self.history.append({"role": "assistant", "content": reply})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("[%s] turn failed", self.id)
                await self.send_json({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

        self.turn = asyncio.create_task(run())


@app.get("/health")
async def health():
    return {
        "ok": not config.missing_keys(),
        "missing_keys": config.missing_keys(),
        "llm_model": config.OPENAI_MODEL,
        "tts_model": tts_stream.resolve_model(),
        "reference_id": config.FISH_REFERENCE_ID,
        "sample_rate": config.FISH_SAMPLE_RATE,
        "rag_enabled": config.RAG_ENABLED,
        "kb_chunks": await asyncio.to_thread(rag.collection_count),
    }


@app.get("/voices")
async def voices():
    """List the voices on the Fish account, for picking a FISH_REFERENCE_ID."""
    if not config.FISH_API_KEY:
        return JSONResponse({"error": "FISH_API_KEY is not set"}, status_code=400)
    try:
        result = await tts_stream.client().voices.list()
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)

    items = getattr(result, "items", None) or getattr(result, "data", None) or result
    out = []
    for v in items:
        out.append(
            {
                "id": getattr(v, "id", None) or getattr(v, "_id", None),
                "title": getattr(v, "title", None) or getattr(v, "name", None),
                "visibility": getattr(v, "visibility", None),
            }
        )
    return {"voices": out}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = Session(websocket, session_id)
    logger.info("[%s] connected", session_id)

    missing = config.missing_keys()
    await session.send_json(
        {
            "type": "ready",
            "playback_sample_rate": config.FISH_SAMPLE_RATE,
            "llm_model": config.OPENAI_MODEL,
            "tts_model": tts_stream.resolve_model(),
            "voice": config.FISH_REFERENCE_ID,
            "rag_enabled": config.RAG_ENABLED,
            "kb_chunks": await asyncio.to_thread(rag.collection_count),
            "missing_keys": missing,
        }
    )
    if missing:
        await session.send_json(
            {"type": "error", "error": f"Missing credentials: {', '.join(missing)}. See .env.example."}
        )

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await session.send_json({"type": "error", "error": "Malformed JSON frame."})
                continue

            kind = event.get("type")

            if kind == "text":
                question = (event.get("text") or "").strip()
                if not question:
                    continue
                # A new message supersedes whatever is still being spoken.
                await session.cancel_turn()
                verbatim = bool(event.get("verbatim"))
                await session.send_json({"type": "transcript", "role": "user", "text": question})
                session.start_turn(question, speak_verbatim=verbatim)

            elif kind == "interrupt":
                await session.cancel_turn()

            elif kind == "reset":
                await session.cancel_turn(notify=False)
                session.history.clear()
                await session.send_json({"type": "info", "info": "history_cleared"})

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] websocket loop failed", session_id)
    finally:
        await session.cancel_turn(notify=False)
        logger.info("[%s] disconnected", session_id)


class OrderConnection(Session):
    """A Session that can also take an order.

    Inherits the sinks, the cancellation behaviour and the history handling from
    Session, and adds the cart the ordering tools mutate.
    """

    def __init__(self, websocket: WebSocket, session_id: str):
        super().__init__(websocket, session_id)
        self.order = ordering_tools.OrderSession()
        # Mic samples accumulate here while the talk button is held.
        self.speech: bytearray = bytearray()
        self.recording = False

    @property
    def _max_speech_bytes(self) -> int:
        return config.STT_MAX_SECONDS * config.STT_SAMPLE_RATE * stt.SAMPLE_WIDTH

    def start_turn(self, question: str, speak_verbatim: bool = False) -> None:
        self.history.append({"role": "user", "content": question})

        async def run() -> None:
            try:
                reply = await order_pipeline.respond(
                    question,
                    self.history[:-1],
                    self.order,
                    self.send_json,
                    self.send_audio,
                )
                if reply:
                    self.history.append({"role": "assistant", "content": reply})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("[%s] order turn failed", self.id)
                await self.send_json({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

        self.turn = asyncio.create_task(run())

    async def push_cart(self, catalog) -> None:
        await self.send_json({
            "type": "cart",
            "cart": ordering_tools.cart_summary(catalog, self.order),
            "order_type": "delivery" if self.order.customer.is_delivery else "pickup",
            "still_needed": self.order.customer.missing(),
            "order_id": self.order.placed_order_id,
        })


@app.websocket("/ws/order/{session_id}")
async def order_endpoint(websocket: WebSocket, session_id: str):
    """The ordering bot: type an order, hear it back, and it gets placed.

    Separate from /ws above, which is the general assistant and is unchanged.
    """
    await websocket.accept()
    session = OrderConnection(websocket, session_id)
    logger.info("[%s] order connected", session_id)

    missing = config.missing_order_keys()
    if missing:
        await session.send_json({
            "type": "error",
            "error": f"Missing credentials: {', '.join(missing)}. See .env.example.",
        })
        await websocket.close()
        return

    try:
        catalog = await speedoy_menu.get_catalog()
    except Exception as exc:
        logger.exception("[%s] could not load the menu", session_id)
        await session.send_json({"type": "error", "error": f"Could not load the menu: {exc}"})
        await websocket.close()
        return

    await session.send_json({
        "type": "ready",
        "playback_sample_rate": config.FISH_SAMPLE_RATE,
        "mic_sample_rate": config.STT_SAMPLE_RATE,
        "llm_model": config.ORDER_MODEL,
        "tts_model": tts_stream.resolve_model(),
        "voice": config.FISH_REFERENCE_ID,
        "restaurant": catalog.name,
        "categories": len(catalog.categories),
        "items": len(catalog.items),
        "currency": catalog.currency,
    })
    await session.push_cart(catalog)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Mic samples, while the talk button is held.
            chunk = message.get("bytes")
            if chunk is not None:
                if session.recording and len(session.speech) < session._max_speech_bytes:
                    session.speech.extend(chunk)
                continue

            raw = message.get("text")
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await session.send_json({"type": "error", "error": "Malformed JSON frame."})
                continue

            kind = event.get("type")

            if kind == "text":
                question = (event.get("text") or "").strip()
                if not question:
                    continue
                await session.cancel_turn()
                await session.send_json({"type": "transcript", "role": "user", "text": question})
                session.start_turn(question)

            elif kind == "speech_start":
                # Talking supersedes whatever is still being spoken, the same
                # way a typed message does.
                await session.cancel_turn()
                session.speech.clear()
                session.recording = True

            elif kind == "speech_end":
                session.recording = False
                audio = bytes(session.speech)
                session.speech.clear()
                await session.send_json({
                    "type": "transcribing",
                    "seconds": round(stt.duration_seconds(audio), 1),
                })
                try:
                    heard = await stt.transcribe(audio)
                except stt.SpeechUnavailable as exc:
                    await session.send_json({"type": "error", "error": str(exc)})
                    continue
                if not heard:
                    await session.send_json({
                        "type": "info", "info": "nothing_heard",
                    })
                    continue
                await session.send_json({"type": "transcript", "role": "user", "text": heard})
                session.start_turn(heard)

            elif kind == "interrupt":
                await session.cancel_turn()

            elif kind == "reset":
                await session.cancel_turn(notify=False)
                session.history.clear()
                session.order = ordering_tools.OrderSession()
                session.speech.clear()
                session.recording = False
                await session.push_cart(catalog)
                await session.send_json({"type": "info", "info": "order_cleared"})

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("[%s] order websocket failed", session_id)
    finally:
        await session.cancel_turn(notify=False)
        logger.info("[%s] order disconnected", session_id)


@app.get("/menu")
async def menu_debug():
    """What the ordering bot knows, for eyeballing without opening a socket."""
    if config.missing_order_keys():
        return JSONResponse(
            {"error": f"Missing: {', '.join(config.missing_order_keys())}"}, status_code=400
        )
    try:
        catalog = await speedoy_menu.get_catalog()
    except speedoy_client.SpeedoyError as exc:
        return JSONResponse({"error": exc.message}, status_code=502)

    return {
        "restaurant": catalog.name,
        "currency": catalog.currency,
        "open_status": catalog.open_status,
        "estimated_delivery_minutes": catalog.estimated_delivery_minutes,
        "categories": [
            {"name": c.name, "items": [{"name": i.name, "price": i.price,
                                        "needs_choice": i.needs_choice} for i in c.items]}
            for c in catalog.categories
        ],
        "digest_chars": len(catalog.compact_text()),
    }


# Mounted last so the API routes above win.
config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")


def _serve() -> None:
    """Run uvicorn, listening on IPv4 *and* IPv6 where the OS allows it.

    Windows resolves `localhost` to ::1 before 127.0.0.1, but a plain
    host="0.0.0.0" bind is IPv4-only -- so http://localhost:8010 gets
    ERR_CONNECTION_REFUSED while http://127.0.0.1:8010 works. Binding "::"
    instead just inverts the problem. A dual-stack socket serves both.
    """
    import socket

    import uvicorn

    wildcard = config.HOST in {"0.0.0.0", "::", ""}

    if wildcard and socket.has_dualstack_ipv6():
        server = uvicorn.Server(uvicorn.Config(app, host=config.HOST, port=config.PORT))
        sock = socket.create_server(
            ("::", config.PORT), family=socket.AF_INET6, dualstack_ipv6=True, reuse_port=False
        )
        logger.info("Listening on http://localhost:%d and http://127.0.0.1:%d", config.PORT, config.PORT)
        server.run(sockets=[sock])
    else:
        if wildcard:
            logger.warning(
                "No dual-stack IPv6 on this machine; binding IPv4 only. "
                "Use http://127.0.0.1:%d if http://localhost:%d is refused.",
                config.PORT, config.PORT,
            )
        uvicorn.run(app, host=config.HOST or "0.0.0.0", port=config.PORT)


if __name__ == "__main__":
    if config.missing_keys():
        logger.warning(
            "Starting without %s -- the UI will load but turns will fail. Copy .env.example to .env.",
            ", ".join(config.missing_keys()),
        )

    _serve()
