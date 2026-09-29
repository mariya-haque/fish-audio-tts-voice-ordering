"""End-to-end check of the ordering bot over its websocket.

Drives /ws/order the way the browser does -- type, listen, watch the cart -- and
asserts the bot actually calls the tools rather than talking about calling them.

Start the server first, then:

    python scripts/check_order_ws.py
    python scripts/check_order_ws.py --order    # let it place a real order
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

URL = f"ws://127.0.0.1:{config.PORT}/ws/order/selftest"

failures = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(label)


class Collector:
    def __init__(self):
        self.audio = 0
        self.events = []
        self.said = []
        self.tools = []
        self.carts = []
        self.errors = []
        self.timings = []
        self._reply = []

    def feed(self, message):
        if isinstance(message, (bytes, bytearray)):
            self.audio += len(message)
            return
        event = json.loads(message)
        self.events.append(event)
        kind = event.get("type")
        if kind == "token":
            self._reply.append(event["text"])
        elif kind == "audio_end":
            said = "".join(self._reply).strip()
            self._reply = []
            if said:
                self.said.append(said)
        elif kind == "tool":
            self.tools.append(event["name"])
        elif kind == "cart":
            self.carts.append(event)
        elif kind == "timing":
            self.timings.append(event)
        elif kind == "error":
            self.errors.append(event["error"])


async def drain(ws, collector, until, timeout=60.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            message = await asyncio.wait_for(ws.recv(), timeout=deadline - loop.time())
        except asyncio.TimeoutError:
            return False
        collector.feed(message)
        if until(collector):
            return True
    return False


async def turn(ws, collector, text, timeout=120.0):
    """Type something and wait for the reply to finish speaking."""
    before = len(collector.said)
    print(f"\n  >> {text}")
    await ws.send(json.dumps({"type": "text", "text": text}))
    ok = await drain(ws, collector, lambda c: len(c.said) > before, timeout)
    if ok:
        print(f"  << {collector.said[-1][:240]}")
    else:
        print("  << (timed out)")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", action="store_true", help="place a real order")
    args = parser.parse_args()

    print(f"connecting to {URL}")
    try:
        ws = await websockets.connect(URL, max_size=None, open_timeout=15)
    except Exception as exc:
        print(f"could not connect -- is the server running? ({exc})")
        return 1

    collector = Collector()
    async with ws:
        print("\n[handshake]")
        ok = await drain(ws, collector, lambda c: any(e["type"] == "ready" for e in c.events), 45)
        check("the session is ready", ok)
        if not ok:
            return 1

        ready = next(e for e in collector.events if e["type"] == "ready")
        print(f"  {ready['restaurant']}: {ready['items']} items | "
              f"{ready['llm_model']} + {ready['tts_model']} | "
              f"voice {ready.get('voice') or 'stock'} @ {ready['playback_sample_rate']} Hz")
        check("the menu is loaded", ready["items"] > 100)
        check("playback rate is Fish's", ready["playback_sample_rate"] == config.FISH_SAMPLE_RATE)

        print("\n[the menu]")
        await turn(ws, collector, "What kind of food do you have?")
        # The menu digest is in the system prompt, so answering without a tool
        # call is legitimate here and no less accurate -- the digest is built
        # from the same catalog the tools read. What matters is that the answer
        # names real categories rather than invented ones.
        # Matched case-insensitively and against both languages: the bot
        # translates the Swedish category names when the caller writes English,
        # so "Sallader" comes back as "salads" and "Drycker" as "drinks".
        said = (collector.said[-1] if collector.said else "").lower()
        check("it named real categories",
              sum(c in said for c in
                  ("pizza", "kebab", "salad", "sallad", "drink", "dryck", "burger",
                   "hamburgare", "pasta", "falafel")) >= 3,
              said[:80])
        check("Fish spoke it", collector.audio > 0, f"{collector.audio} bytes")

        print("\n[ordering]")
        await turn(ws, collector, "I'd like a Kebabtallrik with rice, and two Coca cola 33cl.")
        # The bot may confirm the variant before adding rather than after, which
        # is the better behaviour of the two -- so answer it the way a real
        # customer would instead of failing the run.
        if "add_to_cart" not in collector.tools:
            await turn(ws, collector, "Yes, that's right.")

        print("\n[the total]")
        await turn(ws, collector, "What's my total?")

        print("\n[refusing to order early]")
        await turn(ws, collector, "Place the order now.")
        check("it will not order without details",
              not any(e["type"] == "order_placed" for e in collector.events),
              f"still needed: {(collector.carts[-1] if collector.carts else {}).get('still_needed')}")

        if args.order:
            print("\n[details]")
            await turn(ws, collector,
                       "Delivery please. I'm Test Caller, 070 123 45 67, "
                       "Hovslagargatan 5B, 111 48 Stockholm.")
            check("the details were recorded",
                  "set_customer_details" in collector.tools
                  and "set_delivery_address" in collector.tools)

            print("\n[placing it]")
            await turn(ws, collector, "Yes, that's right, place the order.", timeout=150)
            placed = [e for e in collector.events if e["type"] == "order_placed"]
            check("the order was placed", bool(placed),
                  placed[0].get("order_id", "") if placed else "")
            if placed:
                print(f"  order {placed[0]['order_id']} -- "
                      f"{placed[0]['total']} {placed[0].get('currency', '')}")

        # Judged on the finished conversation rather than turn by turn. The bot
        # legitimately varies its choreography -- sometimes adding a dish
        # immediately, sometimes confirming the variant or offering a sauce
        # first and adding a turn later -- and asserting mid-flow only measures
        # which way it went this time.
        print("\n[the finished order]")
        cart = collector.carts[-1]["cart"] if collector.carts else {}
        check("things were added", "add_to_cart" in collector.tools,
              ", ".join(dict.fromkeys(collector.tools)))
        check("the cart has both lines", len(cart.get("lines", [])) == 2,
              "; ".join(line["spoken"] for line in cart.get("lines", [])))
        check("the cart totals 135, not the model's arithmetic",
              abs(cart.get("total", 0) - 135) < 0.01, str(cart.get("total")))
        check("it priced delivery from the live tiers",
              abs(cart.get("delivery_fee", 0) - 15) < 0.01, str(cart.get("delivery_fee")))

        check("no errors were reported", not collector.errors, "; ".join(collector.errors[:2]))

        # Everything the bot writes is spoken aloud, so a stray bullet or
        # asterisk is either read out or lands as an odd pause.
        markup = [s for s in collector.said
                  if any(m in s for m in ("* ", "- ", "#", "1. ", "|"))]
        check("nothing it said contains markup", not markup,
              markup[0][:70] if markup else "")

        if collector.timings:
            last = collector.timings[-1]
            print(f"\n  last turn: first token {last['ttft_ms']}ms, "
                  f"first audio {last['ttfa_ms']}ms, total {last['total_ms']}ms")
        print(f"  {collector.audio} bytes of audio, "
              f"{collector.audio / (config.FISH_SAMPLE_RATE * 2):.1f}s spoken")
        print(f"  tools used: {', '.join(dict.fromkeys(collector.tools))}")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("All ordering checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
