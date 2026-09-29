"""Drive the ordering tools in text, with no audio and no model in the loop.

This is the harness that proves the ordering half works before any of it is
trusted to a voice. It walks the same call_tool entry point the Realtime bridge
uses, so anything green here is green for the bot.

By default it stops just short of the POST and prints the payload. Pass --place
to actually order, which then reads the order back with getOrderdetail and
compares the server's totals against ours -- the one way to settle whether our
reading of the payload is right.

    python scripts/check_order_flow.py
    python scripts/check_order_flow.py --item "Coca cola 33cl"
    python scripts/check_order_flow.py --place
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import tools  # noqa: E402
from speedoy import client, menu, order  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  ' + detail if detail else ''}")


def show(name: str, result: dict) -> dict:
    flag = "ok " if result.get("ok") else "ERR"
    trimmed = {k: v for k, v in result.items() if k not in {"ok", "raw"}}
    text = json.dumps(trimmed, ensure_ascii=False, default=str)
    print(f"  {flag} {name:24} {text[:150]}")
    return result


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", default="Coca cola 33cl", help="a plain dish to order")
    parser.add_argument("--dish", default="Kebabtallrik", help="a dish that needs a choice")
    parser.add_argument("--place", action="store_true", help="actually place the order")
    args = parser.parse_args()

    if config.missing_order_keys():
        print(f"Missing credentials: {', '.join(config.missing_order_keys())}")
        return 1

    session = tools.OrderSession()
    catalog = await menu.get_catalog()
    print(f"{catalog.name}: {len(catalog.categories)} categories, {len(catalog.items)} items\n")

    # --- browsing ------------------------------------------------------------
    print("browsing")
    result = show("list_menu_categories", await tools.call_tool("list_menu_categories", {}, session))
    check("categories come back", result.get("ok") and len(result.get("categories", [])) > 5)

    result = show("search_menu", await tools.call_tool("search_menu", {"query": args.item}, session))
    check("the plain dish is found", result.get("ok") and result.get("count") == 1)

    result = show("search_menu(junk)", await tools.call_tool("search_menu", {"query": "zzzz"}, session))
    check("nonsense is refused", not result.get("ok"))

    # --- a dish that needs a choice -----------------------------------------
    print("\nrequired choices")
    result = show("get_item_options", await tools.call_tool("get_item_options", {"item": args.dish}, session))
    required = result.get("required_choice") or {}
    check("the required choice is reported", bool(required.get("options")),
          required.get("name", ""))

    result = show("add without choice", await tools.call_tool(
        "add_to_cart", {"item": args.dish}, session))
    check("adding without the choice is refused", not result.get("ok"))
    check("the refusal names the choices",
          "choose one of" in (result.get("error") or "").lower())

    choice_name = required["options"][0]["name"] if required.get("options") else ""
    result = show("add with choice", await tools.call_tool(
        "add_to_cart", {"item": args.dish, "choice": choice_name}, session))
    check("adding with the choice works", result.get("ok"), result.get("added", ""))

    # --- a plain dish, times two --------------------------------------------
    print("\nbuilding the order")
    result = show("add_to_cart", await tools.call_tool(
        "add_to_cart", {"item": args.item, "quantity": 2}, session))
    check("quantities work", result.get("ok"))

    result = show("get_cart", await tools.call_tool("get_cart", {}, session))
    cart = result.get("cart") or {}
    expected = round(sum(line["line_total"] for line in cart["lines"]) + cart["delivery_fee"], 2)
    check("the total adds up", abs(cart["total"] - expected) < 0.01,
          f"{cart['total']} vs {expected}")
    check("two lines are in the order", len(cart["lines"]) == 2)

    # --- the ordering gate ---------------------------------------------------
    print("\nthe ordering gate")
    result = show("place (no details)", await tools.call_tool(
        "place_order", {"confirmed_total": cart["total"]}, session))
    check("placing without a name or phone is refused", not result.get("ok"))

    show("set_order_type", await tools.call_tool(
        "set_order_type", {"order_type": "delivery"}, session))
    show("set_customer_details", await tools.call_tool("set_customer_details", {
        "first_name": "Test", "last_name": "Caller", "phone": "070 123 45 67",
        "email": "testuser@mailinator.com",
    }, session))
    result = show("set_delivery_address", await tools.call_tool("set_delivery_address", {
        "address": "Hovslagargatan 5B, 111 48 Stockholm, Sweden",
        "apartment": "12", "label": "Work",
    }, session))
    check("the address is accepted", result.get("ok"))
    check("nothing is outstanding", not (result.get("still_needed") or []))

    result = show("place (wrong total)", await tools.call_tool(
        "place_order", {"confirmed_total": 1.0}, session))
    check("a wrong confirmed total is refused", not result.get("ok"))
    check("the refusal states the real total", "expected_total" in result)

    # --- the payload ---------------------------------------------------------
    print("\npayload")
    payload = order.build_payload(session.cart, session.customer, catalog)
    check("platfrom is spelled the API's way", "platfrom" in payload)
    check("the address uses SpecialIntructions", "SpecialIntructions" in payload["Address"])
    check("every line carries a real menu item id",
          all(m["MenuItemId"] in catalog.items for m in payload["MenuItems"]))
    check("the chosen variant is in the options",
          any(m["Options"] for m in payload["MenuItems"]))
    check("TotalAmount matches the cart",
          abs(payload["TotalAmount"] - cart["total"]) < 0.01)

    first = payload["MenuItems"][0]
    print(f"  MenuItems[0]: {first['Name']} x{first['Quantity']} @ {first['Price']} "
          f"opts={len(first['Options'])} tpl={first['TemplateSelectedPrimaryOptionItemId'] or '-'}")
    print(f"  TotalAmount={payload['TotalAmount']}  WithTax={payload['TotalAmountWithTax']}  "
          f"delivery={payload['IsDeliveryOptionSelected']}")

    if not args.place:
        print("\n  (not placing -- pass --place to send this for real)")
        await client.aclose()
        print()
        print("all checks passed" if not failures else f"{failures} check(s) FAILED")
        return 1 if failures else 0

    # --- for real ------------------------------------------------------------
    print("\nPLACING THE ORDER FOR REAL")
    result = await tools.call_tool("place_order", {"confirmed_total": cart["total"]}, session)
    show("place_order", result)
    check("the order was accepted", result.get("ok"), result.get("error", ""))
    if not result.get("ok"):
        await client.aclose()
        return 1

    print(f"\n  raw response: {json.dumps(result.get('raw'), ensure_ascii=False, default=str)[:400]}")
    order_id = result.get("order_id")
    check("an order id came back", bool(order_id), str(order_id))

    result = show("place again", await tools.call_tool(
        "place_order", {"confirmed_total": cart["total"]}, session))
    check("the same order cannot be placed twice", not result.get("ok"))

    if order_id:
        print("\nreading the order back")
        status = await tools.call_tool("check_order_status", {"order_id": order_id}, session)
        if not status.get("ok"):
            print(f"  getOrderdetail unavailable: {status.get('error')}")
            print("  (the collection authenticates this one with a user token, not ApiKey)")
        else:
            detail = status.get("detail") or {}
            server_total = detail.get("TotalAmount")
            server_with_tax = detail.get("TotalAmountWithTax")
            print(f"  ours   TotalAmount={payload['TotalAmount']}  WithTax={payload['TotalAmountWithTax']}")
            print(f"  server TotalAmount={server_total}  WithTax={server_with_tax}")
            check("the server banked the total we quoted",
                  server_total is None or abs(float(server_total) - payload["TotalAmount"]) < 0.01,
                  "if this fails, the caller was told the wrong price")

    await client.aclose()
    print()
    print("all checks passed" if not failures else f"{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
