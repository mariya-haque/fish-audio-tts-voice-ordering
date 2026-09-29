"""HTTP client for the Speedoy API.

Every endpoint here is transcribed from SpeedoyCollection.postman_collection2.json.
They are all POST with a JSON body -- including the ones that only read -- and
they all answer with the same envelope:

    {"Success": bool, "APIMessage": str, "APICode": int, "Data": ...}

The collection shows two auth schemes. The older endpoints want a per-user GUID
in a `token` header plus a matching `Bearer` and a session cookie. The
bot-facing ones -- get-full-restaurant-details, get-restaurant-options and
place-order-by-phone-bot -- take a single static `ApiKey` header instead, which
is the only scheme a server-side bot can actually hold. Those are the ones we
use; `SpeedoyError` carries `APIMessage` up so the bot can say what went wrong
in its own words.
"""

import logging
from functools import lru_cache
from typing import Any

import httpx

import config

logger = logging.getLogger("speedoy.client")


class SpeedoyError(RuntimeError):
    """A call reached Speedoy and came back unhappy.

    `message` is API prose meant to be spoken aloud; `code` is APICode.
    """

    def __init__(self, message: str, code: int | None = None, endpoint: str = ""):
        super().__init__(message)
        self.message = message
        self.code = code
        self.endpoint = endpoint


@lru_cache(maxsize=1)
def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=config.SPEEDOY_BASE_URL,
        timeout=config.SPEEDOY_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )


async def aclose() -> None:
    """Shut the connection pool down cleanly on server exit."""
    if _client.cache_info().currsize:
        await _client().aclose()
        _client.cache_clear()


async def _post(path: str, body: dict, headers: dict | None = None) -> Any:
    """POST, unwrap the envelope, and turn a failure into SpeedoyError."""
    if headers is None:
        if not config.SPEEDOY_API_KEY:
            raise SpeedoyError("SPEEDOY_API_KEY is not set.", endpoint=path)
        headers = {"ApiKey": config.SPEEDOY_API_KEY}

    try:
        response = await _client().post(path, json=body, headers=headers)
    except httpx.HTTPError as exc:
        raise SpeedoyError(f"Could not reach the ordering system: {exc}", endpoint=path) from exc

    if response.status_code >= 400:
        # The body usually says far more than the status does -- a 500 here is
        # often a validation complaint about one field, not an outage.
        detail = (response.text or "").strip()[:400]
        logger.warning("%s -> HTTP %s: %s", path, response.status_code, detail)
        raise SpeedoyError(
            f"The ordering system returned HTTP {response.status_code}."
            + (f" {detail}" if detail else ""),
            code=response.status_code,
            endpoint=path,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise SpeedoyError("The ordering system sent a response we could not read.", endpoint=path) from exc

    # Not every endpoint is guaranteed to use the envelope; tolerate a bare body.
    if not isinstance(payload, dict) or "Success" not in payload:
        return payload

    if not payload.get("Success"):
        raise SpeedoyError(
            payload.get("APIMessage") or "The ordering system rejected the request.",
            code=payload.get("APICode"),
            endpoint=path,
        )

    return payload.get("Data")


# --- endpoints ---------------------------------------------------------------
async def get_full_restaurant_details(restaurant_id: str | None = None) -> dict:
    """The whole restaurant: MenuCategories -> MenuItems, delivery charges, hours.

    One call returns the entire menu, so the bot never has to page through it.
    """
    rid = restaurant_id or config.SPEEDOY_RESTAURANT_ID
    return await _post("/restaurant/get-full-restaurant-details", {"RestaurantId": rid})


async def get_restaurant_options(restaurant_id: str | None = None) -> dict:
    """Every option group for the restaurant, as {"Options": [...], "OptionTemplates": [...]}.

    Menu items reference these by id in their OptionIds, so this is the other
    half of the menu -- sizes, extra toppings, drinks that come with a meal.
    """
    rid = restaurant_id or config.SPEEDOY_RESTAURANT_ID
    return await _post("/restaurant/get-restaurant-options", {"RestaurantId": rid})


async def place_order_by_phone_bot(payload: dict) -> Any:
    """Place a real order. This is the endpoint the collection provides for bots."""
    return await _post("/order/place-order-by-phone-bot", payload)


async def get_order_detail(order_id: str) -> Any:
    """Look an order back up, and see what the server made of our totals.

    This one is not on the ApiKey scheme -- it answers 403 to the bot's key and
    wants a per-user GUID in `token` plus a matching Bearer. A phone bot has no
    user session, so this is available only if SPEEDOY_USER_TOKEN is set, which
    is really a development convenience for verifying placed orders.
    """
    token = config.SPEEDOY_USER_TOKEN
    if not token:
        raise SpeedoyError(
            "Looking orders up needs a user token, which the bot does not have. "
            "Set SPEEDOY_USER_TOKEN to enable it.",
            code=403,
            endpoint="/order/getOrderdetail",
        )
    return await _post(
        "/order/getOrderdetail",
        {"OrderId": order_id},
        headers={
            "token": token,
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
