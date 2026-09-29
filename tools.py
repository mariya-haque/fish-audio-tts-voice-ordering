"""The bot's hands: function tools the Realtime model can call, and one dispatcher.

Two rules shape everything here.

Names in, ids out. The model never sees or supplies a Speedoy id -- it says
"kebabtallrik" and this layer resolves it against the live catalog. A model
cannot hallucinate an id it is never shown, so a wrong dish becomes a
"which did you mean?" instead of a bad order.

Refuse rather than guess. Every tool returns {"ok": false, "error": "..."} with a
sentence the bot can say, and an ambiguous match comes back as a list of
candidates. The model is instructed to ask; this layer makes asking the only
option that works.
"""

import logging
from dataclasses import dataclass, field

from speedoy import client, menu, order
from speedoy.cart import AmbiguousChoice, Cart
from speedoy.order import CustomerDetails

logger = logging.getLogger("tools")


@dataclass
class OrderSession:
    """One caller: their cart, their details, and the order once placed."""

    cart: Cart = field(default_factory=Cart)
    customer: CustomerDetails = field(default_factory=CustomerDetails)
    placed_order_id: str | None = None
    placed: bool = False


def _ok(**payload) -> dict:
    return {"ok": True, **payload}


def _err(message: str, **extra) -> dict:
    return {"ok": False, "error": message, **extra}


def cart_summary(catalog: menu.Catalog, session: OrderSession) -> dict:
    """The cart priced for this caller, delivery distance included.

    Every quote the bot gives comes through here, so the number it reads aloud
    and the number it later confirms are computed the same way.
    """
    return session.cart.summary(
        catalog,
        session.customer.is_delivery,
        order.distance_km(session.customer, catalog),
    )


def _item_view(item: menu.MenuItem, catalog: menu.Catalog) -> dict:
    """One dish as the model should see it: names and prices, never ids."""
    view: dict = {
        "name": item.name,
        "price": item.price,
        "category": item.category,
    }
    if item.description:
        view["description"] = item.description
    if item.allergens:
        view["allergens"] = item.allergens

    template = catalog.template_for(item)
    if template:
        view["required_choice"] = {
            "name": template.primary.name,
            "options": [
                {"name": o.name, "extra_price": o.price}
                for o in template.primary.items
                if o.available
            ],
        }
    return view


# --- the tools ---------------------------------------------------------------
async def list_menu_categories(session: OrderSession, **_) -> dict:
    catalog = await menu.get_catalog()
    return _ok(
        restaurant=catalog.name,
        currency=catalog.currency,
        categories=[
            {"name": c.name, "item_count": len(c.items)} for c in catalog.categories
        ],
    )


async def browse_category(session: OrderSession, category: str = "", **_) -> dict:
    catalog = await menu.get_catalog()
    found = catalog.find_category(category)
    if not found:
        names = ", ".join(c.name for c in catalog.categories)
        return _err(f"There is no category called '{category}'. We have: {names}.")
    return _ok(
        category=found.name,
        currency=catalog.currency,
        items=[_item_view(i, catalog) for i in found.items],
    )


async def search_menu(session: OrderSession, query: str = "", **_) -> dict:
    catalog = await menu.get_catalog()
    hits = catalog.find_items(query)
    if not hits:
        return _err(
            f"Nothing on the menu matches '{query}'.",
            categories=[c.name for c in catalog.categories],
        )
    return _ok(
        currency=catalog.currency,
        count=len(hits),
        items=[_item_view(i, catalog) for i in hits[:12]],
    )


async def get_item_options(session: OrderSession, item: str = "", choice: str = "", **_) -> dict:
    """Everything that can be added to one dish.

    Extras on a templated dish are priced per size, so the caller's choice has
    to be known before the add-on prices mean anything. Without one, the
    required choice comes back on its own.
    """
    catalog = await menu.get_catalog()
    hits = catalog.find_items(item)
    if not hits:
        return _err(f"Nothing on the menu matches '{item}'.")
    if len(hits) > 1:
        return _err(
            f"'{item}' matches several dishes. Ask which one.",
            candidates=[i.name for i in hits[:8]],
        )

    found = hits[0]
    template = catalog.template_for(found)
    result: dict = {"item": found.name, "price": found.price, "currency": catalog.currency}

    if template:
        result["required_choice"] = {
            "name": template.primary.name,
            "options": [
                {"name": o.name, "extra_price": o.price}
                for o in template.primary.items
                if o.available
            ],
        }
        if not choice:
            result["note"] = (
                f"Extra prices depend on the {template.primary.name} chosen. "
                "Ask which one, then call this again with it."
            )
            return _ok(**result)

        selected = template.primary.find(choice)
        if len(selected) != 1:
            return _err(
                f"'{choice}' is not one of the {template.primary.name} choices.",
                candidates=[o.name for o in template.primary.items if o.available],
            )
        result["chosen"] = selected[0].name
        groups = template.secondary_groups(selected[0].id)
    else:
        groups = catalog.plain_groups_for(found)

    result["extras"] = [
        {
            "group": g.name,
            "pick_one": g.single_select,
            "options": [{"name": o.name, "price": o.price} for o in g.items],
        }
        for g in groups
    ]
    if not groups:
        result["extras_note"] = "This dish has no extras."
    return _ok(**result)


async def add_to_cart(
    session: OrderSession,
    item: str = "",
    quantity: int = 1,
    choice: str = "",
    extras: list | None = None,
    special_instructions: str = "",
    **_,
) -> dict:
    catalog = await menu.get_catalog()
    hits = catalog.find_items(item)
    if not hits:
        return _err(f"Nothing on the menu matches '{item}'.")
    if len(hits) > 1:
        return _err(
            f"'{item}' matches several dishes. Ask which one before adding it.",
            candidates=[i.name for i in hits[:8]],
        )

    try:
        line = session.cart.add(
            catalog,
            hits[0],
            quantity=quantity,
            choice=choice or None,
            extras=[e for e in (extras or []) if e],
            special_instructions=special_instructions,
        )
    except AmbiguousChoice as exc:
        return _err(str(exc), candidates=exc.candidates)
    except ValueError as exc:
        return _err(str(exc))

    return _ok(
        added=line.describe(),
        line_total=line.total,
        cart=cart_summary(catalog, session),
    )


async def remove_from_cart(session: OrderSession, item: str = "", **_) -> dict:
    catalog = await menu.get_catalog()
    matches = session.cart.find_lines(item)
    if not matches:
        return _err(f"There is no '{item}' in the order.")
    if len(matches) > 1:
        return _err(
            f"There are several '{item}' lines. Ask which one to remove.",
            candidates=[line.describe() for line in matches],
        )
    removed = session.cart.remove(matches[0].line_id)
    return _ok(
        removed=removed.describe() if removed else item,
        cart=cart_summary(catalog, session),
    )


async def clear_cart(session: OrderSession, **_) -> dict:
    catalog = await menu.get_catalog()
    session.cart.clear()
    return _ok(cleared=True, cart=cart_summary(catalog, session))


async def get_cart(session: OrderSession, **_) -> dict:
    """The cart and its total. The bot must call this before it can place anything."""
    catalog = await menu.get_catalog()
    summary = cart_summary(catalog, session)
    return _ok(
        cart=summary,
        order_type="delivery" if session.customer.is_delivery else "pickup",
        still_needed=session.customer.missing(),
    )


async def set_order_type(session: OrderSession, order_type: str = "", **_) -> dict:
    value = (order_type or "").strip().lower()
    if value in {"delivery", "deliver", "hemleverans"}:
        session.customer.is_delivery = True
    elif value in {"pickup", "takeaway", "collection", "avhamtning", "avhämtning"}:
        session.customer.is_delivery = False
    else:
        return _err("Order type must be either delivery or pickup.")

    catalog = await menu.get_catalog()
    return _ok(
        order_type="delivery" if session.customer.is_delivery else "pickup",
        cart=cart_summary(catalog, session),
        still_needed=session.customer.missing(),
    )


async def set_customer_details(
    session: OrderSession,
    first_name: str = "",
    last_name: str = "",
    phone: str = "",
    country_code: int = 0,
    email: str = "",
    **_,
) -> dict:
    customer = session.customer
    if first_name:
        first_name = first_name.strip()
        # People give their name as one string -- "I'm Test Caller" -- and the
        # model passes it straight through as first_name. The API rejects a
        # blank lastName outright, so split rather than making the bot ask
        # again for a surname it has already been told.
        if not last_name and " " in first_name:
            first_name, _, last_name = first_name.partition(" ")
        customer.first_name = first_name
    if last_name:
        customer.last_name = last_name.strip()
    if phone:
        # Callers say numbers with spaces and dashes in them.
        customer.phone = "".join(ch for ch in phone if ch.isdigit())
    if country_code:
        customer.country_code = int(country_code)
    if email:
        customer.email = email.strip()

    return _ok(
        name=customer.name,
        phone=customer.phone,
        still_needed=customer.missing(),
    )


async def set_delivery_address(
    session: OrderSession,
    address: str = "",
    apartment: str = "",
    label: str = "",
    instructions: str = "",
    **_,
) -> dict:
    customer = session.customer
    if address:
        customer.address = address.strip()
        customer.geocoded = False
    if apartment:
        customer.apartment = apartment.strip()
    if label:
        customer.label = label.strip()
    if instructions:
        customer.delivery_instructions = instructions.strip()

    if not customer.address:
        return _err("No address was given.")

    if not customer.geocoded:
        coordinates = await order.geocode(customer.address)
        customer.geocoded = True
        if coordinates:
            customer.latitude, customer.longitude = coordinates

    customer.is_delivery = True
    return _ok(
        address=customer.address,
        located=customer.latitude is not None,
        still_needed=customer.missing(),
    )


async def place_order(session: OrderSession, confirmed_total: float = 0, **_) -> dict:
    """Place the order for real.

    confirmed_total is the gate. The model can only know the right number by
    calling get_cart, and it is told to read that number aloud and get a yes
    before coming here -- so a mismatch means the caller was never told the
    real price, and the order is refused. It also catches the case where the
    cart changed after the read-back.
    """
    catalog = await menu.get_catalog()
    customer = session.customer

    if session.placed:
        return _err(
            "This order has already been placed.",
            order_id=session.placed_order_id,
        )
    if session.cart.is_empty:
        return _err("The order is empty. Add something first.")

    gaps = customer.missing()
    if gaps:
        return _err(f"Still needed before ordering: {', '.join(gaps)}.", still_needed=gaps)

    expected = session.cart.total(
        catalog, customer.is_delivery, order.distance_km(customer, catalog)
    )
    if abs(float(confirmed_total or 0) - expected) > 0.01:
        return _err(
            f"The confirmed total does not match the order, which comes to "
            f"{expected:g} {catalog.currency}. Read the order back to the caller "
            f"and confirm the correct total before placing it.",
            expected_total=expected,
        )

    try:
        result = await order.place(session.cart, customer, catalog)
    except client.SpeedoyError as exc:
        logger.warning("order rejected: %s", exc.message)
        return _err(f"The restaurant could not take the order: {exc.message}")

    session.placed = True
    session.placed_order_id = order.extract_order_id(result)
    logger.info("order placed: %s", session.placed_order_id)

    return _ok(
        order_id=session.placed_order_id,
        total=expected,
        currency=catalog.currency,
        estimated_minutes=catalog.estimated_delivery_minutes,
        order_type="delivery" if customer.is_delivery else "pickup",
        raw=result,
    )


async def check_order_status(session: OrderSession, order_id: str = "", **_) -> dict:
    target = order_id or session.placed_order_id
    if not target:
        return _err("No order has been placed in this call yet.")
    try:
        detail = await client.get_order_detail(target)
    except client.SpeedoyError as exc:
        return _err(f"Could not look that order up: {exc.message}")
    return _ok(order_id=target, detail=detail)


# --- registry ----------------------------------------------------------------
HANDLERS = {
    "list_menu_categories": list_menu_categories,
    "browse_category": browse_category,
    "search_menu": search_menu,
    "get_item_options": get_item_options,
    "add_to_cart": add_to_cart,
    "remove_from_cart": remove_from_cart,
    "clear_cart": clear_cart,
    "get_cart": get_cart,
    "set_order_type": set_order_type,
    "set_customer_details": set_customer_details,
    "set_delivery_address": set_delivery_address,
    "place_order": place_order,
    "check_order_status": check_order_status,
}

# Tools that change the cart, so the bridge knows when to push a UI update.
MUTATING = {"add_to_cart", "remove_from_cart", "clear_cart", "set_order_type", "place_order"}


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    """One tool in the shape chat completions wants: nested under "function"."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


_STRING = {"type": "string"}

SCHEMAS = [
    _tool(
        "list_menu_categories",
        "List the menu categories. Use this first when the caller asks what there is, "
        "then offer to go into one. Never recite every dish at once.",
        {},
    ),
    _tool(
        "browse_category",
        "List the dishes and prices in one category.",
        {"category": {**_STRING, "description": "The category name, e.g. 'Pizza klass 2'."}},
        ["category"],
    ),
    _tool(
        "search_menu",
        "Find dishes by name. Use this whenever the caller names something, to get the "
        "real name and price before you say them.",
        {"query": {**_STRING, "description": "What the caller called the dish."}},
        ["query"],
    ),
    _tool(
        "get_item_options",
        "The choices and extras available on one dish. Extra prices depend on the "
        "required choice, so pass the choice once the caller has made it.",
        {
            "item": {**_STRING, "description": "The dish name."},
            "choice": {**_STRING, "description": "The required choice, if already made."},
        },
        ["item"],
    ),
    _tool(
        "add_to_cart",
        "Add a dish to the order. If the dish has a required choice you must pass it, "
        "otherwise this returns the available choices for you to ask about.",
        {
            "item": {**_STRING, "description": "The dish name."},
            "quantity": {"type": "integer", "minimum": 1, "description": "How many. Defaults to 1."},
            "choice": {**_STRING, "description": "The required choice, e.g. a size or variant."},
            "extras": {
                "type": "array",
                "items": _STRING,
                "description": "Extra toppings, sauces or drinks, by name.",
            },
            "special_instructions": {**_STRING, "description": "Free text for the kitchen."},
        },
        ["item"],
    ),
    _tool(
        "remove_from_cart",
        "Take a dish back out of the order.",
        {"item": {**_STRING, "description": "The dish name to remove."}},
        ["item"],
    ),
    _tool("clear_cart", "Empty the order and start again.", {}),
    _tool(
        "get_cart",
        "The current order and its exact total. Call this before reading the order "
        "back, and use the total it returns as confirmed_total when placing.",
        {},
    ),
    _tool(
        "set_order_type",
        "Record whether this is delivery or pickup. Ask before collecting an address.",
        {"order_type": {"type": "string", "enum": ["delivery", "pickup"]}},
        ["order_type"],
    ),
    _tool(
        "set_customer_details",
        "Record the caller's name and phone number. Both a first name and a "
        "surname are required, so ask for a surname if they only give one name.",
        {
            "first_name": _STRING,
            "last_name": {**_STRING, "description": "Required; the order is refused without it."},
            "phone": {**_STRING, "description": "Digits only; spaces and dashes are fine."},
            "country_code": {"type": "integer", "description": "Defaults to 46 for Sweden."},
            "email": _STRING,
        },
    ),
    _tool(
        "set_delivery_address",
        "Record the delivery address. Sets the order to delivery.",
        {
            "address": {**_STRING, "description": "Street, number and city as spoken."},
            "apartment": _STRING,
            "label": {**_STRING, "description": "Home or Work."},
            "instructions": {**_STRING, "description": "Door codes, floor, where to leave it."},
        },
        ["address"],
    ),
    _tool(
        "place_order",
        "Place the order for real. Only call this after reading the full order and "
        "total back to the caller and hearing them agree. Pass the exact total from "
        "get_cart; a mismatch is refused.",
        {
            "confirmed_total": {
                "type": "number",
                "description": "The total from get_cart that the caller just agreed to.",
            }
        },
        ["confirmed_total"],
    ),
    _tool(
        "check_order_status",
        "Look up an order that has already been placed.",
        {"order_id": {**_STRING, "description": "Defaults to the order placed on this call."}},
        ["order_id"],
    ),
]


async def call_tool(name: str, arguments: dict, session: OrderSession) -> dict:
    """Run one tool. Never raises -- a crash here would strand the caller mid-call."""
    handler = HANDLERS.get(name)
    if handler is None:
        return _err(f"Unknown tool '{name}'.")
    try:
        return await handler(session, **(arguments or {}))
    except client.SpeedoyError as exc:
        logger.warning("%s failed against Speedoy: %s", name, exc.message)
        return _err(f"The ordering system had a problem: {exc.message}")
    except TypeError as exc:
        logger.warning("%s called with bad arguments %r: %s", name, arguments, exc)
        return _err(f"Those arguments did not fit {name}.")
    except Exception as exc:
        logger.exception("%s blew up", name)
        return _err(f"Something went wrong running {name}: {type(exc).__name__}.")
