"""The menu, normalised into something a language model can reason about.

Two calls build the whole catalog: get-full-restaurant-details for the items and
get-restaurant-options for the choices attached to them.

Speedoy attaches options to an item in two different ways, and the awkward one
is by far the more common:

  * OptionIds -- a plain list of option group ids. Nine orderable items use it.
  * OptionTemplateId -- 86 items, i.e. essentially all the real food. A template
    has one PrimaryOption (the Variant: Standard, Familj, Barn, Panpizza ...)
    which is a required choice that carries its own price, plus SecondryOptions:
    a flattened cross-product table where every row is
    (primary item, secondary group, secondary item) with an overridden Price.
    So "extra tomato" costs a different amount on a family pizza than on a
    child's, and the add-ons on offer depend on the size you picked. That table
    is why the options response is six megabytes for one restaurant.

Forty items -- drinks and ice cream -- have neither, and can be ordered as-is.

Everything here is read-only and cached; Catalog is rebuilt on a TTL.
"""

import asyncio
import logging
import time
import unicodedata
from dataclasses import dataclass, field

import config

from . import client

logger = logging.getLogger("speedoy.menu")


def normalise(text: str) -> str:
    """Fold case and strip diacritics, so 'Kebabratter' matches 'Kebabrätter'.

    Callers say dish names into a microphone; expecting the ring on a Swedish
    vowel to survive that round trip is optimistic.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


@dataclass(frozen=True)
class OptionItem:
    id: str
    name: str
    price: float
    available: bool = True
    default: bool = False


@dataclass(frozen=True)
class OptionGroup:
    id: str
    name: str
    items: tuple[OptionItem, ...]
    single_select: bool          # Type == 1
    max_selectable: int = 1
    required: bool = False       # a template's primary option

    def find(self, query: str) -> list[OptionItem]:
        return _match(query, self.items, lambda o: o.name)


@dataclass(frozen=True)
class Template:
    """A primary choice plus the add-ons available for each primary choice."""

    id: str
    name: str
    primary: OptionGroup
    # primary option item id -> secondary group id -> group (prices already
    # specialised to that primary choice)
    secondary: dict[str, dict[str, OptionGroup]] = field(default_factory=dict)

    def secondary_groups(self, primary_item_id: str) -> list[OptionGroup]:
        return list(self.secondary.get(primary_item_id, {}).values())


@dataclass(frozen=True)
class MenuItem:
    id: str
    name: str
    price: float
    category: str
    category_id: str
    description: str = ""
    allergens: str = ""
    available: bool = True
    option_ids: tuple[str, ...] = ()
    template_id: str | None = None
    # Kept for the order payload's tax fields.
    tax_id: str | None = None
    delivery_tax_id: str | None = None
    takeaway_tax_id: str | None = None
    tax_percent_delivery: float = 0.0
    tax_percent_takeaway: float = 0.0

    @property
    def needs_choice(self) -> bool:
        return self.template_id is not None or bool(self.option_ids)


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    items: tuple[MenuItem, ...]


def _match(query: str, candidates, key) -> list:
    """Rank candidates against a spoken query: exact, then prefix, then substring.

    Returns only the best tier that matched, so an exact hit never comes back
    alongside six loose ones and forces a needless "which did you mean?".
    """
    q = normalise(query)
    if not q:
        return []

    exact, prefix, substring, loose = [], [], [], []
    q_words = [w for w in q.split() if len(w) > 2]

    for candidate in candidates:
        name = normalise(key(candidate))
        if name == q:
            exact.append(candidate)
        elif name.startswith(q):
            prefix.append(candidate)
        elif q in name:
            substring.append(candidate)
        elif q_words and all(w in name for w in q_words):
            loose.append(candidate)

    return exact or prefix or substring or loose


class Catalog:
    """An immutable snapshot of one restaurant's menu."""

    def __init__(self, details: dict, options: dict):
        self.raw_restaurant = details
        self.name: str = details.get("Name") or "the restaurant"
        self.currency = "SEK"
        self.minimum_order = float(details.get("MinimumOrderAmount") or 0.0)
        self.estimated_delivery_minutes = float(details.get("EstimatedDeliveryTime") or 30.0)
        self.delivery_charge = details.get("DeliveryCharge") or {}
        # 2 == open, per the sample payloads.
        self.open_status = details.get("RestaurantOpenStatus")
        self.restaurant_address = details.get("Address") or {}

        self.option_groups: dict[str, OptionGroup] = {}
        self.templates: dict[str, Template] = {}
        self._load_options(options)

        self.categories: list[Category] = []
        self.items: dict[str, MenuItem] = {}
        self._load_items(details)

        self.fetched_at = time.monotonic()

    # --- construction --------------------------------------------------------
    def _load_options(self, options: dict) -> None:
        for raw in options.get("Options") or []:
            group = self._group(raw)
            self.option_groups[group.id] = group

        for raw in options.get("OptionTemplates") or []:
            if not raw.get("IsAvailable", True):
                continue
            primary_raw = raw.get("PrimaryOption") or {}
            if not primary_raw.get("OptionItems"):
                continue
            primary = self._group(primary_raw, required=True)

            # Collapse the cross-product rows into
            # primary item -> secondary group -> items, with the row's own price.
            buckets: dict[str, dict[str, dict]] = {}
            for row in raw.get("SecondryOptions") or []:
                if not row.get("IsAvailable", True):
                    continue
                primary_id = row.get("DefaultPrimaryOptionItemId")
                group_raw = row.get("DefaultSecondaryOption") or {}
                item_raw = row.get("DefaultSecondaryOptionItem") or {}
                group_id = row.get("DefaultSecondaryOptionId") or group_raw.get("Id")
                item_id = row.get("DefaultSecondaryOptionItemId") or item_raw.get("Id")
                if not (primary_id and group_id and item_id):
                    continue
                if not item_raw.get("IsAvailable", True):
                    continue

                slot = buckets.setdefault(primary_id, {}).setdefault(
                    group_id,
                    {"raw": group_raw, "items": {}},
                )
                # The row's Price is the effective price at this primary choice,
                # which is the whole point of the table.
                slot["items"][item_id] = OptionItem(
                    id=item_id,
                    name=item_raw.get("Name") or "",
                    price=float(row.get("Price") or item_raw.get("Price") or 0.0),
                    available=True,
                    default=bool(item_raw.get("IsSelectedByDefault")),
                )

            secondary: dict[str, dict[str, OptionGroup]] = {}
            for primary_id, groups in buckets.items():
                secondary[primary_id] = {
                    gid: OptionGroup(
                        id=gid,
                        name=(payload["raw"].get("Name") or "").strip(),
                        items=tuple(payload["items"].values()),
                        single_select=payload["raw"].get("Type") == 1,
                        max_selectable=int(payload["raw"].get("MaxSelectable") or 1),
                    )
                    for gid, payload in groups.items()
                }

            self.templates[raw["Id"]] = Template(
                id=raw["Id"],
                name=(raw.get("Name") or "").strip(),
                primary=primary,
                secondary=secondary,
            )

    @staticmethod
    def _group(raw: dict, required: bool = False) -> OptionGroup:
        items = tuple(
            OptionItem(
                id=o["Id"],
                name=(o.get("Name") or "").strip(),
                price=float(o.get("Price") or 0.0),
                available=bool(o.get("IsAvailable", True)),
                default=bool(o.get("IsSelectedByDefault")),
            )
            for o in (raw.get("OptionItems") or [])
            if o.get("Id") and o.get("IsAvailable", True)
        )
        return OptionGroup(
            id=raw.get("Id") or "",
            name=(raw.get("Name") or "").strip(),
            items=items,
            single_select=raw.get("Type") == 1,
            max_selectable=int(raw.get("MaxSelectable") or 1),
            required=required,
        )

    def _load_items(self, details: dict) -> None:
        for raw_cat in details.get("MenuCategories") or []:
            # An unavailable category takes its items with it -- two of the
            # twenty are off, and offering those dishes would be a dead end.
            if not raw_cat.get("IsAvailable", True):
                continue

            items = []
            for raw in raw_cat.get("MenuItems") or []:
                if not raw.get("IsAvailable", True):
                    continue
                item = MenuItem(
                    id=raw["Id"],
                    name=(raw.get("Name") or "").strip(),
                    price=float(raw.get("Price") or 0.0),
                    category=(raw_cat.get("Name") or "").strip(),
                    category_id=raw_cat["Id"],
                    description=(raw.get("Descriptions") or "").strip(),
                    allergens=(raw.get("Allergens") or "").strip(),
                    option_ids=tuple(raw.get("OptionIds") or ()),
                    template_id=raw.get("OptionTemplateId") or None,
                    tax_id=raw.get("TaxId"),
                    delivery_tax_id=raw.get("DeliveryTaxId"),
                    takeaway_tax_id=raw.get("TakeawayTaxId"),
                    tax_percent_delivery=float(raw.get("TaxPercentageForDelivery") or 0.0),
                    tax_percent_takeaway=float(raw.get("TaxPercentageForTakeaway") or 0.0),
                )
                items.append(item)
                self.items[item.id] = item

            if items:
                self.categories.append(
                    Category(id=raw_cat["Id"], name=(raw_cat.get("Name") or "").strip(),
                             items=tuple(items))
                )

    # --- lookups -------------------------------------------------------------
    def all_items(self) -> list[MenuItem]:
        return list(self.items.values())

    def find_items(self, query: str) -> list[MenuItem]:
        return _match(query, self.all_items(), lambda i: i.name)

    def find_category(self, query: str) -> Category | None:
        hits = _match(query, self.categories, lambda c: c.name)
        return hits[0] if len(hits) == 1 else (hits[0] if hits else None)

    def template_for(self, item: MenuItem) -> Template | None:
        return self.templates.get(item.template_id) if item.template_id else None

    def plain_groups_for(self, item: MenuItem) -> list[OptionGroup]:
        """Option groups from the OptionIds path, skipping dangling references.

        At least one live item points at an option id the options endpoint does
        not return, so a missing group is expected rather than exceptional.
        """
        groups = []
        for oid in item.option_ids:
            group = self.option_groups.get(oid)
            if group is None:
                logger.debug("item %s references unknown option group %s", item.id, oid)
                continue
            if group.items:
                groups.append(group)
        return groups

    def required_choice(self, item: MenuItem) -> OptionGroup | None:
        """The one choice that must be made before this item can be ordered."""
        template = self.template_for(item)
        return template.primary if template else None

    def transcription_keywords(self, limit: int = 100) -> list[str]:
        """Dish and variant names, fed to the transcriber as hints.

        Swedish menu items survive speech-to-text far better when the model has
        been told they are words.
        """
        seen: dict[str, None] = {}
        for category in self.categories:
            seen.setdefault(category.name, None)
        for item in self.all_items():
            seen.setdefault(item.name, None)
        for template in self.templates.values():
            for option in template.primary.items:
                seen.setdefault(option.name, None)
        words = [w for w in seen if w and len(w) > 2]
        return words[:limit]

    # --- the digest that goes into the model's instructions ------------------
    def compact_text(self) -> str:
        """The whole menu as spoken-language-friendly text.

        Item ids are deliberately absent: the model refers to dishes by name and
        the tool layer resolves them, so a hallucinated id can never reach the
        API. Variants are listed once per template rather than once per dish,
        which keeps 86 items from repeating the same thirteen sizes.
        """
        lines = [f"MENU FOR {self.name} (prices in {self.currency}):"]

        template_users: dict[str, list[str]] = {}
        for category in self.categories:
            lines.append("")
            lines.append(f"{category.name}:")
            for item in category.items:
                bits = [f"  {item.name} - {item.price:g}"]
                if item.description:
                    bits.append(f" ({item.description})")
                if item.template_id:
                    template = self.templates.get(item.template_id)
                    if template:
                        bits.append(f" [choose a {template.primary.name}]")
                        template_users.setdefault(item.template_id, []).append(item.name)
                elif item.option_ids:
                    names = [g.name for g in self.plain_groups_for(item)]
                    if names:
                        bits.append(f" [extras: {', '.join(names)}]")
                lines.append("".join(bits))

        if template_users:
            lines.append("")
            lines.append("REQUIRED CHOICES (the price shown is added to the dish):")
            for template_id in template_users:
                template = self.templates[template_id]
                choices = ", ".join(
                    f"{o.name} +{o.price:g}" if o.price else o.name
                    for o in template.primary.items
                    if o.available
                )
                lines.append(f"  {template.primary.name}: {choices}")

        lines.append("")
        lines.append(
            "Extra toppings and sauces are available on many dishes but their price "
            "depends on the size chosen, so always use get_item_options rather than "
            "quoting one from memory."
        )
        return "\n".join(lines)


# --- module-level cache ------------------------------------------------------
_catalog: Catalog | None = None
_lock = asyncio.Lock()


async def get_catalog(force: bool = False) -> Catalog:
    """The current catalog, refetched when it goes stale.

    The lock stops a burst of concurrent callers from each firing their own pair
    of requests at a cold cache.
    """
    global _catalog

    fresh = (
        _catalog is not None
        and not force
        and (time.monotonic() - _catalog.fetched_at) < config.MENU_TTL_SECONDS
    )
    if fresh:
        return _catalog  # type: ignore[return-value]

    async with _lock:
        if _catalog is not None and not force:
            if (time.monotonic() - _catalog.fetched_at) < config.MENU_TTL_SECONDS:
                return _catalog

        details, options = await asyncio.gather(
            client.get_full_restaurant_details(),
            client.get_restaurant_options(),
        )
        _catalog = Catalog(details or {}, options or {})
        logger.info(
            "menu loaded: %s -- %d categories, %d items, %d templates",
            _catalog.name, len(_catalog.categories), len(_catalog.items), len(_catalog.templates),
        )
        return _catalog


def cached_catalog() -> Catalog | None:
    """Whatever is in hand, without touching the network."""
    return _catalog
