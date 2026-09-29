"""The cart, and the only place that knows Speedoy's wire format for an item.

Prices are assembled the way the menu implies: the dish's own Price, plus the
price of the required primary choice (a family pizza is the base plus 130), plus
each extra at whatever that extra costs *at that size*.

On the payload shape. The collection's one known-good MenuItems example uses the
plain `Options[]` array, and it turns out that is the only thing the server
accepts: a template item's required choice goes into `Options[]` exactly like
any other option, keyed by the template's primary option group id.

`TemplateSelectedPrimaryOptionItemId` must be left empty. Setting it -- even to
the correct, live option item id -- makes place-order-by-phone-bot answer
HTTP 500 "Object reference not set to an instance of an object", because the
server then tries to walk the `OptionTemplate` object that the payload sends as
null. Established by trying all four combinations against the live endpoint;
only the two with an empty template id are accepted, and of those only the one
below actually carries the caller's choice.
"""

import itertools
from dataclasses import dataclass, field

from .menu import Catalog, MenuItem, OptionGroup, OptionItem

# The API stamps these on every nested object. Reproduced from the collection.
EPOCH = "0001-01-01T00:00:00"
STATUS_ACTIVE = 1


@dataclass(frozen=True)
class Selection:
    """One chosen option item, remembering which group it came from."""

    group_id: str
    group_name: str
    item_id: str
    item_name: str
    price: float
    is_primary: bool = False


@dataclass
class CartLine:
    line_id: int
    item: MenuItem
    quantity: int = 1
    primary: Selection | None = None
    extras: list[Selection] = field(default_factory=list)
    special_instructions: str = ""

    @property
    def unit_price(self) -> float:
        price = self.item.price
        if self.primary:
            price += self.primary.price
        return price + sum(e.price for e in self.extras)

    @property
    def total(self) -> float:
        return round(self.unit_price * self.quantity, 2)

    def describe(self) -> str:
        """One line, phrased to be read aloud."""
        bits = [f"{self.quantity} x {self.item.name}" if self.quantity > 1 else self.item.name]
        if self.primary:
            bits.append(self.primary.item_name)
        if self.extras:
            bits.append("with " + ", ".join(e.item_name for e in self.extras))
        if self.special_instructions:
            bits.append(f"({self.special_instructions})")
        return " - ".join(bits) + f" - {self.total:g}"

    def to_payload(self) -> dict:
        """A MenuItems[] entry, matching the collection's sample field for field."""
        groups: dict[str, list[Selection]] = {}
        for selection in ([self.primary] if self.primary else []) + self.extras:
            groups.setdefault(selection.group_id, []).append(selection)

        options = [
            {
                "OptionId": group_id,
                "OptionItems": [
                    {
                        "OptionItemId": s.item_id,
                        "Name": s.item_name,
                        "Price": s.price,
                        "OptionItemTotal": 0,
                        "OptionItemTotalQuantity": 0,
                        "Quantity": 1,
                        "Id": s.item_id,
                        "CreateTime": EPOCH,
                        "UpdateTime": EPOCH,
                        "Status": STATUS_ACTIVE,
                    }
                    for s in selections
                ],
                "Id": None,
                "CreateTime": EPOCH,
                "UpdateTime": EPOCH,
                "Status": STATUS_ACTIVE,
            }
            for group_id, selections in groups.items()
        ]

        return {
            "MenuItemId": self.item.id,
            "Name": self.item.name,
            "Price": round(self.unit_price, 2),
            "UnitPrice": self.item.price,
            "MenuItemTotal": 0,
            "MenuItemGrandTotal": 0,
            "Quantity": self.quantity,
            "SpecialInstructions": self.special_instructions,
            "TaxId": self.item.tax_id,
            "TaxableAmount": 0,
            "Options": options,
            "OptionTemplate": None,
            # Must stay empty -- see the module docstring. The primary choice is
            # already in Options[] above, which is where the server reads it.
            "TemplateSelectedPrimaryOptionItemId": "",
            "TemplateSecondaryOptions": [],
            "TemplateSecondaryOptionsQuantity": [],
            "TemplateSelectedOptions": None,
            "SelectedOptions": None,
            "ItemGuid": None,
            "AllowedPromoCode": True,
            "ArticleNumber": "",
            "Department": "",
            "CategoryId": self.item.category_id,
            "ServiceInfoType": 0,
            "Id": None,
            "CreateTime": EPOCH,
            "UpdateTime": EPOCH,
            "Status": STATUS_ACTIVE,
        }


class AmbiguousChoice(ValueError):
    """A spoken name matched more than one thing. Carries the candidates so the
    bot can ask which, rather than picking one and being wrong."""

    def __init__(self, message: str, candidates: list[str]):
        super().__init__(message)
        self.candidates = candidates


class Cart:
    """One caller's order in progress."""

    def __init__(self) -> None:
        self.lines: list[CartLine] = []
        self._ids = itertools.count(1)

    # --- resolving spoken words to option items ------------------------------
    @staticmethod
    def _resolve_option(group: OptionGroup, spoken: str) -> OptionItem:
        hits = group.find(spoken)
        if not hits:
            raise ValueError(
                f"'{spoken}' is not one of the {group.name} choices. "
                f"Available: {', '.join(o.name for o in group.items)}."
            )
        if len(hits) > 1:
            raise AmbiguousChoice(
                f"'{spoken}' matches more than one {group.name} choice.",
                [o.name for o in hits],
            )
        return hits[0]

    def add(
        self,
        catalog: Catalog,
        item: MenuItem,
        quantity: int = 1,
        choice: str | None = None,
        extras: list[str] | None = None,
        special_instructions: str = "",
    ) -> CartLine:
        """Add one dish, resolving its choice and extras by name.

        Raises ValueError with a sentence the bot can say if a required choice
        is missing or a name does not resolve.
        """
        quantity = max(1, int(quantity or 1))
        template = catalog.template_for(item)
        primary: Selection | None = None
        secondary_groups: list[OptionGroup] = []

        if template:
            if not choice:
                names = ", ".join(o.name for o in template.primary.items if o.available)
                raise ValueError(
                    f"{item.name} needs a {template.primary.name} first. Choose one of: {names}."
                )
            option = self._resolve_option(template.primary, choice)
            primary = Selection(
                group_id=template.primary.id,
                group_name=template.primary.name,
                item_id=option.id,
                item_name=option.name,
                price=option.price,
                is_primary=True,
            )
            secondary_groups = template.secondary_groups(option.id)
        else:
            secondary_groups = catalog.plain_groups_for(item)
            if choice:
                # No template, but the caller named something -- try the plain groups.
                extras = [choice] + list(extras or [])

        chosen_extras: list[Selection] = []
        for spoken in extras or []:
            for group in secondary_groups:
                hits = group.find(spoken)
                if len(hits) == 1:
                    chosen_extras.append(
                        Selection(group.id, group.name, hits[0].id, hits[0].name, hits[0].price)
                    )
                    break
                if len(hits) > 1:
                    raise AmbiguousChoice(
                        f"'{spoken}' matches more than one extra.", [o.name for o in hits]
                    )
            else:
                available = ", ".join(g.name for g in secondary_groups) or "none"
                raise ValueError(
                    f"'{spoken}' is not available on {item.name}. Extra groups: {available}."
                )

        line = CartLine(
            line_id=next(self._ids),
            item=item,
            quantity=quantity,
            primary=primary,
            extras=chosen_extras,
            special_instructions=special_instructions or "",
        )
        self.lines.append(line)
        return line

    # --- mutation ------------------------------------------------------------
    def remove(self, line_id: int) -> CartLine | None:
        for index, line in enumerate(self.lines):
            if line.line_id == line_id:
                return self.lines.pop(index)
        return None

    def find_lines(self, spoken: str) -> list[CartLine]:
        from .menu import _match  # local import keeps the module's public surface small

        return _match(spoken, self.lines, lambda line: line.item.name)

    def clear(self) -> None:
        self.lines.clear()

    # --- totals --------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.lines

    def subtotal(self) -> float:
        return round(sum(line.total for line in self.lines), 2)

    def delivery_fee(
        self, catalog: Catalog, is_delivery: bool, distance_km: float | None = None
    ) -> float:
        """The delivery charge, picked from the restaurant's distance tiers.

        The tiers are cumulative upper bounds -- up to 1 km, up to 2 km, up to
        3 km -- and anything past the last one pays the last one's price. Only
        Status 1 tiers count; this restaurant carries a retired set alongside
        the live one, and reading the retired amounts would undercharge.

        Quoting this wrong is the expensive mistake: the caller agrees to a
        number and the server bills a different one. So with no address yet, we
        assume the top tier rather than the bottom. Over-quoting corrects itself
        the moment the address is known; under-quoting reaches the doorstep.
        """
        if not is_delivery:
            return 0.0

        charge = catalog.delivery_charge or {}
        free_from = float(charge.get("FreeDeliveryFromAmount") or 0)
        if free_from and self.subtotal() >= free_from:
            return 0.0

        active = [
            t for t in (charge.get("FixAmounts") or [])
            if t.get("Status") == STATUS_ACTIVE
        ]
        if not active:
            return 0.0

        tiers = sorted(active, key=lambda t: float(t.get("DistanceUpToKM") or 0))
        if distance_km is None:
            return float(tiers[-1].get("Amount") or 0)

        for tier in tiers:
            if distance_km <= float(tier.get("DistanceUpToKM") or 0):
                return float(tier.get("Amount") or 0)
        return float(tiers[-1].get("Amount") or 0)

    def total(
        self, catalog: Catalog, is_delivery: bool, distance_km: float | None = None
    ) -> float:
        """What the caller is told they will pay, and what they confirm."""
        return round(self.subtotal() + self.delivery_fee(catalog, is_delivery, distance_km), 2)

    def total_with_tax(
        self, catalog: Catalog, is_delivery: bool, distance_km: float | None = None
    ) -> float:
        """The tax-inclusive figure for TotalAmountWithTax.

        The restaurant reports IsExclusiveTax false -- menu prices already
        include Moms -- so this is the same number. Confirmed against a placed
        order: the server banked SubTotal 90 and TotalAmount 105 for a 90 kr
        dish with a 15 kr delivery, adding no tax on top.
        """
        return self.total(catalog, is_delivery, distance_km)

    # --- output --------------------------------------------------------------
    def to_payload(self) -> list[dict]:
        return [line.to_payload() for line in self.lines]

    def summary(
        self, catalog: Catalog, is_delivery: bool, distance_km: float | None = None
    ) -> dict:
        """A structured cart, for the model to read back and the UI to render."""
        fee = self.delivery_fee(catalog, is_delivery, distance_km)
        return {
            "lines": [
                {
                    "line_id": line.line_id,
                    "name": line.item.name,
                    "quantity": line.quantity,
                    "choice": line.primary.item_name if line.primary else None,
                    "extras": [e.item_name for e in line.extras],
                    "special_instructions": line.special_instructions,
                    "unit_price": round(line.unit_price, 2),
                    "line_total": line.total,
                    "spoken": line.describe(),
                }
                for line in self.lines
            ],
            "subtotal": self.subtotal(),
            "delivery_fee": fee,
            "total": self.total(catalog, is_delivery, distance_km),
            "currency": catalog.currency,
            "item_count": sum(line.quantity for line in self.lines),
        }
