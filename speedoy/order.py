"""Turning a cart plus a caller's details into a place-order-by-phone-bot body.

The field names come from the collection verbatim, including the two the API
misspells -- `platfrom` at the top level and `SpecialIntructions` inside the
address. They are the API's spelling, not typos to correct: fixing them would
simply mean the server never sees the value.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx

import config

from . import client
from .cart import EPOCH, STATUS_ACTIVE, Cart
from .menu import Catalog

logger = logging.getLogger("speedoy.order")

# From the collection's sample payloads.
ORDER_TYPE_DELIVERY = 2
PAYMENT_METHOD_CASH = 1
PLATFORM_PHONE_BOT = 2
TAKEAWAY_OPTION = 2


@dataclass
class CustomerDetails:
    """What the bot collects by voice. Blank until the caller says it."""

    first_name: str = ""
    last_name: str = ""
    phone: str = ""
    country_code: int = 46          # Sweden; the restaurant is in Vasteras
    email: str = ""
    address: str = ""
    apartment: str = ""
    label: str = "Home"
    delivery_instructions: str = ""
    latitude: float | None = None
    longitude: float | None = None
    is_delivery: bool = True
    order_notes: str = ""
    # Set once geocoding has been attempted, so we do not retry on every turn.
    geocoded: bool = field(default=False, repr=False)

    @property
    def name(self) -> str:
        return " ".join(p for p in (self.first_name, self.last_name) if p).strip()

    def missing(self) -> list[str]:
        """Human-readable names of what still has to be asked for.

        first name, last name, phone and email are all mandatory: leave any one
        of them blank and place-order-by-phone-bot answers "Profile not
        complete" -- with HTTP 200, so the status code gives no warning. Better
        to notice here, where the bot can simply ask, than at the moment the
        caller has already said yes. Email is not listed because it falls back
        to SPEEDOY_DEFAULT_EMAIL rather than being asked for.
        """
        gaps = []
        if not self.first_name:
            gaps.append("first name")
        if not self.last_name:
            gaps.append("last name")
        if not self.phone:
            gaps.append("phone number")
        if self.is_delivery and not self.address:
            gaps.append("delivery address")
        return gaps


async def geocode(address: str) -> tuple[float, float] | None:
    """Spoken address -> coordinates, via Nominatim.

    Free and keyless, but it insists on an identifying User-Agent and will
    happily return nothing. A failure here is not fatal: the order payload needs
    *a* coordinate pair and the API does not appear to check it closely -- the
    collection's own working sample pairs a Stockholm address with a Vasteras
    restaurant.
    """
    if not (config.GEOCODER_ENABLED and address.strip()):
        return None

    params = {"q": address, "format": "json", "limit": 1}
    if config.GEOCODER_COUNTRY_CODES:
        params["countrycodes"] = config.GEOCODER_COUNTRY_CODES

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            response = await http.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers={"User-Agent": config.GEOCODER_USER_AGENT},
            )
            response.raise_for_status()
            results = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("geocoding %r failed: %s", address, exc)
        return None

    if not results:
        logger.info("geocoder found nothing for %r", address)
        return None

    try:
        return float(results[0]["lat"]), float(results[0]["lon"])
    except (KeyError, TypeError, ValueError):
        return None


def distance_km(customer: CustomerDetails, catalog: Catalog) -> float | None:
    """How far the food has to travel, which is what sets the delivery fee.

    None when either end is unknown, which the fee calculation reads as
    "assume the most expensive tier" rather than "assume free".
    """
    address = catalog.restaurant_address or {}
    origin_lat, origin_lon = address.get("Latitude"), address.get("Longitude")
    if origin_lat is None or origin_lon is None:
        return None
    if customer.latitude is None or customer.longitude is None:
        return None

    radius = 6371.0
    lat1, lon1 = math.radians(float(origin_lat)), math.radians(float(origin_lon))
    lat2, lon2 = math.radians(customer.latitude), math.radians(customer.longitude)
    a = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return radius * 2 * math.asin(math.sqrt(a))


def build_address(customer: CustomerDetails, catalog: Catalog) -> dict:
    """The Address block. Note SpecialIntructions -- the API's spelling."""
    latitude = customer.latitude
    longitude = customer.longitude
    if latitude is None or longitude is None:
        latitude, longitude = config.SPEEDOY_DEFAULT_LAT, config.SPEEDOY_DEFAULT_LON

    return {
        "Area": customer.address or None,
        "Latitude": latitude,
        "Longitude": longitude,
        "Location": None,
        "SpecialIntructions": customer.delivery_instructions or "-#-",
        "Label": customer.label or "Home",
        "Apartment": customer.apartment or "",
        "FullAddress": customer.address or None,
        "Province": None,
        "City": None,
        "ZipCode": None,
        "Id": None,
        "CreateTime": EPOCH,
        "UpdateTime": EPOCH,
        "Status": STATUS_ACTIVE,
    }


def build_payload(cart: Cart, customer: CustomerDetails, catalog: Catalog) -> dict:
    """The complete place-order-by-phone-bot body."""
    is_delivery = customer.is_delivery
    away = distance_km(customer, catalog)
    total = cart.total(catalog, is_delivery, away)
    scheduled = datetime.now(timezone.utc) + timedelta(
        minutes=catalog.estimated_delivery_minutes or 30
    )

    return {
        "CustomerId": config.SPEEDOY_CUSTOMER_ID,
        "RestaurantId": config.SPEEDOY_RESTAURANT_ID,
        "Address": build_address(customer, catalog),
        "PaymentMethod": PAYMENT_METHOD_CASH,
        "PaymentGatewayTransactionId": None,
        "PaymentAcceptResponse": None,
        "IsDeliveryOptionSelected": is_delivery,
        "OrderType": ORDER_TYPE_DELIVERY,
        "MenuItems": cart.to_payload(),
        "PromoCodeId": None,
        "IsEcoFriendly": False,
        "EcoFriendlyValue": 0,
        "ScheduleDateTime": scheduled.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-1] + "Z",
        "isContactLess": False,
        "platfrom": PLATFORM_PHONE_BOT,       # the API's spelling, not ours
        "firstName": customer.first_name,
        "lastName": customer.last_name,
        # A blank email is refused outright with "Profile not complete", so the
        # configured fallback stands in when the caller did not offer one.
        "email": customer.email or config.SPEEDOY_DEFAULT_EMAIL,
        "IsGuestUser": True,
        "countryCode": customer.country_code,
        "phoneNumber": customer.phone,
        "SpecialInstructions": customer.order_notes or "",
        "TotalAmount": total,
        "ForceProceed": False,
        "TotalAmountWithTax": cart.total_with_tax(catalog, is_delivery, away),
        "TakeAwayOption": TAKEAWAY_OPTION if not is_delivery else 0,
        "CarColor": "",
        "CarRegNo": "",
    }


async def place(cart: Cart, customer: CustomerDetails, catalog: Catalog) -> dict:
    """Send the order. Raises SpeedoyError with the API's own wording on refusal."""
    payload = build_payload(cart, customer, catalog)
    logger.info(
        "placing order: %d line(s), total %s, %s",
        len(cart.lines), payload["TotalAmount"], "delivery" if customer.is_delivery else "pickup",
    )
    result = await client.place_order_by_phone_bot(payload)
    return result if isinstance(result, dict) else {"result": result}


def extract_order_id(result: dict) -> str | None:
    """Dig the order id out of whatever shape came back.

    The collection shows no saved response for this endpoint, so rather than
    betting on one key we look through the plausible ones.
    """
    if not isinstance(result, dict):
        return None
    for key in ("OrderId", "Id", "orderId", "id", "OrderNumber", "OrderNo"):
        value = result.get(key)
        if value:
            return str(value)
    for nested_key in ("Order", "Data", "order"):
        nested = result.get(nested_key)
        if isinstance(nested, dict):
            found = extract_order_id(nested)
            if found:
                return found
    return None
