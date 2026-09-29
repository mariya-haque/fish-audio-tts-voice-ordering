"""Read-only probe of the Speedoy API and the catalog built from it.

Fetches the restaurant and its options, prints what the bot will actually see,
and exercises the lookups the tool layer depends on. Places no order.

    python scripts/check_speedoy.py
    python scripts/check_speedoy.py --menu        # print the full model digest
    python scripts/check_speedoy.py --item calzone
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from speedoy import client, menu  # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  ' + detail if detail else ''}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu", action="store_true", help="print the digest sent to the model")
    parser.add_argument("--item", help="look one dish up and show its choices")
    args = parser.parse_args()

    missing = config.missing_order_keys()
    if missing:
        print(f"Missing credentials: {', '.join(missing)}. See .env.example.")
        return 1

    print(f"base url: {config.SPEEDOY_BASE_URL}")
    print(f"restaurant: {config.SPEEDOY_RESTAURANT_ID}\n")

    try:
        catalog = await menu.get_catalog(force=True)
    except client.SpeedoyError as exc:
        print(f"could not build the catalog: {exc}")
        return 1

    # --- the restaurant ------------------------------------------------------
    print(f"name: {catalog.name}")
    print(f"open status: {catalog.open_status} (2 = open)")
    print(f"minimum order: {catalog.minimum_order:g}   est. delivery: {catalog.estimated_delivery_minutes:g} min")
    print(f"delivery: fixed={catalog.delivery_charge.get('IsFixedFee')} "
          f"free from={catalog.delivery_charge.get('FreeDeliveryFromAmount')} "
          f"tax={catalog.delivery_charge.get('DeliveryTax')}")
    print()

    # --- shape ---------------------------------------------------------------
    items = catalog.all_items()
    plain = [i for i in items if not i.needs_choice]
    templated = [i for i in items if i.template_id]
    with_options = [i for i in items if i.option_ids]

    print(f"categories:        {len(catalog.categories)}")
    print(f"orderable items:   {len(items)}")
    print(f"  plain:           {len(plain)}")
    print(f"  needs a variant: {len(templated)}")
    print(f"  loose options:   {len(with_options)}")
    print(f"option groups:     {len(catalog.option_groups)}")
    print(f"templates:         {len(catalog.templates)}")
    print()

    check("categories loaded", len(catalog.categories) >= 15)
    check("items loaded", len(items) >= 100)
    check("templates loaded", len(catalog.templates) >= 5)
    check("every item has an id and a name", all(i.id and i.name for i in items))
    check("every templated item resolves its template",
          all(catalog.template_for(i) is not None for i in templated))
    check("every template has a priced primary choice",
          all(t.primary.items for t in catalog.templates.values()))

    # --- lookups the tools rely on -------------------------------------------
    print()
    for query in ("coca cola 33cl", "calzone", "kebab", "definitely not on the menu"):
        hits = catalog.find_items(query)
        names = ", ".join(i.name for i in hits[:4]) or "-"
        print(f"  find({query!r:28}) -> {len(hits):3} {names[:60]}")

    check("exact name gives exactly one hit", len(catalog.find_items("Coca cola 33cl")) == 1)
    check("nonsense gives none", len(catalog.find_items("definitely not on the menu")) == 0)
    check("diacritic-free query matches", len(catalog.find_items("Kebabratter")) >= 0)

    # --- one dish in full ----------------------------------------------------
    target = args.item or "calzone"
    hits = catalog.find_items(target)
    if hits:
        item = hits[0]
        print(f"\n{item.name}  {item.price:g} {catalog.currency}   ({item.category})")
        template = catalog.template_for(item)
        if template:
            print(f"  required -- {template.primary.name}:")
            for option in template.primary.items[:6]:
                print(f"      {option.name:24} +{option.price:g}")
            first = template.primary.items[0]
            groups = template.secondary_groups(first.id)
            print(f"  extras when '{first.name}' is chosen: {len(groups)} group(s)")
            for group in groups[:3]:
                sample = ", ".join(f"{o.name} +{o.price:g}" for o in group.items[:4])
                print(f"      {group.name:24} {len(group.items):3} items   {sample[:58]}")
        for group in catalog.plain_groups_for(item):
            sample = ", ".join(f"{o.name} +{o.price:g}" for o in group.items[:4])
            print(f"  {group.name:26} {len(group.items):3} items   {sample[:58]}")

    # --- the digest ----------------------------------------------------------
    digest = catalog.compact_text()
    print(f"\nmodel digest: {len(digest)} chars, {len(digest.splitlines())} lines")
    check("digest is a sane size for a prompt", 1000 < len(digest) < 40000, f"{len(digest)} chars")
    check("digest leaks no ids", not any(len(w) == 24 and w.isalnum() and not w.isalpha()
                                         for w in digest.split()))

    keywords = catalog.transcription_keywords()
    print(f"transcription keywords: {len(keywords)}  e.g. {', '.join(keywords[:6])}")

    if args.menu:
        print("\n" + "=" * 70)
        print(digest)

    await client.aclose()
    print()
    print("all checks passed" if not failures else f"{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
