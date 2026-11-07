#!/usr/bin/env python3
"""Stream the official BPI 2019 XES log into a deterministic, compact JSONL sample."""
from __future__ import annotations

import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/BPI_Challenge_2019.xes"
OUT = ROOT / "data/processed/traces.jsonl.gz"
STATS = ROOT / "data/processed/dataset_stats.json"


def attributes(element):
    result = {}
    for child in element:
        key = child.attrib.get("key")
        if not key:
            continue
        value = child.attrib.get("value")
        if child.tag.endswith("boolean"):
            value = value == "true"
        elif child.tag.endswith(("float", "int")):
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
        result[key] = value
    return result


def selected(case_id: str) -> bool:
    digest = hashlib.sha1(case_id.encode()).digest()[0]
    return digest % 12 == 0


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    activities = Counter()
    categories = Counter()
    total = kept = 0
    with gzip.open(OUT, "wt", encoding="utf-8") as target:
        for _, trace in ET.iterparse(SOURCE, events=("end",)):
            if not trace.tag.endswith("trace"):
                continue
            total += 1
            attrs = attributes(trace)
            case_id = str(attrs.get("concept:name", ""))
            if case_id and selected(case_id):
                events = []
                for position, event in enumerate(trace):
                    if not event.tag.endswith("event"):
                        continue
                    item = attributes(event)
                    activity = str(item.get("concept:name", "Unknown"))
                    timestamp = str(item.get("time:timestamp", ""))
                    if not timestamp:
                        continue
                    amount = float(item.get("Cumulative net worth (EUR)", 0.0) or 0.0)
                    events.append({
                        "id": f"{case_id}:{position:03d}",
                        "activity": activity,
                        "timestamp": timestamp,
                        "amount": amount,
                        "user": str(item.get("org:resource", item.get("User", "NONE"))),
                    })
                    activities[activity] += 1
                if 2 <= len(events) <= 256:
                    case = {
                        "id": case_id,
                        "purchase_document": str(attrs.get("Purchasing Document", case_id.split("_")[0])),
                        "item": str(attrs.get("Item", "00001")),
                        "vendor": str(attrs.get("Vendor", "Unknown vendor")),
                        "vendor_name": str(attrs.get("Name", "Unknown vendor")),
                        "category": str(attrs.get("Item Category", "Unknown")),
                        "spend_area": str(attrs.get("Spend area text", "Unclassified")),
                        "spend_class": str(attrs.get("Spend classification text", "Unclassified")),
                        "item_type": str(attrs.get("Item Type", "Unknown")),
                        "amount": events[-1]["amount"],
                        "events": events,
                    }
                    target.write(json.dumps(case, separators=(",", ":")) + "\n")
                    categories[case["category"]] += 1
                    kept += 1
            trace.clear()
    stats = {
        "source": "BPI Challenge 2019 / 4TU.Centre for Research Data",
        "doi": "10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1",
        "source_cases": total,
        "sample_cases": kept,
        "sampling": "SHA1(case_id) first byte mod 12 == 0; deterministic case-level sample",
        "activities": dict(activities.most_common()),
        "categories": dict(categories.most_common()),
        "license_note": "Public research dataset; values and identifiers are anonymized. See dataset citation in README.",
    }
    STATS.write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k not in ("activities", "categories")}, indent=2))
    print(f"Activities: {len(activities)}; categories: {dict(categories)}")


if __name__ == "__main__":
    main()
