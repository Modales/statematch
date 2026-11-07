from __future__ import annotations

from collections import defaultdict
from datetime import datetime


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def evaluate_rules(case):
    events = case["events"]
    findings = []
    requires_gr = "3-way" in case.get("category", "")
    gr_indices = [i for i, e in enumerate(events) if e["activity"] == "Record Goods Receipt"]
    clear_indices = [i for i, e in enumerate(events) if e["activity"] == "Clear Invoice"]
    if requires_gr and clear_indices and not gr_indices:
        findings.append({
            "code": "MISSING_GOODS_RECEIPT", "severity": "critical",
            "title": "Clearance without a goods receipt",
            "detail": "This item category requires goods receipt before invoice clearance.",
            "event_ids": [events[i]["id"] for i in clear_indices],
        })
    elif requires_gr and gr_indices and clear_indices and min(clear_indices) < min(gr_indices):
        findings.append({
            "code": "CLEAR_BEFORE_GR", "severity": "critical",
            "title": "Invoice cleared before goods receipt",
            "detail": "The event order violates the documented three-way-match control.",
            "event_ids": [events[min(clear_indices)]["id"], events[min(gr_indices)]["id"]],
        })
    if gr_indices:
        first_gr = min(gr_indices)
        premature = [i for i, e in enumerate(events) if e["activity"] == "Remove Payment Block" and i < first_gr]
        if premature:
            findings.append({
                "code": "PREMATURE_UNBLOCK", "severity": "critical",
                "title": "Payment block removed before receipt",
                "detail": "A payment block was removed before the first recorded goods receipt.",
                "event_ids": [events[i]["id"] for i in premature] + [events[first_gr]["id"]],
            })
    invoice_groups = defaultdict(list)
    for index, event in enumerate(events):
        if event["activity"] == "Record Invoice Receipt":
            key = round(float(event.get("amount", 0.0)), 2)
            invoice_groups[key].append((index, event))
    duplicate_ids = set()
    duplicate_amounts = set()
    shortest_gap = None
    for amount, group in invoice_groups.items():
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                i, a = group[left]
                j, b = group[right]
                hours = abs((parse_time(b["timestamp"]) - parse_time(a["timestamp"])).total_seconds()) / 3600
                if hours <= 2:
                    duplicate_ids.update([a["id"], b["id"]])
                    duplicate_amounts.add(amount)
                    shortest_gap = hours if shortest_gap is None else min(shortest_gap, hours)
    if duplicate_ids:
        findings.append({
            "code": "POSSIBLE_DUPLICATE_INVOICE", "severity": "critical",
            "title": "Repeated invoices within two hours",
            "detail": f"{len(duplicate_ids)} invoice events share matching values ({', '.join(f'{amount:,.2f}' for amount in sorted(duplicate_amounts))} EUR). Shortest interval: {shortest_gap:.2f} hours. Review before clearance.",
            "event_ids": sorted(duplicate_ids),
        })
    initial_amount = float(events[0].get("amount", 0.0)) if events else 0.0
    mismatched = [e["id"] for e in events if e["activity"] == "Record Invoice Receipt" and initial_amount and abs(float(e.get("amount", 0)) - initial_amount) > max(1.0, abs(initial_amount) * 0.01)]
    if mismatched:
        findings.append({
            "code": "VALUE_VARIANCE", "severity": "warning",
            "title": "Invoice value differs from opening value",
            "detail": "The value can be legitimate for partial invoices; verify allocation before automatic reconciliation.",
            "event_ids": mismatched,
        })
    return findings


def derive_status(case, findings, model_risk):
    if case.get("status") in ("reconciled", "held", "in_review", "awaiting_receipt"):
        return case["status"]
    if any(item["severity"] == "critical" for item in findings):
        return "exception"
    if model_risk >= 0.62 or findings:
        return "review"
    if any(e["activity"] == "Clear Invoice" for e in case["events"]):
        return "ready"
    return "open"
