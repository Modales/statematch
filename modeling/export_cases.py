#!/usr/bin/env python3
"""Score real holdout traces and export the application seed dataset."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))
from app.inference import ModelService  # noqa: E402
from app.rules import derive_status, evaluate_rules  # noqa: E402
from modeling.data import load_traces, temporal_split  # noqa: E402
from modeling.train import mutation  # noqa: E402

OUT = ROOT / "api/app/data"


def interesting(case):
    activities = {event["activity"] for event in case["events"]}
    return (
        "Record Goods Receipt" in activities
        and "Record Invoice Receipt" in activities
        and "Clear Invoice" in activities
        and 6 <= len(case["events"]) <= 24
    )


def main():
    random.seed(41)
    _, _, test = temporal_split(load_traces())
    candidates = [case for case in test if interesting(case)]
    random.shuffle(candidates)
    model = ModelService()
    selected = []
    clean = candidates[:9]
    selected.extend(clean)
    for kind in ["duplicate_invoice", "premature_unblock", "clear_before_gr"]:
        made = 0
        for case in candidates[9:]:
            changed = mutation(case, kind)
            if changed and interesting(changed):
                selected.append(changed)
                made += 1
            if made == 3:
                break
    # Add real, non-mutated traces for operational diversity.
    selected.extend(candidates[30:39])
    output = []
    for case in selected:
        scored = model.score(case)
        findings = evaluate_rules(case)
        case.update({key: value for key, value in scored.items() if key != "activity_surprise"})
        case["rule_findings"] = findings
        case["links"] = model.links(case)
        case["status"] = derive_status(case, findings, case["model_risk"])
        case["source"] = "BPI 2019 · temporal holdout"
        if case.get("synthetic_deviation"):
            case["source"] = f"BPI 2019 · injected {case['synthetic_deviation'].replace('_', ' ')}"
        case["resolution"] = None
        output.append(case)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases.json").write_text(json.dumps(output, indent=2))
    (OUT / "demo_template.json").write_text(json.dumps(clean[0], indent=2))
    metrics = json.loads((ROOT / "modeling/artifacts/metrics.json").read_text())
    metrics["model"] = {
        "name": "StateSpaceProcessModel",
        "checkpoint": "statematch_mamba.pt",
        "attention": False,
        "llm": False,
        "deployment": "Portable PyTorch CPU inference",
        "implementation": "2 selective SSM blocks, d_model=48, d_state=8, expand=2",
    }
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Exported {len(output)} scored cases")
    print({status: sum(1 for case in output if case["status"] == status) for status in sorted({case["status"] for case in output})})


if __name__ == "__main__":
    main()
