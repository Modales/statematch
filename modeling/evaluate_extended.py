#!/usr/bin/env python3
"""Evaluate the published checkpoint on the complete chronological BPI holdout.

The original 800-case test metrics are replicated, then the 1,863 later cases
are evaluated without retraining or threshold tuning. Deviations use at most
one deterministic mutation per source case so bootstrap resamples are independent
at the case level.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from app.inference import ModelService  # noqa: E402
from modeling.data import TraceDataset, collate, load_traces, temporal_split  # noqa: E402
from modeling.train import mutation  # noqa: E402

DEVIATION_KINDS = ["duplicate_invoice", "premature_unblock", "clear_before_gr"]


def sequence_score(losses):
    values = np.asarray(losses, dtype=float)
    return float(values.mean() + 0.35 * np.quantile(values, 0.9))


def infer_cases(cases, model, vocab, batch_size=64):
    """Return token-level predictions and losses grouped by source case."""
    loader = DataLoader(
        TraceDataset(cases, vocab), batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    records = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for activities, categories, numeric, targets, mask in loader:
            result = model(activities, categories, numeric)
            losses = F.cross_entropy(
                result["logits"].reshape(-1, len(vocab.activities)),
                targets.reshape(-1), reduction="none",
            ).reshape(targets.shape)
            predictions = result["logits"].argmax(-1)
            for row in range(len(activities)):
                case = cases[offset + row]
                length = int(mask[row].sum().item())
                actual = targets[row, :length].cpu().numpy()
                predicted = predictions[row, :length].cpu().numpy()
                token_loss = losses[row, :length].cpu().numpy()
                records.append({
                    "id": case["id"],
                    "category": case["category"],
                    "event_count": len(case["events"]),
                    "first_event": min(event["timestamp"] for event in case["events"]),
                    "last_event": max(event["timestamp"] for event in case["events"]),
                    "targets": actual,
                    "predictions": predicted,
                    "losses": token_loss,
                    "loss_sum": float(token_loss.sum()),
                    "correct": int((actual == predicted).sum()),
                    "unknown_events": int((actual == 3).sum()),
                    "sequence_score": sequence_score(token_loss),
                    "synthetic_deviation": case.get("synthetic_deviation"),
                })
            offset += len(activities)
    return records


def summarize(records):
    if not records:
        raise ValueError("Cannot summarize an empty evaluation slice")
    actual = np.concatenate([record["targets"] for record in records])
    predicted = np.concatenate([record["predictions"] for record in records])
    tokens = int(sum(len(record["targets"]) for record in records))
    unknown_events = int(sum(record["unknown_events"] for record in records))
    event_targets = tokens - len(records)  # One EOS target per case.
    return {
        "cases": len(records),
        "tokens": tokens,
        "nll": float(sum(record["loss_sum"] for record in records) / tokens),
        "accuracy": float(accuracy_score(actual, predicted)),
        # Matches the original experiment: classes absent from both actual and
        # predicted values do not contribute to this bounded-slice macro F1.
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "unknown_events": unknown_events,
        "unknown_event_rate": float(unknown_events / max(1, event_targets)),
        "cases_with_unknown_activity": int(sum(record["unknown_events"] > 0 for record in records)),
        "first_event": min(record["first_event"] for record in records),
        "last_event": max(record["last_event"] for record in records),
    }


def percentile_interval(values):
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def case_bootstrap(records, repetitions=300, seed=1907):
    """Percentile CIs that preserve within-case token correlation."""
    if len(records) < 2 or repetitions < 2:
        raise ValueError("Bootstrap needs at least two cases and two repetitions")
    rng = np.random.default_rng(seed)
    nll_values, accuracy_values, f1_values = [], [], []
    for _ in range(repetitions):
        indices = rng.integers(0, len(records), len(records))
        sample = [records[int(index)] for index in indices]
        actual = np.concatenate([record["targets"] for record in sample])
        predicted = np.concatenate([record["predictions"] for record in sample])
        tokens = sum(len(record["targets"]) for record in sample)
        nll_values.append(sum(record["loss_sum"] for record in sample) / tokens)
        accuracy_values.append(accuracy_score(actual, predicted))
        f1_values.append(f1_score(actual, predicted, average="macro", zero_division=0))
    return {
        "nll_ci95": percentile_interval(nll_values),
        "accuracy_ci95": percentile_interval(accuracy_values),
        "macro_f1_ci95": percentile_interval(f1_values),
    }


def deterministic_deviations(cases):
    """Create no more than one paired deviation for each original case."""
    pairs = []
    for case in cases:
        digest = hashlib.sha1(case["id"].encode()).digest()
        start = int.from_bytes(digest[:2], "big") % len(DEVIATION_KINDS)
        for offset in range(len(DEVIATION_KINDS)):
            kind = DEVIATION_KINDS[(start + offset) % len(DEVIATION_KINDS)]
            changed = mutation(case, kind)
            if changed is not None:
                pairs.append((case, changed, kind))
                break
    return pairs


def deviation_summary(clean_scores, changed_scores):
    labels = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(changed_scores))])
    scores = np.concatenate([clean_scores, changed_scores])
    return {
        "pairs": len(clean_scores),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "score_increase_rate": float(np.mean(changed_scores > clean_scores)),
        "median_score_delta": float(np.median(changed_scores - clean_scores)),
    }


def evaluate_deviations(pairs, model, vocab, batch_size, bootstrap_repetitions, seed):
    flattened = [case for pair in pairs for case in pair[:2]]
    records = infer_cases(flattened, model, vocab, batch_size)
    clean_records = records[0::2]
    changed_records = records[1::2]
    clean_scores = np.array([record["sequence_score"] for record in clean_records])
    changed_scores = np.array([record["sequence_score"] for record in changed_records])
    result = deviation_summary(clean_scores, changed_scores)
    rng = np.random.default_rng(seed)
    auroc_values, increase_values = [], []
    for _ in range(bootstrap_repetitions):
        indices = rng.integers(0, len(pairs), len(pairs))
        sampled_clean = clean_scores[indices]
        sampled_changed = changed_scores[indices]
        labels = np.concatenate([np.zeros(len(indices)), np.ones(len(indices))])
        scores = np.concatenate([sampled_clean, sampled_changed])
        auroc_values.append(roc_auc_score(labels, scores))
        increase_values.append(np.mean(sampled_changed > sampled_clean))
    result["auroc_ci95"] = percentile_interval(auroc_values)
    result["score_increase_rate_ci95"] = percentile_interval(increase_values)
    result["by_kind"] = {}
    for kind in DEVIATION_KINDS:
        indices = [index for index, pair in enumerate(pairs) if pair[2] == kind]
        result["by_kind"][kind] = deviation_summary(clean_scores[indices], changed_scores[indices])
    return result


def grouped_metrics(records, key_fn, names):
    output = []
    for name in names:
        selected = [record for record in records if key_fn(record) == name]
        if selected:
            output.append({"name": name, **summarize(selected)})
    return output


def length_bucket(record):
    length = record["event_count"]
    if length <= 4:
        return "2–4 events"
    if length <= 8:
        return "5–8 events"
    if length <= 16:
        return "9–16 events"
    return "17+ events"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.bootstrap < 2:
        raise SystemExit("--bootstrap must be at least 2")

    started = time.perf_counter()
    traces_path = ROOT / "data/processed/traces.jsonl.gz"
    if not traces_path.is_file():
        raise SystemExit("Run the BPI preparation pipeline first: data/processed/traces.jsonl.gz is missing")
    _, _, test = temporal_split(load_traces(traces_path))
    if len(test) <= 800:
        raise SystemExit("The extended evaluation requires more than 800 chronological test cases")
    later = test[800:]

    encoded = (ROOT / "modeling/artifacts/statematch_mamba.pt.base64").read_bytes()
    decoded = base64.b64decode(encoded, validate=True)
    checkpoint_sha256 = hashlib.sha256(decoded).hexdigest()
    (ROOT / "verification").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extended-", dir=ROOT / "verification") as temporary:
        checkpoint = Path(temporary) / "statematch_mamba.pt"
        checkpoint.write_bytes(decoded)
        service = ModelService(checkpoint)
        print(f"Scoring {len(test)} temporal test cases, including {len(later)} later cases", flush=True)
        records = infer_cases(test, service.model, service.vocab, args.batch_size)
        anchor_records, later_records = records[:800], records[800:]

        anchor = summarize(anchor_records)
        full_test = summarize(records)
        extended = summarize(later_records)
        extended.update(case_bootstrap(later_records, args.bootstrap))

        windows = []
        for index, indices in enumerate(np.array_split(np.arange(len(later_records)), 3), start=1):
            selected = [later_records[int(position)] for position in indices]
            windows.append({"name": f"Later window {index}", **summarize(selected)})

        pairs = deterministic_deviations(later)
        print(f"Scoring {len(pairs)} independent clean/deviation pairs", flush=True)
        deviations = evaluate_deviations(
            pairs, service.model, service.vocab, args.batch_size, args.bootstrap, seed=1919
        )

    baseline_metrics = json.loads((ROOT / "modeling/artifacts/metrics.json").read_text())
    original_nll = baseline_metrics["models"]["mamba"]["nll"]
    anchor["original_reported_nll"] = original_nll
    anchor["nll_absolute_delta"] = abs(anchor["nll"] - original_nll)
    if anchor["nll_absolute_delta"] > 1e-6:
        raise RuntimeError(f"Original test NLL did not replicate: {anchor['nll']} versus {original_nll}")

    result = {
        "protocol": "Frozen checkpoint; chronological holdout; no retraining, threshold tuning, or test-case selection",
        "source_dataset": {
            "name": "BPI Challenge 2019",
            "doi": "10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1",
            "sample_cases": 21302,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "bootstrap": {
            "method": "Case-level percentile bootstrap; whole traces resampled with replacement",
            "repetitions": args.bootstrap,
            "seed": 1907,
        },
        "anchor_800": anchor,
        "full_temporal_test": full_test,
        "later_period": extended,
        "later_windows": windows,
        "window_method": "Three equal-count contiguous partitions of the later-period cases, ordered by case end timestamp",
        "length_buckets": grouped_metrics(
            later_records, length_bucket, ["2–4 events", "5–8 events", "9–16 events", "17+ events"]
        ),
        "categories": grouped_metrics(
            later_records,
            lambda record: record["category"],
            sorted({record["category"] for record in later_records}),
        ),
        "deviations": {
            **deviations,
            "assignment": "SHA1(case_id) selects the first applicable deviation; at most one pair per source case",
            "score": "Mean token NLL + 0.35 × 90th-percentile token NLL, including EOS as in the original experiment",
            "note": "Explicitly injected process deviations on real BPI traces; not observed fraud labels or evidence of production fraud detection",
        },
        "academic_basis": [
            {
                "citation": "Teinemaa et al. (2019), Outcome-oriented predictive process monitoring: Review and benchmark",
                "url": "https://dl.acm.org/doi/abs/10.1145/3301300",
                "use": "Chronological evaluation and explicit benchmarking protocol",
            },
            {
                "citation": "Ko and Comuzzi (2023), A Systematic Review of Anomaly Detection for Business Process Event Logs",
                "url": "https://link.springer.com/article/10.1007/s12599-023-00794-y",
                "use": "Bounded interpretation of event-log anomaly evaluation",
            },
            {
                "citation": "Nolle et al. (2019), BINet: Multivariate business process anomaly detection using deep learning",
                "url": "https://arxiv.org/abs/1902.03155",
                "use": "Case-level anomaly scoring and deviation-specific reporting",
            },
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": time.perf_counter() - started,
    }
    for path in [ROOT / "modeling/artifacts/metrics.json", ROOT / "api/app/data/model_metrics.json"]:
        payload = json.loads(path.read_text())
        payload["extended_evaluation"] = result
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "anchor_nll_delta": anchor["nll_absolute_delta"],
        "full_temporal_test": {key: full_test[key] for key in ("cases", "nll", "accuracy", "macro_f1")},
        "later_period": {key: extended[key] for key in ("cases", "nll", "nll_ci95", "accuracy", "macro_f1")},
        "deviations": {key: deviations[key] for key in ("pairs", "auroc", "auroc_ci95", "score_increase_rate")},
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
