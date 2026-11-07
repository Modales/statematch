#!/usr/bin/env python3
"""Train Mamba and an equal-width Transformer on a temporal BPI holdout."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import ROOT, TraceDataset, Vocabulary, build_vocab, collate, load_traces, temporal_split
from mamba import ModelConfig, StateSpaceProcessModel, TinyTransformer, parameter_count

ARTIFACTS = ROOT / "modeling/artifacts"


def choose_device():
    import os
    requested = os.environ.get("STATEMATCH_DEVICE", "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_tokens = 0.0
    actual_all, predicted_all = [], []
    for activities, categories, numeric, targets, mask in loader:
        activities, categories, numeric = activities.to(device), categories.to(device), numeric.to(device)
        targets, mask = targets.to(device), mask.to(device)
        with torch.set_grad_enabled(training):
            loss, result = model.loss(activities, categories, numeric, targets, mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        tokens = mask.sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        pred = result["logits"].argmax(-1)
        selected = mask.bool()
        actual_all.extend(targets[selected].cpu().numpy().tolist())
        predicted_all.extend(pred[selected].cpu().numpy().tolist())
    return {
        "nll": total_loss / max(1, total_tokens),
        "accuracy": accuracy_score(actual_all, predicted_all),
        "macro_f1": f1_score(actual_all, predicted_all, average="macro", zero_division=0),
    }


def markov_baseline(train, test, vocab):
    counts = Counter()
    transitions = {}
    for case in train:
        previous = 1
        for event in case["events"]:
            current = vocab.activity_to_id.get(event["activity"], 3)
            transitions.setdefault(previous, Counter())[current] += 1
            counts[current] += 1
            previous = current
        transitions.setdefault(previous, Counter())[2] += 1
    actual, predicted, losses = [], [], []
    for case in test:
        previous = 1
        # Score the same EOS target as both neural models.
        for event in [*case["events"], {"activity": "<EOS>"}]:
            current = vocab.activity_to_id.get(event["activity"], 3)
            distribution = transitions.get(previous, counts)
            predicted.append(distribution.most_common(1)[0][0])
            probability = (distribution[current] + 0.1) / (sum(distribution.values()) + 0.1 * len(vocab.activities))
            losses.append(-math.log(probability))
            actual.append(current)
            previous = current
    return {
        "nll": float(np.mean(losses)),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "parameters": 0,
    }


def mutation(case, kind):
    import copy
    from datetime import timedelta
    from data import timestamp
    mutated = copy.deepcopy(case)
    events = mutated["events"]
    if kind == "duplicate_invoice":
        candidates = [i for i, e in enumerate(events) if e["activity"] == "Record Invoice Receipt"]
        if not candidates:
            return None
        index = candidates[0]
        duplicate = copy.deepcopy(events[index])
        duplicate["id"] = mutated["id"] + ":injected-duplicate"
        duplicate["timestamp"] = (timestamp(duplicate["timestamp"]) + timedelta(hours=1)).isoformat()
        events.insert(index + 1, duplicate)
    elif kind == "premature_unblock":
        first_gr = next((i for i, e in enumerate(events) if e["activity"] == "Record Goods Receipt"), len(events) - 1)
        if first_gr == 0:
            return None
        template = copy.deepcopy(events[max(0, first_gr - 1)])
        template.update({"id": mutated["id"] + ":injected-unblock", "activity": "Remove Payment Block"})
        events.insert(first_gr, template)
    elif kind == "clear_before_gr":
        clear = next((i for i, e in enumerate(events) if e["activity"] == "Clear Invoice"), None)
        gr = next((i for i, e in enumerate(events) if e["activity"] == "Record Goods Receipt"), None)
        if clear is None or gr is None or clear < gr:
            return None
        event = events.pop(clear)
        event["id"] = mutated["id"] + ":injected-clear"
        events.insert(gr, event)
    else:
        raise ValueError(kind)
    mutated["id"] = case["id"] + "~" + kind
    mutated["synthetic_deviation"] = kind
    return mutated


def sequence_nll(model, case, vocab, device, max_len=49):
    from data import encode_case
    ids, nums, targets, category = encode_case(case, vocab, max_len)
    activities = torch.tensor([ids], device=device)
    categories = torch.tensor([category], device=device)
    numeric = torch.tensor([nums], dtype=torch.float32, device=device)
    target = torch.tensor([targets], device=device)
    mask = torch.ones_like(target, dtype=torch.float32)
    with torch.no_grad():
        _, result = model.loss(activities, categories, numeric, target, mask)
        token_loss = F.cross_entropy(result["logits"].reshape(-1, len(vocab.activities)), target.reshape(-1), reduction="none")
    values = token_loss.cpu().numpy()
    return float(np.mean(values) + 0.35 * np.quantile(values, 0.9))


def evaluate_anomaly(model, clean, vocab, device):
    kinds = ["duplicate_invoice", "premature_unblock", "clear_before_gr"]
    pairs = []
    for case in clean:
        for kind in kinds:
            changed = mutation(case, kind)
            if changed is not None:
                pairs.append((case, changed))
        if len(pairs) >= 180:
            break
    labels, scores = [], []
    for original, changed in pairs:
        labels.extend([0, 1])
        scores.extend([sequence_nll(model, original, vocab, device), sequence_nll(model, changed, vocab, device)])
    return {
        "paired_cases": len(pairs),
        "synthetic_deviation_auroc": roc_auc_score(labels, scores),
        "note": "Measures separation of clean traces from explicitly injected process deviations; not fraud detection.",
    }


def latency_benchmark(model, vocab, device, category):
    lengths = [8, 16, 32, 48]
    output = []
    model.eval()
    for length in lengths:
        activities = torch.randint(4, len(vocab.activities), (1, length), device=device)
        categories = torch.tensor([category], device=device)
        numeric = torch.randn(1, length, 3, device=device)
        for _ in range(2):
            with torch.no_grad():
                model(activities, categories, numeric)
        start = time.perf_counter()
        repeats = 12
        for _ in range(repeats):
            with torch.no_grad():
                model(activities, categories, numeric)
        if device.type == "mps":
            torch.mps.synchronize()
        output.append({"length": length, "ms": (time.perf_counter() - start) * 1000 / repeats})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train", type=int, default=9000)
    parser.add_argument("--max-eval", type=int, default=1200)
    args = parser.parse_args()
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    traces = load_traces()
    train, valid, test = temporal_split(traces)
    random.Random(19).shuffle(train)
    train = train[:args.max_train]
    valid, test = valid[:args.max_eval], test[:args.max_eval]
    vocab = build_vocab(train)
    config = ModelConfig(activity_vocab=len(vocab.activities), category_vocab=len(vocab.categories))
    device = choose_device()
    print(f"Device: {device}; train={len(train)}, valid={len(valid)}, test={len(test)}")
    train_loader = DataLoader(TraceDataset(train, vocab), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(TraceDataset(valid, vocab), batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(TraceDataset(test, vocab), batch_size=args.batch_size, collate_fn=collate)
    metrics = {
        "dataset": {"train": len(train), "validation": len(valid), "test": len(test), "split": "temporal 75/12.5/12.5 before caps"},
        "config": config.__dict__,
        "vocab_size": len(vocab.activities),
        "device": str(device),
        "models": {},
    }
    for name, cls in [("mamba", StateSpaceProcessModel), ("transformer", TinyTransformer)]:
        model = cls(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3 if name == "mamba" else 1e-3, weight_decay=1e-4)
        best_state, best_valid, history = None, float("inf"), []
        started = time.perf_counter()
        for epoch in range(args.epochs):
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            valid_metrics = run_epoch(model, valid_loader, device)
            history.append({"epoch": epoch + 1, "train_nll": train_metrics["nll"], "validation_nll": valid_metrics["nll"]})
            print(name, history[-1], flush=True)
            if valid_metrics["nll"] < best_valid:
                best_valid = valid_metrics["nll"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if name == "mamba":
                    torch.save({"model": best_state, "config": config.__dict__, "vocab": vocab.to_dict(), "validation_nll": best_valid}, ARTIFACTS / "statematch_mamba.pt")
        model.load_state_dict(best_state)
        test_metrics = run_epoch(model, test_loader, device)
        result = {
            **test_metrics,
            "parameters": parameter_count(model),
            "training_seconds": time.perf_counter() - started,
            "history": history,
        }
        if name == "mamba":
            result.update(evaluate_anomaly(model, test, vocab, device))
            result["latency"] = latency_benchmark(model, vocab, device, 1)
            torch.save({"model": best_state, "config": config.__dict__, "vocab": vocab.to_dict(), "metrics": result}, ARTIFACTS / "statematch_mamba.pt")
        metrics["models"][name] = result
    metrics["models"]["markov"] = markov_baseline(train, test, vocab)
    metrics["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "vocab.json").write_text(json.dumps(vocab.to_dict(), indent=2))
    print(json.dumps({name: {k: v for k, v in result.items() if k in ("nll", "accuracy", "macro_f1", "parameters", "synthetic_deviation_auroc")} for name, result in metrics["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
