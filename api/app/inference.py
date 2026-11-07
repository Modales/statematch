from __future__ import annotations

import base64
import io
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modeling"))
from data import encode_case  # noqa: E402
from mamba import ModelConfig, StateSpaceProcessModel  # noqa: E402


class ModelService:
    def __init__(self, checkpoint_path=None):
        path = Path(checkpoint_path or ROOT / "modeling/artifacts/statematch_mamba.pt")
        from data import Vocabulary
        source = path
        encoded = path.with_suffix(path.suffix + ".base64")
        if not path.is_file() and encoded.is_file():
            source = io.BytesIO(base64.b64decode(encoded.read_text(encoding="ascii"), validate=True))
        payload = torch.load(source, map_location="cpu", weights_only=True)
        self.config = ModelConfig.from_dict(payload["config"])
        self.vocab = Vocabulary.from_dict(payload["vocab"])
        self.model = StateSpaceProcessModel(self.config)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.checkpoint = str(path)
        self.metadata = payload.get("metrics", {"validation_nll": payload.get("validation_nll")})

    def _tensors(self, case):
        ids, nums, targets, category = encode_case(case, self.vocab)
        return (
            torch.tensor([ids], dtype=torch.long),
            torch.tensor([category], dtype=torch.long),
            torch.tensor([nums], dtype=torch.float32),
            torch.tensor([targets], dtype=torch.long),
        )

    @staticmethod
    def _sequence_score(token_nll):
        if not token_nll:
            return 0.0
        values = np.asarray(token_nll, dtype=float)
        return float(values.mean() + 0.35 * np.quantile(values, 0.9))

    def score(self, case, include_hidden=False):
        activities, categories, numeric, targets = self._tensors(case)
        with torch.no_grad():
            result = self.model(activities, categories, numeric)
            logits = result["logits"]
            token_nll = F.cross_entropy(
                logits.reshape(-1, len(self.vocab.activities)), targets.reshape(-1), reduction="none"
            ).numpy()
            probabilities = F.softmax(logits, dim=-1)
        event_count = min(len(case["events"]), len(token_nll))
        nll = token_nll[:event_count].tolist()
        sequence_score = self._sequence_score(nll)
        risk = float(1 / (1 + math.exp(-(sequence_score - 1.25) / 0.55)))
        last = min(event_count, logits.shape[1]) - 1
        top_probability, top_id = probabilities[0, max(0, last)].max(dim=-1)
        event_rows = []
        for index, event in enumerate(case["events"]):
            surprise = float(nll[index]) if index < len(nll) else None
            event_rows.append({**event, "surprise": surprise})
        output = {
            "events": event_rows,
            "model_nll": float(np.mean(nll)) if nll else None,
            "sequence_score": sequence_score,
            "model_risk": risk,
            "predicted_next": self.vocab.activities[int(top_id)],
            "next_confidence": float(top_probability),
            "activity_surprise": nll,
        }
        if include_hidden:
            output["hidden"] = result["hidden"][0, :event_count].numpy()
        return output

    def counterfactuals(self, case, limit=6):
        baseline = self.score(case)["sequence_score"]
        rows = []
        for index, event in enumerate(case["events"]):
            reduced = {**case, "events": case["events"][:index] + case["events"][index + 1:]}
            if len(reduced["events"]) < 2:
                continue
            replay_score = self.score(reduced)["sequence_score"]
            rows.append({
                "event_id": event["id"],
                "activity": event["activity"],
                "delta": baseline - replay_score,
                "baseline_score": baseline,
                "replay_score": replay_score,
            })
        rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
        return rows[:limit]

    def links(self, case):
        scored = self.score(case, include_hidden=True)
        hidden = scored["hidden"]
        events = case["events"]
        receipts = [i for i, e in enumerate(events) if e["activity"] == "Record Goods Receipt"]
        invoices = [i for i, e in enumerate(events) if e["activity"] in ("Record Invoice Receipt", "Vendor creates invoice")]
        candidates = []
        for invoice in invoices:
            for receipt in receipts:
                if receipt >= len(hidden) or invoice >= len(hidden):
                    continue
                a, b = hidden[receipt], hidden[invoice]
                cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                distance = max(0, invoice - receipt)
                amount_delta = abs(float(events[receipt].get("amount", 0)) - float(events[invoice].get("amount", 0)))
                amount_penalty = min(1.0, amount_delta / (abs(float(events[invoice].get("amount", 0))) + 1.0))
                score = max(0.0, min(1.0, 0.55 * ((cosine + 1) / 2) + 0.3 * (1 - amount_penalty) + 0.15 * math.exp(-distance / 8)))
                candidates.append((score, receipt, invoice))
        candidates.sort(reverse=True)
        used_receipts, used_invoices, links = set(), set(), []
        for score, receipt, invoice in candidates:
            if receipt in used_receipts or invoice in used_invoices:
                continue
            used_receipts.add(receipt)
            used_invoices.add(invoice)
            links.append({
                "receipt_id": events[receipt]["id"],
                "invoice_id": events[invoice]["id"],
                "receipt_activity": events[receipt]["activity"],
                "invoice_activity": events[invoice]["activity"],
                "score": round(score, 4),
                "method": "Mamba hidden-state similarity + amount compatibility + temporal decay",
            })
        return links


@lru_cache(maxsize=1)
def get_model():
    return ModelService()
