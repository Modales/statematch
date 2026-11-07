from __future__ import annotations

import gzip
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


def load_traces(path=ROOT / "data/processed/traces.jsonl.gz"):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def temporal_split(traces):
    dated = []
    for case in traces:
        end = max(timestamp(e["timestamp"]) for e in case["events"])
        dated.append((end, case))
    dated.sort(key=lambda item: item[0])
    n = len(dated)
    train_end = int(n * 0.75)
    valid_end = int(n * 0.875)
    return (
        [case for _, case in dated[:train_end]],
        [case for _, case in dated[train_end:valid_end]],
        [case for _, case in dated[valid_end:]],
    )


class Vocabulary:
    def __init__(self, activities, categories):
        self.activities = SPECIAL + sorted(activities)
        self.activity_to_id = {value: index for index, value in enumerate(self.activities)}
        self.categories = ["<PAD>"] + sorted(categories)
        self.category_to_id = {value: index for index, value in enumerate(self.categories)}

    def to_dict(self):
        return {"activities": self.activities, "categories": self.categories}

    @classmethod
    def from_dict(cls, value):
        obj = cls([], [])
        obj.activities = value["activities"]
        obj.activity_to_id = {v: i for i, v in enumerate(obj.activities)}
        obj.categories = value["categories"]
        obj.category_to_id = {v: i for i, v in enumerate(obj.categories)}
        return obj


def build_vocab(traces):
    activities = {event["activity"] for case in traces for event in case["events"]}
    categories = {case["category"] for case in traces}
    return Vocabulary(activities, categories)


def numeric_features(events, index):
    if index < 0:
        return [0.0, 0.0, 0.0]
    event = events[index]
    amount = float(event.get("amount", 0.0))
    scaled_amount = math.copysign(math.log1p(abs(amount)) / 12.0, amount)
    if index == 0:
        delta = 0.0
    else:
        seconds = max(0.0, (timestamp(event["timestamp"]) - timestamp(events[index - 1]["timestamp"])).total_seconds())
        delta = math.log1p(seconds) / 16.0
    position = index / max(1, len(events) - 1)
    return [scaled_amount, delta, position]


def encode_case(case, vocab, max_len=49):
    events = case["events"][: max_len - 1]
    ids = [vocab.activity_to_id.get(event["activity"], 3) for event in events]
    ids.append(vocab.activity_to_id["<EOS>"])
    numeric = [numeric_features(events, i) for i in range(len(events))]
    numeric.append([0.0, 0.0, 1.0])
    # BOS represents the state before the first observed event.
    input_ids = [vocab.activity_to_id["<BOS>"]] + ids[:-1]
    input_numeric = [[0.0, 0.0, 0.0]] + numeric[:-1]
    target_ids = ids
    category = vocab.category_to_id.get(case["category"], 0)
    return input_ids, input_numeric, target_ids, category


class TraceDataset(Dataset):
    def __init__(self, traces, vocab, max_len=49):
        self.traces = traces
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, index):
        return encode_case(self.traces[index], self.vocab, self.max_len)


def collate(batch, max_len=49):
    size = min(max_len, max(len(item[0]) for item in batch))
    activities = torch.zeros(len(batch), size, dtype=torch.long)
    targets = torch.zeros(len(batch), size, dtype=torch.long)
    numeric = torch.zeros(len(batch), size, 3, dtype=torch.float32)
    mask = torch.zeros(len(batch), size, dtype=torch.float32)
    categories = torch.zeros(len(batch), dtype=torch.long)
    for row, (ids, nums, target, category) in enumerate(batch):
        length = min(size, len(ids))
        activities[row, :length] = torch.tensor(ids[:length])
        targets[row, :length] = torch.tensor(target[:length])
        numeric[row, :length] = torch.tensor(nums[:length], dtype=torch.float32)
        mask[row, :length] = 1.0
        categories[row] = category
    return activities, categories, numeric, targets, mask
