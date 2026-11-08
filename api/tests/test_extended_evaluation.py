import copy

import numpy as np

from modeling.evaluate_extended import (
    case_bootstrap,
    deterministic_deviations,
    deviation_summary,
    summarize,
)


def record(identifier, losses, targets, predictions):
    actual = np.asarray(targets)
    predicted = np.asarray(predictions)
    token_loss = np.asarray(losses, dtype=float)
    return {
        "id": identifier,
        "category": "test",
        "event_count": len(actual) - 1,
        "first_event": "2018-01-01T00:00:00Z",
        "last_event": "2018-01-01T00:00:00Z",
        "targets": actual,
        "predictions": predicted,
        "losses": token_loss,
        "loss_sum": float(token_loss.sum()),
        "correct": int((actual == predicted).sum()),
        "unknown_events": int((actual == 3).sum()),
        "sequence_score": float(token_loss.mean()),
        "synthetic_deviation": None,
    }


def test_summarize_uses_token_weighting_and_case_coverage():
    records = [
        record("a", [0.0, 1.0], [4, 2], [4, 2]),
        record("b", [2.0], [3], [0]),
    ]
    result = summarize(records)
    assert result["cases"] == 2
    assert result["tokens"] == 3
    assert result["nll"] == 1.0
    assert result["accuracy"] == 2 / 3
    assert result["unknown_event_rate"] == 1.0
    assert result["cases_with_unknown_activity"] == 1


def test_case_bootstrap_is_deterministic_and_bounded():
    records = [
        record("a", [0.2, 0.3], [4, 2], [4, 2]),
        record("b", [0.8, 0.4], [4, 2], [4, 5]),
        record("c", [1.1, 0.2], [5, 2], [4, 2]),
        record("d", [0.4, 0.6], [4, 2], [4, 2]),
    ]
    first = case_bootstrap(records, repetitions=20, seed=42)
    second = case_bootstrap(records, repetitions=20, seed=42)
    assert first == second
    for values in first.values():
        assert 0 <= values[0] <= values[1]
    assert first["nll_ci95"][1] < 2


def test_deviation_assignment_has_one_pair_per_source_case():
    case = {
        "id": "source-1",
        "category": "3-way match, invoice before GR",
        "events": [
            {"id": "e1", "activity": "Create Purchase Order Item", "timestamp": "2018-01-01T00:00:00Z", "amount": 10},
            {"id": "e2", "activity": "Record Goods Receipt", "timestamp": "2018-01-02T00:00:00Z", "amount": 10},
            {"id": "e3", "activity": "Record Invoice Receipt", "timestamp": "2018-01-03T00:00:00Z", "amount": 10},
            {"id": "e4", "activity": "Clear Invoice", "timestamp": "2018-01-04T00:00:00Z", "amount": 10},
        ],
    }
    cases = [{**copy.deepcopy(case), "id": f"source-{index}"} for index in range(8)]
    original = copy.deepcopy(cases)
    pairs = deterministic_deviations(cases)
    assert len(pairs) == len(cases)
    assert len({pair[0]["id"] for pair in pairs}) == len(cases)
    assert all(pair[1]["synthetic_deviation"] == pair[2] for pair in pairs)
    assert cases == original


def test_deviation_summary_reports_paired_score_increases():
    result = deviation_summary(
        np.array([0.1, 0.2, 0.4]), np.array([0.3, 0.1, 0.5])
    )
    assert result["pairs"] == 3
    assert result["score_increase_rate"] == 2 / 3
    assert 0 <= result["auroc"] <= 1
    assert 0 <= result["average_precision"] <= 1
