import os
import tempfile
from pathlib import Path

os.environ["STATEMATCH_DB"] = str(Path(tempfile.gettempdir()) / "statematch-test.sqlite3")
if Path(os.environ["STATEMATCH_DB"]).exists():
    Path(os.environ["STATEMATCH_DB"]).unlink()

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_uses_real_non_llm_checkpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["attention"] is False
    assert body["llm"] is False


def test_case_queue_and_evidence_are_real():
    cases = client.get("/api/cases").json()["items"]
    assert len(cases) >= 20
    case_id = cases[0]["id"]
    detail = client.get(f"/api/cases/{case_id}")
    assert detail.status_code == 200
    assert detail.json()["model_nll"] > 0
    evidence = client.get(f"/api/cases/{case_id}/evidence").json()
    assert evidence["items"]
    assert "baseline_score" in evidence["items"][0]


def test_critical_case_cannot_be_approved_but_can_be_held():
    cases = client.get("/api/cases?status=exception").json()["items"]
    case_id = cases[0]["id"]
    denied = client.post(f"/api/cases/{case_id}/actions", json={"action": "approve_match"})
    assert denied.status_code == 409
    held = client.post(f"/api/cases/{case_id}/actions", json={"action": "hold_clearance", "note": "pytest verification"})
    assert held.status_code == 200
    assert held.json()["case"]["status"] == "held"
    audit = client.get(f"/api/cases/{case_id}/audit").json()
    assert audit["chain_valid"] is True
    assert audit["entries"][-1]["action"] == "hold_clearance"


def test_clean_case_action_and_audit_export():
    cases = client.get("/api/cases?status=ready").json()["items"]
    case_id = cases[0]["id"]
    response = client.post(f"/api/cases/{case_id}/actions", json={"action": "approve_match"})
    assert response.status_code == 200
    assert response.json()["case"]["status"] == "reconciled"
    exported = client.get(f"/api/cases/{case_id}/audit?format=csv")
    assert exported.status_code == 200
    assert "approve_match" in exported.text


def test_csv_ingestion_scores_and_persists():
    content = "case_id,activity,timestamp,amount,user\nUPLOAD-1,Create Purchase Order Item,2026-01-01T00:00:00Z,100,vendor_a\nUPLOAD-1,Record Goods Receipt,2026-01-02T00:00:00Z,100,user_a\nUPLOAD-1,Record Invoice Receipt,2026-01-03T00:00:00Z,100,user_b\nUPLOAD-1,Clear Invoice,2026-01-04T00:00:00Z,100,user_c\n"
    response = client.post("/api/ingest", files={"file": ("trace.csv", content, "text/csv")})
    assert response.status_code == 200, response.text
    assert response.json()["imported"] == ["UPLOAD-1"]
    case = client.get("/api/cases/UPLOAD-1").json()
    assert case["source"] == "CSV upload · trace.csv"
    assert len(case["events"]) == 4


def test_demo_injection_is_scored_not_mocked():
    response = client.post("/api/demo/inject", json={"scenario": "duplicate_invoice"})
    assert response.status_code == 200
    case = response.json()
    assert case["synthetic_deviation"] == "duplicate_invoice"
    assert case["model_nll"] > 0
    assert "POSSIBLE_DUPLICATE_INVOICE" in [finding["code"] for finding in case["rule_findings"]]
