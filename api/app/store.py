from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path=None):
        configured = path or os.environ.get("STATEMATCH_DB", str(ROOT / "statematch.sqlite3"))
        Path(configured).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(configured, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    vendor TEXT,
                    amount REAL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audits (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    hash TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS audits_case ON audits(case_id, sequence);
            """)
            count = self.connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
            if count == 0:
                self._seed()
            self.connection.commit()

    def _seed(self):
        cases = json.loads((DATA / "cases.json").read_text())
        for case in cases:
            self._insert_case(case, commit=False)
            self._append_audit(case["id"], "case.scored", "statematch-engine", "Real checkpoint inference and deterministic controls", {
                "model_nll": case["model_nll"], "model_risk": case["model_risk"],
                "rule_codes": [finding["code"] for finding in case["rule_findings"]],
                "source": case["source"],
            }, commit=False)

    def _insert_case(self, case, commit=True):
        now = utcnow()
        case["updated_at"] = now
        self.connection.execute(
            "INSERT OR REPLACE INTO cases(id,status,vendor,amount,updated_at,payload) VALUES(?,?,?,?,?,?)",
            (case["id"], case["status"], case.get("vendor"), case.get("amount", 0), now, json.dumps(case)),
        )
        if commit:
            self.connection.commit()

    def _append_audit(self, case_id, action, actor, note, payload, commit=True):
        previous = self.connection.execute(
            "SELECT hash FROM audits WHERE case_id=? ORDER BY sequence DESC LIMIT 1", (case_id,)
        ).fetchone()
        previous_hash = previous[0] if previous else "GENESIS"
        created_at = utcnow()
        body = {
            "id": str(uuid.uuid4()), "case_id": case_id, "action": action,
            "actor": actor, "note": note, "created_at": created_at,
            "payload": payload, "previous_hash": previous_hash,
        }
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.connection.execute(
            "INSERT INTO audits(id,case_id,action,actor,note,created_at,payload,previous_hash,hash) VALUES(?,?,?,?,?,?,?,?,?)",
            (body["id"], case_id, action, actor, note, created_at, json.dumps(payload), previous_hash, digest),
        )
        if commit:
            self.connection.commit()
        return {**body, "hash": digest}

    def list_cases(self, status=None, search=None):
        with self.lock:
            rows = self.connection.execute("SELECT payload FROM cases ORDER BY updated_at DESC").fetchall()
        cases = [json.loads(row[0]) for row in rows]
        if status and status != "all":
            cases = [case for case in cases if case["status"] == status]
        if search:
            needle = search.lower()
            cases = [case for case in cases if needle in json.dumps(case).lower()]
        return cases

    def get_case(self, case_id):
        with self.lock:
            row = self.connection.execute("SELECT payload FROM cases WHERE id=?", (case_id,)).fetchone()
        if row is None:
            raise KeyError(case_id)
        return json.loads(row[0])

    def save_case(self, case):
        with self.lock:
            self._insert_case(case)

    def apply_action(self, case_id, action, actor, note=None):
        transitions = {
            "approve_match": ("reconciled", "Match approved and ledger state updated"),
            "hold_clearance": ("held", "Clearance held pending exception review"),
            "assign_review": ("in_review", "Case assigned to an operations analyst"),
            "request_receipt": ("awaiting_receipt", "Goods receipt confirmation requested"),
            "reopen": ("open", "Case reopened for further processing"),
        }
        if action not in transitions:
            raise ValueError(f"Unsupported action: {action}")
        with self.lock:
            case = self.get_case(case_id)
            if action == "approve_match" and any(f["severity"] == "critical" for f in case["rule_findings"]):
                raise PermissionError("A critical control finding prevents match approval")
            status, message = transitions[action]
            previous_status = case["status"]
            case["status"] = status
            case["resolution"] = {"action": action, "actor": actor, "note": note, "at": utcnow()}
            self._insert_case(case, commit=False)
            receipt = self._append_audit(case_id, action, actor, note or message, {
                "from": previous_status, "to": status,
                "rule_codes": [f["code"] for f in case["rule_findings"]],
            }, commit=False)
            self.connection.commit()
            return case, receipt

    def audit(self, case_id):
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM audits WHERE case_id=? ORDER BY sequence", (case_id,)
            ).fetchall()
        output = []
        previous_hash = "GENESIS"
        valid = True
        for row in rows:
            body = {
                "id": row["id"], "case_id": row["case_id"], "action": row["action"],
                "actor": row["actor"], "note": row["note"], "created_at": row["created_at"],
                "payload": json.loads(row["payload"]), "previous_hash": row["previous_hash"],
            }
            digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            entry_valid = digest == row["hash"] and previous_hash == row["previous_hash"]
            valid = valid and entry_valid
            output.append({**body, "hash": row["hash"], "valid": entry_valid})
            previous_hash = row["hash"]
        return {"case_id": case_id, "chain_valid": valid, "entries": output}

    def summary(self):
        cases = self.list_cases()
        statuses = {}
        for case in cases:
            statuses[case["status"]] = statuses.get(case["status"], 0) + 1
        resolved = sum(statuses.get(key, 0) for key in ("reconciled", "held", "awaiting_receipt"))
        return {
            "total_cases": len(cases),
            "statuses": statuses,
            "exception_count": statuses.get("exception", 0),
            "ready_count": statuses.get("ready", 0),
            "reconciled_count": statuses.get("reconciled", 0),
            "resolved_count": resolved,
            "value_under_review": sum(case.get("amount", 0) for case in cases if case["status"] in ("exception", "review")),
            "event_count": sum(len(case["events"]) for case in cases),
            "review_reduction_simulation": round(100 * resolved / max(1, len(cases)), 1),
            "metric_note": "Review reduction is a workflow simulation on the loaded queue, not a customer outcome.",
        }
