from __future__ import annotations

import copy
import csv
import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .inference import get_model
from .rules import derive_status, evaluate_rules
from .store import DATA, ROOT, Store

app = FastAPI(title="StateMatch API", version=__version__, description="Attention-free procure-to-pay controls and exception workflows")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)
store = Store()


class ActionRequest(BaseModel):
    action: str
    actor: str = Field(default="demo.analyst", min_length=2, max_length=80)
    note: str | None = Field(default=None, max_length=500)


class DemoRequest(BaseModel):
    scenario: str = "duplicate_invoice"


def public_case(case):
    return case


@app.get("/api/health")
def health():
    model = get_model()
    return {
        "status": "ok", "version": __version__, "model_loaded": True,
        "architecture": "Mamba selective state space", "attention": False,
        "llm": False, "device": "cpu", "checkpoint": Path(model.checkpoint).name,
    }


@app.get("/api/dashboard/summary")
def summary():
    return store.summary()


@app.get("/api/cases")
def cases(status: str | None = None, search: str | None = None):
    return {"items": store.list_cases(status, search)}


@app.get("/api/cases/{case_id}")
def case(case_id: str):
    try:
        return public_case(store.get_case(case_id))
    except KeyError:
        raise HTTPException(404, "Case not found")


@app.get("/api/cases/{case_id}/evidence")
def evidence(case_id: str):
    try:
        current = store.get_case(case_id)
    except KeyError:
        raise HTTPException(404, "Case not found")
    return {
        "case_id": case_id,
        "method": "Leave-one-event-out replay through the same Mamba checkpoint",
        "caveat": "Score changes are counterfactual evidence, not proof of causality.",
        "items": get_model().counterfactuals(current),
        "rules": current["rule_findings"],
    }


@app.post("/api/cases/{case_id}/actions")
def action(case_id: str, request: ActionRequest):
    try:
        updated, receipt = store.apply_action(case_id, request.action, request.actor, request.note)
        return {"case": updated, "receipt": receipt}
    except KeyError:
        raise HTTPException(404, "Case not found")
    except PermissionError as error:
        raise HTTPException(409, str(error))
    except ValueError as error:
        raise HTTPException(422, str(error))


@app.get("/api/cases/{case_id}/audit")
def audit(case_id: str, format: str = "json"):
    try:
        store.get_case(case_id)
    except KeyError:
        raise HTTPException(404, "Case not found")
    document = store.audit(case_id)
    if format == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["sequence", "created_at", "action", "actor", "note", "hash", "previous_hash", "valid"])
        for index, entry in enumerate(document["entries"], 1):
            writer.writerow([index, entry["created_at"], entry["action"], entry["actor"], entry["note"], entry["hash"], entry["previous_hash"], entry["valid"]])
        return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{case_id}-audit.csv"'})
    if format == "download":
        return JSONResponse(document, headers={"Content-Disposition": f'attachment; filename="{case_id}-audit.json"'})
    return document


@app.post("/api/demo/inject")
def inject_demo(request: DemoRequest):
    if os.environ.get("STATEMATCH_DEMO", "true").lower() != "true":
        raise HTTPException(403, "Demo injection is disabled")
    base = json.loads((DATA / "demo_template.json").read_text())
    events = copy.deepcopy(base["events"])
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    scenario = request.scenario
    if scenario == "duplicate_invoice":
        index = next((i for i, e in enumerate(events) if e["activity"] == "Record Invoice Receipt"), None)
        if index is None:
            raise HTTPException(422, "Demo template has no invoice")
        duplicate = copy.deepcopy(events[index])
        duplicate["id"] = f"DEMO-{stamp}:duplicate"
        events.insert(index + 1, duplicate)
    elif scenario == "premature_unblock":
        index = next((i for i, e in enumerate(events) if e["activity"] == "Record Goods Receipt"), 1)
        template = copy.deepcopy(events[max(0, index - 1)])
        template.update({"id": f"DEMO-{stamp}:unblock", "activity": "Remove Payment Block"})
        events.insert(index, template)
    elif scenario == "clear_before_gr":
        clear = next((i for i, e in enumerate(events) if e["activity"] == "Clear Invoice"), None)
        gr = next((i for i, e in enumerate(events) if e["activity"] == "Record Goods Receipt"), None)
        if clear is None or gr is None:
            raise HTTPException(422, "Demo template cannot support this scenario")
        event = events.pop(clear)
        event["id"] = f"DEMO-{stamp}:clear"
        events.insert(gr, event)
    else:
        raise HTTPException(422, "Unknown demo scenario")
    case = {
        key: copy.deepcopy(value) for key, value in base.items()
        if key not in ("events", "status", "resolution", "source", "synthetic_deviation")
    }
    case.update({"id": f"DEMO-{stamp}-{uuid.uuid4().hex[:4]}", "events": events, "source": f"Live demo · injected {scenario.replace('_', ' ')}", "synthetic_deviation": scenario, "resolution": None})
    model = get_model()
    scored = model.score(case)
    case.update({key: value for key, value in scored.items() if key != "activity_surprise"})
    case["rule_findings"] = evaluate_rules(case)
    case["links"] = model.links(case)
    case["status"] = derive_status(case, case["rule_findings"], case["model_risk"])
    store.save_case(case)
    with store.lock:
        store._append_audit(case["id"], "demo.injected", "demo.controller", f"Injected {scenario} into a real BPI template", {"scenario": scenario, "model_risk": case["model_risk"]})
    return case


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    raw = await file.read(2_500_001)
    if len(raw) > 2_500_000:
        raise HTTPException(413, "CSV upload is limited to 2.5 MB")
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        required = {"case_id", "activity", "timestamp"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Required columns: case_id, activity, timestamp")
        groups = {}
        errors = []
        for line, row in enumerate(reader, 2):
            try:
                case_id = row["case_id"].strip()
                activity = row["activity"].strip()
                timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).isoformat()
                if not case_id or not activity:
                    raise ValueError("case_id and activity cannot be blank")
                event = {
                    "id": f"{case_id}:upload-{line}", "activity": activity, "timestamp": timestamp,
                    "amount": float(row.get("amount") or 0), "user": row.get("user") or "upload",
                }
                groups.setdefault(case_id, {"id": case_id, "events": []})["events"].append(event)
            except (ValueError, TypeError) as error:
                errors.append(f"Line {line}: {error}")
        if not groups:
            raise ValueError("No valid event rows found")
        imported = []
        model = get_model()
        for case_id, case in groups.items():
            if len(case["events"]) > 256:
                errors.append(f"{case_id}: skipped; 256-event limit exceeded")
                continue
            case["events"].sort(key=lambda event: event["timestamp"])
            first = case["events"][0]
            case.update({
                "purchase_document": case_id.split("_")[0], "item": case_id.split("_")[-1],
                "vendor": first.get("user", "Uploaded vendor"), "vendor_name": first.get("user", "Uploaded vendor"),
                "category": "3-way match, invoice before GR", "spend_area": "Uploaded", "spend_class": "Uploaded",
                "item_type": "Uploaded", "amount": case["events"][-1]["amount"], "source": f"CSV upload · {file.filename}",
                "resolution": None,
            })
            scored = model.score(case)
            case.update({key: value for key, value in scored.items() if key != "activity_surprise"})
            case["rule_findings"] = evaluate_rules(case)
            case["links"] = model.links(case)
            case["status"] = derive_status(case, case["rule_findings"], case["model_risk"])
            store.save_case(case)
            with store.lock:
                store._append_audit(case_id, "case.ingested", "csv.upload", f"Imported from {file.filename}", {"events": len(case["events"])})
            imported.append(case_id)
        return {"imported": imported, "errors": errors, "count": len(imported)}
    except (UnicodeDecodeError, csv.Error, ValueError) as error:
        raise HTTPException(422, str(error))


@app.get("/api/model/metrics")
def metrics():
    return json.loads((DATA / "model_metrics.json").read_text())


@app.get("/api/model/architecture")
def architecture():
    model = get_model()
    return {
        "name": "StateSpaceProcessModel", "attention": False, "llm": False,
        "config": model.config.__dict__, "vocab": model.vocab.to_dict(),
        "stages": [
            {"id": "ingest", "name": "Event tensor", "detail": "Activity + category embeddings; amount, log Δtime, position"},
            {"id": "mamba", "name": "Selective state", "detail": "Input-dependent Δ, B and C update a recurrent state without attention"},
            {"id": "heads", "name": "Process outputs", "detail": "Next activity, token surprise, hidden-state link proposals"},
            {"id": "controls", "name": "Hard controls", "detail": "Deterministic receipt, clearance, duplicate and value checks"},
            {"id": "audit", "name": "Action receipt", "detail": "SQLite state transition with SHA-256 chained audit entries"},
        ],
    }


# Production: one Vultr VM serves both the API and the compiled React application.
DIST = ROOT / "app/dist"
if DIST.exists():
    assets = DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "API route not found")
        requested = (DIST / path).resolve()
        if path and requested.is_file() and DIST.resolve() in requested.parents:
            return FileResponse(requested)
        return FileResponse(DIST / "index.html")
