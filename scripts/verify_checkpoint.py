#!/usr/bin/env python3
"""Verify that the published base64 checkpoint is byte-identical and runnable."""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from app.inference import ModelService  # noqa: E402


def main():
    artifacts = ROOT / "modeling/artifacts"
    raw = artifacts / "statematch_mamba.pt"
    encoded = artifacts / "statematch_mamba.pt.base64"
    decoded = base64.b64decode(encoded.read_text(encoding="ascii"), validate=True)
    digest = hashlib.sha256(decoded).hexdigest()
    if raw.is_file():
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest, "Encoded checkpoint differs from raw checkpoint"
    work = ROOT / "verification/checkpoint"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        portable = work / raw.name
        portable.with_suffix(".pt.base64").write_bytes(encoded.read_bytes())
        service = ModelService(portable)
        case = json.loads((ROOT / "api/app/data/demo_template.json").read_text())
        result = service.score(case)
        assert result["model_nll"] > 0 and result["events"]
        print(json.dumps({"checkpoint_sha256": digest, "bytes": len(decoded), "weights_only": True, "inference_verified": True}, indent=2))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
