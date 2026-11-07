#!/usr/bin/env python3
"""Optional Vultr API deployment. Running without --dry-run creates a billable VM."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.vultr.com/v2"


def request(key, path, method="GET", payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    call = urllib.request.Request(API + path, data=body, method=method, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(call, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"Vultr API {error.code}: {detail}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs without contacting Vultr")
    args = parser.parse_args()
    key = os.environ.get("VULTR_API_KEY")
    ssh_key = os.environ.get("VULTR_SSH_KEY_ID")
    region = os.environ.get("VULTR_REGION", "ewr")
    plan = os.environ.get("VULTR_PLAN", "vc2-2c-4gb")
    cloud_init = (ROOT / "deploy/vultr-cloud-init.yaml").read_text()
    for required in ("Dockerfile", "docker-compose.yml"):
        if not (ROOT / required).is_file():
            raise SystemExit(f"Missing deployment input: {required}")
    checkpoint = ROOT / "modeling/artifacts/statematch_mamba.pt"
    encoded = ROOT / "modeling/artifacts/statematch_mamba.pt.base64"
    if not checkpoint.is_file() and not encoded.is_file():
        raise SystemExit("Missing the real Mamba checkpoint")
    print(f"Target: Vultr {region}; plan {plan}; Ubuntu 24.04; HTTP port 8000")
    if args.dry_run:
        print("Local deployment inputs are present. No VM was created.")
        return
    if not key or not ssh_key:
        raise SystemExit("Set VULTR_API_KEY and VULTR_SSH_KEY_ID. No VM was created.")
    plans = request(key, "/plans")["plans"]
    selected = next((item for item in plans if item["id"] == plan), None)
    if selected is None:
        raise SystemExit(f"Plan {plan} is not available to this account; set VULTR_PLAN")
    print(f"Billable plan: {plan}; monthly estimate returned by Vultr: ${selected.get('monthly_cost', 'unknown')}")
    systems = request(key, "/os")["os"]
    os_id = next((item["id"] for item in systems if "Ubuntu 24.04" in item["name"] and item.get("arch") == "x64"), None)
    if os_id is None:
        raise SystemExit("Could not find an x64 Ubuntu 24.04 image; no VM was created")
    instance = request(key, "/instances", "POST", {
        "region": region, "plan": plan, "os_id": os_id,
        "label": "statematch", "hostname": "statematch",
        "sshkey_id": [ssh_key], "enable_ipv6": True, "backups": "disabled",
        "user_data": base64.b64encode(cloud_init.encode()).decode(),
        "tags": ["statematch", "agent-rush"],
    })["instance"]
    instance_id = instance["id"]
    print(f"Created instance {instance_id}; waiting for an address and cloud-init build", flush=True)
    address = None
    for _ in range(180):
        current = request(key, f"/instances/{instance_id}")["instance"]
        address = current.get("main_ip")
        if current.get("status") == "active" and address and address != "0.0.0.0":
            break
        time.sleep(5)
    else:
        raise SystemExit(f"Instance {instance_id} did not become active within 15 minutes")
    record = {"instance_id": instance_id, "region": region, "plan": plan, "url": f"http://{address}:8000"}
    (ROOT / "deploy/deployment.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"VM address: {address}; waiting for StateMatch health", flush=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen(record["url"] + "/api/health", timeout=5) as response:
                health = json.load(response)
            if health.get("model_loaded"):
                print(json.dumps({**record, "health": health}, indent=2))
                return
        except (OSError, ValueError):
            time.sleep(5)
    print(f"VM exists but application health did not pass. Inspect instance {instance_id}.", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
