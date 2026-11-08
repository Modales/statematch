# StateMatch

**Attention-free procure-to-pay reconciliation, built around a real Mamba sequence model.**

StateMatch reads purchase-item event traces, scores process behavior with a trained selective state-space model, applies deterministic financial controls, and gives an operations analyst a persistent exception workflow. It has no chat interface, attention layer, LLM inference, or LLM API dependency.

Built for the Vultr Agent Rush **Application of Technology** challenge: a non-LLM architecture performing a concrete business workflow rather than a model-only demonstration.

## What works

- **Real model inference:** a portable, two-block Mamba implementation in PyTorch, with the trained checkpoint bundled in the repository.
- **Exception queue:** searchable/filterable cases, per-event surprise, sequence NLL, next-activity predictions, and document values.
- **Match proposals:** receipt–invoice candidates combining Mamba hidden-state similarity, amount compatibility, and temporal decay.
- **Deterministic controls:** missing/late goods receipt, premature payment unblock, repeated invoice events, and value variance.
- **Guarded workflow:** critical findings block match approval. Holds, assignments, receipt requests, reopening, and clean-case approvals persist in SQLite.
- **Counterfactual evidence:** leave-one-event-out replay through the same checkpoint, with an explicit non-causal caveat.
- **Audit receipts:** per-case SHA-256 hash chains, verification, and JSON/CSV export.
- **Live demonstrations:** inject a controlled exception into a real BPI trace, then score and persist it immediately.
- **CSV ingestion:** upload actual traces for parsing, inference, controls, and persistence.
- **Model lab:** measured NLL, accuracy, macro F1, parameter counts, synthetic-deviation AUROC, and device-specific latency.

## A three-minute demonstration

1. Open **Operations**. Select an exception and inspect the highlighted trace and deterministic finding.
2. Open **Replay evidence**. Each row is computed by removing that event and replaying the trace through Mamba.
3. Open **Match ledger**, then **Audit chain**. Export the JSON or CSV receipt.
4. Choose **Inject exception → Duplicate invoice**. The new case is scored live; the repeated-invoice control prevents approval.
5. Hold clearance and reopen the audit chain to see the persisted action receipt.
6. Select a ready case and approve its match. Try the same action on a critical case: the API returns HTTP 409.
7. Open **Import CSV**, download the example, and import it. This is real ingestion, not a fixture swap.
8. Open **Model lab** for the measured comparison and **Architecture** for the decision boundary and source links.

## Architecture

```text
CSV / BPI event trace
        │
        ▼
Activity + category embeddings
amount + log Δtime + position
        │
        ▼
Mamba selective state-space blocks
(input-dependent Δ, B, C; no QKᵀ attention)
        │
        ├── Next-activity probability / token surprise
        ├── Sequence score / heuristic review risk
        └── Hidden-state receipt–invoice link proposals
        │
        ▼
Deterministic controls ─── Critical finding? ─── Approval blocked
        │
        ▼
Human workflow action → SQLite → SHA-256 chained audit receipt
        │
        ▼
React operations console + FastAPI, served by one process in production
```

The application model is a compact selective SSM inspired by the Mamba recurrence, not an LLM and not a wrapper around a hosted model. Its recurrence is implemented in `modeling/mamba.py` and runs on CPU without custom CUDA extensions.

## Measured experiment

The current checkpoint and metrics were produced on CPU with seed 19, four epochs, 6,000 training traces, 800 validation traces, and 800 test traces. Splitting is chronological before the training/evaluation caps. The activity vocabulary contains 42 tokens including special tokens; model inputs are capped at 48 observed events plus EOS.

| Model | Parameters | Test NLL ↓ | Accuracy ↑ | Macro F1 ↑ | Synthetic-deviation AUROC ↑ |
|---|---:|---:|---:|---:|---:|
| **Mamba** | 38,538 | **0.7143** | 75.56% | **0.3816** | **0.8699** |
| Transformer | 54,810 | 0.7167 | **76.40%** | 0.3632 | Not evaluated |
| Markov | 0 | 1.1986 | 56.44% | 0.2652 | Not evaluated |

Protocol:

- Mamba: `d_model=48`, `d_state=8`, `expand=2`, two layers.
- Transformer comparison: same width, expansion, layer count, and embeddings; it has more parameters.
- Checkpoints are selected by validation NLL. All three models include the EOS target in testing.
- Deviation AUROC uses 181 pairs of clean and explicitly mutated traces, not labeled fraud.
- Latency is measured on the experiment device after warm-up and is not a cross-hardware or asymptotic-superiority claim.
- Exact values and histories are in `modeling/artifacts/metrics.json`; the application serves the same measurements.

**Interpretation:** Mamba learned useful process structure and had slightly lower NLL and higher macro F1 than this particular Transformer run, while the Transformer had higher accuracy. This is one bounded experiment—not evidence of universal architectural superiority, real fraud detection, production savings, or superiority on every procurement process.

## Extended academic evaluation

`modeling/evaluate_extended.py` freezes the published checkpoint, replicates the original 800-case anchor, and scores the complete 2,663-case temporal test. The remaining 1,863 cases form a separate later-period evaluation with no retraining, threshold tuning, or test-case selection. Confidence intervals use 300 whole-case percentile bootstrap resamples, preserving within-case token correlation.

| Evaluation slice | Cases | NLL ↓ | Accuracy ↑ | Macro F1 ↑ |
|---|---:|---:|---:|---:|
| Original anchor | 800 | 0.7143 | 75.56% | 0.3816 |
| Complete temporal test | 2,663 | 0.8795 | 70.69% | 0.3580 |
| Later-period subset | 1,863 | 0.9538 | 68.50% | 0.3546 |

The replicated anchor NLL differs from the original report by only `1.64e-08`. Later-period 95% intervals are NLL `0.9025–1.0060`, accuracy `67.05%–69.81%`, and macro F1 `0.3388–0.4030`. Three equal-count, case-end ordered windows have NLL `0.9070`, `0.7530`, and `1.2381`: performance is not temporally stable, and the final window is materially harder. This is a monitoring/retraining requirement, not a hidden failure or an architecture-superiority claim.

The expanded deviation experiment uses at most one deterministic mutation per source case. Its AUROC is **0.9119** with a case-bootstrap 95% interval of **0.9028–0.9217**; 97.75% of mutated traces score higher than their paired original.

| Injected deviation | Pairs | AUROC | Mutated score higher than original |
|---|---:|---:|---:|
| Duplicate invoice | 690 | 0.8601 | 94.49% |
| Premature unblock | 887 | 0.9249 | 99.55% |
| Clearance before goods receipt | 286 | 0.9867 | 100.00% |

Long traces remain difficult: later-period accuracy is 72.77% for 5–8-event cases, 58.77% for 9–16 events, and 44.86% for 17+ events. The 17+ bucket contains only 31 cases, so it is diagnostic rather than a precise subgroup estimate. All slice, category, coverage, and deviation details are stored under `extended_evaluation` in both metrics files. These are injected process deviations on real BPI traces—not observed fraud labels.

## Data provenance

Source: **BPI Challenge 2019**, published by 4TU.Centre for Research Data.

- Official DOI: https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1
- 251,734 source purchase-item cases.
- Deterministic case-level sample: 21,302 traces using `SHA1(case_id)` first byte modulo 12.
- The interface ships 27 model-scored holdout cases: real traces plus clearly labeled injected deviations.
- Values and identifiers are anonymized in the source. Monetary amounts are source-log cumulative net worth, not independently audited balances.
- Dataset statistics are recorded in `data/dataset_stats.json`.

The original XES and processed trace sample are not committed. See **Reproduce the experiment** below. Dataset use remains subject to its original terms; see `CREDITS.md`.

## Run locally

Prerequisites: Python 3.12 and Node.js 22.12 or newer. No LLM credentials are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
npm ci
npm --prefix app ci
npm run build
npm test
npm start
```

Open http://localhost:8000. FastAPI serves both the API and the compiled React application. The database is created at `./statematch.sqlite3` on first startup; subsequent actions and uploads survive restarts.

For live frontend development:

```bash
npm run dev -- --host 127.0.0.1 --port 7100
```

This starts Uvicorn on port 8000 and Vite on the requested port, with `/api` proxied to the backend. Stop both with Ctrl-C.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `STATEMATCH_DB` | `./statematch.sqlite3` | SQLite path; use `/data/statematch.sqlite3` in the container |
| `STATEMATCH_DEMO` | `true` | Enable controlled exception injection |
| `PORT` | `8000` | Production HTTP port |
| `STATEMATCH_DEVICE` | `cpu` | Training device; `mps` is optional for local experiments |
| `STATEMATCH_PORT` | `8000` | Host port used by Docker Compose |

`.env.example` documents these settings. Never commit real API credentials.

## Verify

```bash
npm test                    # 6 API/workflow tests + 4 extended-evaluation tests
npm run lint                # React/TypeScript lint
npm run build               # Production frontend
npm run verify:checkpoint   # Byte identity + inference through bundled checkpoint
npx playwright install chromium
npm run verify              # 20 live desktop/mobile workflow checks
npm run verify:production   # 10 cold-start, compiled UI, model, chart, and API checks
```

The browser scripts start their own temporary servers and stop them in `finally`. They write process evidence to the ignored `verification/` directory. The checkpoint verifier checks the exact SHA-256 and exercises the base64 fallback.

Verified in the delivery environment:

- 10 backend and extended-evaluation tests passed.
- 20 browser checks passed, including CSV import, injection, replay, actions, audit, all four Model Lab charts, and mobile overflow.
- 10 production checks passed with zero browser console errors and zero failed HTTP responses.
- 11 live Vultr API/browser checks passed, including exact checkpoint identity, extended metrics, 27 seed cases, counterfactual inference, desktop/mobile charts, and no mobile page overflow, console errors, or failed HTTP responses.
- Frontend lint passed; production dependency audits reported zero vulnerabilities.
- Checkpoint SHA-256: `5c07a9217910e732debdd29f71028838b468875d815deea6c7f4d8afd9aa07e5`.

## Docker

```bash
docker compose up --build -d
curl http://localhost:8000/api/health
docker compose ps
```

The named `statematch-data` volume preserves `/data/statematch.sqlite3`. The application runs as a non-root user with a container health check. To disable injection, set `STATEMATCH_DEMO=false` before startup.

```bash
docker compose logs -f
docker compose down          # preserves the database volume
```

## Deploy on Vultr

**Live demo:** http://64.176.194.39:8000  
**Health:** http://64.176.194.39:8000/api/health

Deployed on 2026-11-08 to Vultr Cloud Compute instance `62d33ad4-7a2a-4b11-a4e4-4873938e10d0` in `ewr`, using `vc2-2c-4gb` (2 vCPU, 4 GB RAM; Vultr estimate $20/month), Ubuntu 24.04. Cloud-init completed and the container is healthy.

Eleven live API/browser checks passed, including checkpoint identity, extended metrics, counterfactual inference, desktop/mobile charts, zero console errors, and zero failed HTTP responses.

This is a public HTTP demonstration without authentication or HTTPS. Do not upload sensitive data. The VM remains running and billable. Do not rerun `deploy/vultr.py` against this deployment: it creates a new VM rather than updating the existing one.

### Option A: cloud-init in the Vultr console

1. Create a **Cloud Compute** VM with an x64 Ubuntu 24.04 image and at least 4 GB RAM.
2. Select your SSH key and supply the contents of `deploy/vultr-cloud-init.yaml` as cloud-init user data.
3. Cloud-init installs Docker, clones `https://github.com/Modales/statematch`, starts the Compose service, configures the firewall, and checks application health.
4. Open `http://YOUR_SERVER_IP:8000` and verify `http://YOUR_SERVER_IP:8000/api/health`.

This exposes the demo over HTTP port 8000. Put it behind HTTPS and authentication before using sensitive data.

### Option B: create the VM through the Vultr API

```bash
export VULTR_API_KEY='your-key'
export VULTR_SSH_KEY_ID='your-ssh-key-id'
export VULTR_REGION='ewr'             # optional
export VULTR_PLAN='vc2-2c-4gb'       # optional; must be available to your account
python3 deploy/vultr.py --dry-run     # no API request, no VM created
python3 deploy/vultr.py               # creates a billable VM
```

The script discovers the Ubuntu image from Vultr, submits cloud-init, waits for the public address and health check, and writes non-secret instance metadata to the ignored `deploy/deployment.json`. A failed health check is reported as a failure, never as a successful deployment.

For an existing Ubuntu VM, clone the repository and use the Docker commands above. The Dockerfile uses CPU PyTorch wheels and a multi-stage frontend build; local image construction was not executable in the original delivery environment because Docker was unavailable.

## CSV contract

Required columns:

- `case_id`: case identifier, shared by its event rows.
- `activity`: process activity name.
- `timestamp`: ISO-8601 timestamp, for example `2026-01-03T09:00:00Z`.

Optional columns: `amount` (numeric) and `user` (actor/vendor label).

```csv
case_id,activity,timestamp,amount,user
DEMO-1001,Create Purchase Order Item,2026-01-01T09:00:00Z,1250,vendor_17
DEMO-1001,Record Goods Receipt,2026-01-02T09:00:00Z,1250,user_08
DEMO-1001,Record Invoice Receipt,2026-01-03T09:00:00Z,1250,user_11
DEMO-1001,Clear Invoice,2026-01-04T09:00:00Z,1250,user_12
```

Uploads are limited to 2.5 MB and 256 events per case. Unknown activities use the model's unknown token. The current upload adapter marks the category as `3-way match, invoice before GR`; that assumption must be replaced by actual ERP metadata in a production connector.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Checkpoint, architecture, and runtime status |
| GET | `/api/cases` | Queue, with optional `status` and `search` |
| GET | `/api/cases/{id}` | Complete case and model outputs |
| GET | `/api/cases/{id}/evidence` | Leave-one-event-out replay |
| POST | `/api/cases/{id}/actions` | Guarded persistent workflow action |
| GET | `/api/cases/{id}/audit?format=csv` | Verify/export audit chain (`json`, `csv`, `download`) |
| POST | `/api/demo/inject` | Score a controlled exception |
| POST | `/api/ingest` | Multipart CSV ingestion |
| GET | `/api/model/metrics` | Measured experiment |
| GET | `/api/model/architecture` | Runtime configuration and processing stages |
| GET | `/api/dashboard/summary` | Loaded-queue statistics |

Interactive API documentation is available at `/docs`.

## Reproduce the experiment

1. Download the official BPI Challenge 2019 XES file from the DOI above and place it at `data/raw/BPI_Challenge_2019.xes`.
2. Install `requirements-dev.txt`.
3. Run the deterministic pipeline:

```bash
python3 modeling/prepare.py
python3 modeling/train.py --epochs 4 --max-train 6000 --max-eval 800
python3 modeling/export_cases.py
npm run evaluate:extended
```

Training defaults to CPU and writes a new checkpoint and measured metrics. Export regenerates the scored application cases. The extended command evaluates the bundled base64 checkpoint on the complete chronological holdout and updates both metrics files; it does not retrain or modify model weights. Existing SQLite state is intentionally not overwritten; use a new `STATEMATCH_DB` path to seed a fresh demo without losing prior actions.

The raw checkpoint is ignored by Git to keep ordinary source operations lightweight. The exact trained bytes are committed as `modeling/artifacts/statematch_mamba.pt.base64`. If the raw `.pt` file is absent, `ModelService` decodes the bundled base64 in memory and loads it with `weights_only=True`. Training regenerates the raw file, which then takes precedence. The base64 file is not a substitute model or a collection of precomputed scores; it is the complete serialized PyTorch checkpoint.

## Boundaries and production work still required

- This is a functional reconciliation/exception prototype, **not a live SAP/ERP integration**. Approvals update StateMatch's internal workflow only; no supplier payment is made.
- The supplied deployment has no authentication or tenant isolation and starts with demo injection enabled. Add SSO/RBAC, CSRF protection where applicable, TLS, rate limits, secrets management, backups, and organization-specific policy before sensitive use.
- Duplicate detection is a deliberately simple same-value/two-hour heuristic. It is not proof of a duplicate or fraud. Production controls should use invoice identifiers, PO/receipt quantities, tax, currency, credit memos, and partial-invoice semantics.
- The displayed risk is a heuristic transformation of sequence surprise, not a calibrated probability of financial loss. Hidden-state links are candidate associations, not validated ground-truth matches.
- Leave-one-event-out changes are counterfactual evidence, not causal explanations.
- Hash chains are tamper-evident within the stored records, not tamper-proof against a database administrator. Use signed/external append-only storage for stronger guarantees.
- Queue “review reduction” is only a simulation over the loaded cases, never a claimed customer outcome.

## Repository map

```text
api/app/                 FastAPI, inference, controls, SQLite, scored seed cases
api/tests/               Workflow and persistence tests
app/src/                 React operations console and measured model lab
modeling/                Mamba, dataset preparation, training, export
modeling/artifacts/      Metrics, vocabulary, and complete encoded checkpoint
scripts/                 Development runner and independent verification
deploy/                  Vultr cloud-init and optional API provisioning
Dockerfile               CPU production image
```

## Foundations

- Mamba paper: https://arxiv.org/abs/2312.00752
- BPI Challenge 2019: https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1
- Predictive-monitoring review and benchmark: https://dl.acm.org/doi/abs/10.1145/3301300
- Event-log anomaly detection review: https://link.springer.com/article/10.1007/s12599-023-00794-y
- BINet anomaly detection: https://arxiv.org/abs/1902.03155
- Source repository: https://github.com/Modales/statematch

See `CREDITS.md` for attribution and third-party notices.
