# Security and operational notes

StateMatch is a demonstration/research application. Before processing confidential procurement data, add authenticated users and roles, tenant isolation, TLS, rate limiting, audit-log export to append-only external storage, backup/restore testing, and organization-specific control validation.

- Do not commit `.env`, Vultr API keys, SSH keys, or customer data.
- Checkpoint loading uses PyTorch `weights_only=True`; only load model files from a trusted source.
- CSV uploads are capped at 2.5 MB and 256 events per case. This is not a substitute for request-rate limits or authentication.
- Critical workflow approval is guarded in the API, not only hidden in the interface.
- SQLite writes are serialized in one process. Use a production transactional store before running multiple writers or replicas.
- The per-case hash chain detects edits only when compared with the stored hash history; it is not externally anchored or signed.
- Demo injection is enabled by default for evaluation. Disable it with `STATEMATCH_DEMO=false` outside a demonstration.
- The container uses a non-root user, a health check, and `no-new-privileges`. Its named volume contains workflow data and must be protected appropriately.
- Production npm dependency audits are clean in the delivery environment. Remaining development-only advisory output should be reassessed when updating the toolchain; no forced major-version audit repair is applied.

Report vulnerabilities through a private channel to the repository owner rather than publishing secrets or sensitive case data in an issue.
