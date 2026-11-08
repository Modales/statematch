# Credits and sources

## Model and data

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Albert Gu and Tri Dao, 2023. https://arxiv.org/abs/2312.00752. StateMatch contains a compact, independently implemented selective SSM recurrence in portable PyTorch; it does not import the official CUDA-specific Mamba package.
- **BPI Challenge 2019** — purchase-order handling event log, published by 4TU.Centre for Research Data; organization listed by the publisher: Eindhoven University of Technology, Department of Mathematics and Computer Science. First published 2019-01-31. DOI: https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1. The log is anonymized by the source owner. The original dataset is not redistributed in this repository; use and citation remain subject to the publisher's terms.
- The application includes derived event traces and injected process deviations. Injected deviations are labeled and are not represented as real supplier fraud.
- **Outcome-oriented predictive process monitoring: Review and benchmark** — I. Teinemaa, M. Dumas, M. La Rosa, and F. M. Maggi, 2019. https://dl.acm.org/doi/abs/10.1145/3301300. The extended evaluation uses a chronological benchmark protocol and case-end ordered windows.
- **A Systematic Review of Anomaly Detection for Business Process Event Logs** — J. Ko and M. Comuzzi, 2023. https://link.springer.com/article/10.1007/s12599-023-00794-y. This motivates explicit limits around event-log anomaly claims.
- **BINet: Multivariate business process anomaly detection using deep learning** — T. Nolle, A. Seeliger, and M. Mühlhäuser, 2019. https://arxiv.org/abs/1902.03155. StateMatch reports case-level anomaly ranking and separates results by injected deviation type; it does not claim to reproduce BINet.
- Extended evaluation confidence intervals use 300 whole-case percentile bootstrap resamples with fixed seeds. Resampling whole traces preserves within-case token correlation; it does not turn synthetic deviations into observed fraud labels.

## Open-source runtime and interface

StateMatch uses FastAPI, Uvicorn, Pydantic, PyTorch, NumPy, scikit-learn, React, Vite, TypeScript, Recharts, Lucide, Tailwind CSS, and Playwright. Their packages retain their respective license notices in the installed dependencies and lockfiles.

- **IBM Plex Sans Variable** and **IBM Plex Mono**, distributed through Fontsource. IBM Plex is licensed under the SIL Open Font License 1.1. https://github.com/IBM/plex and https://openfontlicense.org/
- **Lucide icons**, ISC license. https://github.com/lucide-icons/lucide

No external LLM, hosted model API, stock photography, or generated demonstration imagery is used by the application. All measured model results come from the included training/evaluation pipeline; latency values are device-specific.

No license is asserted for the BPI source data beyond the publisher's own terms. No hosted deployment or production customer outcome is asserted.
