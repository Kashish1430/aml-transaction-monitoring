# CLAUDE.md — AML Alert-Triage System

Orientation for any Claude Code session working in this repository. Read this first, then `PLAN.md` for the phase-by-phase roadmap, then `AML_Transaction_Monitoring_Project_Brief.md` for the full technical spec this project implements.

## What this project is

An alert-triage system for anti-money-laundering (AML) transaction monitoring. It is **not** "a classifier that detects money laundering" — it re-ranks and filters the alerts a naive rules-based engine would raise, so a compliance analyst reviews far fewer false positives at equal or better recall of known laundering typologies. The headline result to defend at all times: *"at equal recall of labelled laundering, this model reduces the alert volume an analyst must review by X% versus the rules baseline."*

This is a CV/portfolio project aimed at banks and financial institutions, so correctness, honesty about numbers, and interview-defensibility matter more than raw model performance.

## Source of truth documents

- `AML_Transaction_Monitoring_Project_Brief.md` — the original technical brief: dataset, feature engineering, modelling, evaluation, explainability, monitoring requirements. Do not deviate from its framing without checking with the user.
- `PLAN.md` — the execution plan: phased build order, repo layout, deployment strategy, cost constraints. Update it as phases complete or the approach changes; keep it current, not aspirational.
- `reports/results.md` — once it exists, the single source of truth for all reported metrics. Never hand-type a number into the README that isn't traceable here.

## Hard constraints

- **Total cost: $0.** Every tool/service choice must have a free tier that covers this project's scale (see the cost ledger in `PLAN.md`). Flag it before introducing anything that could incur cost.
- **Must end in a live, deployed demo** (Streamlit Community Cloud target — see `PLAN.md`), not just notebooks.
- **No leakage.** Rolling/window features must use only information available strictly before each transaction's timestamp. Train/val/test splits are time-ordered, never random k-fold.
- **Accuracy is not a headline metric** at ~2% prevalence. Use precision@k, recall-per-typology, PR-AUC, and FP-reduction-at-equal-recall-vs-baseline instead.
- **Every reported number must be regenerable from code** (a script or notebook cell), not manually computed once and pasted.

## Repository conventions

- `src/` is the source of truth; notebooks in `notebooks/` are thin callers into `src/` for narrative/EDA purposes, not where logic lives.
- All paths, seeds, window sizes, and thresholds live in `config.yaml` — no hardcoded magic numbers scattered in scripts.
- Everything seeded and deterministic.
- `data/`, `models/` are gitignored (large/binary/regenerable). The only bundled data artifact committed to the repo is the small precomputed `app/data/demo_alerts.parquet` used by the live demo (target <20MB — see Phase 8 in `PLAN.md`).
- Tests live in `tests/`, run via `pytest`, and CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push.
- Commit per phase (see `PLAN.md`) with a clear message describing what became runnable.

## Environment

- Python 3.12, conda env `venve` in the project root (already created, currently empty — install `requirements.txt` into it).
- Core deps: pandas, numpy, scikit-learn, xgboost, lightgbm, networkx, shap, matplotlib, seaborn, pyyaml, jupyter, streamlit, pyarrow, pytest, ruff.
- No GPU required anywhere in this project.

## Dataset

IBM Transactions for Anti-Money Laundering (AML), Kaggle publisher `ealtman2019`, dataset `ibm-transactions-for-anti-money-laundering-aml`. Use **HI-Small** only (do not silently switch to a larger split — it changes runtime and memory assumptions throughout the plan). Fully synthetic, labelled with 8 laundering typologies (fan-in, fan-out, bipartite, stack, random, cycle, scatter-gather, gather-scatter).

Confirmed against the actual files (`notebooks/01_eda.ipynb`), not assumed:
- **5,078,345 transactions**, prevalence **~0.10%** (5,177 positives) — far more extreme than the brief's "~2%" estimate. Treat this as the real number everywhere; it makes the false-positive-reduction framing stronger, not weaker.
- **Date span is only ~17-18 days** (2022-09-01 to 2022-09-18). The `windows.rolling_days: [1, 7, 30]` in `config.yaml` still works, but the 30-day window effectively means "all history to date" for most transactions given the short span — not a bug, just know this when interpreting that feature.
- `HI-Small_Trans.csv` columns: `Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering`. Note the duplicate `Account` header (sender/receiver) — pandas auto-renames the second one to `Account.1`; `src/data_loader.py` handles this.
- `HI-Small_accounts.csv` columns: `Bank Name, Bank ID, Account Number, Entity ID, Entity Name` — entity ownership mapping only, no KYC/device/login fields (those belong to the separate mule-ring project, see [Future work] in `PLAN.md`, not this one).
- `HI-Small_Patterns.txt` is **not a CSV** — it's blocks of `BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>: <note>` / raw transaction rows / `END LAUNDERING ATTEMPT`. There is no transaction-ID column anywhere in this dataset, so `src/data_loader.load_patterns` labels 3,209 of the 5,177 laundering transactions (62%) by matching on shared field values; the remaining 38% are laundering but don't match a named typology block — expected, not a parsing bug.
- Data lives locally at `data/raw/` (gitignored) via the Kaggle API (`kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f <filename> -p data/raw`), authenticated via a token file at `~/.kaggle/access_token` (Kaggle's newer single-token flow, not the older `kaggle.json` username+key format).

## Deployment model (important — do not load raw data into the live app)

The deployed Streamlit app never runs feature engineering, model inference, or SHAP live. It only reads a precomputed artifact (`app/data/demo_alerts.parquet` + a small metrics JSON/parquet) built offline by `scripts/build_demo_artifact.py` from the fully-trained pipeline. Full-dataset numbers for `results.md`/README come from running the real pipeline locally, not from the demo app. If asked to change what the live app shows, regenerate the artifact via the script — don't wire the app to heavy computation directly, or it will break on Streamlit Community Cloud's free-tier resource limits.

## Current status

Phase 0 (scaffolding) and Phase 1 (data loading, validation, EDA) are done: `src/data_loader.py` is implemented and tested, HI-Small is downloaded locally, `notebooks/01_eda.ipynb` runs end to end and figures are saved in `reports/figures/`. Next step is Phase 2 (`src/rules_baseline.py`) in `PLAN.md`.
