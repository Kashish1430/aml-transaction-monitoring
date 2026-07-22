# CLAUDE.md — AML Alert-Triage System

Orientation for any Claude Code session working in this repository. Read this first, then `PLAN.md` for the phase-by-phase roadmap, then `AML_Transaction_Monitoring_Project_Brief.md` for the full technical spec this project implements.

## What this project is

An alert-triage system for anti-money-laundering (AML) transaction monitoring. It is **not** "a classifier that detects money laundering" — it re-ranks and filters the alerts a naive rules-based engine would raise, so a compliance analyst reviews far fewer false positives at equal or better recall of known laundering typologies. The headline result to defend at all times: *"at equal recall of labelled laundering, this model reduces the alert volume an analyst must review by X% versus the rules baseline."*

This is a CV/portfolio project aimed at banks and financial institutions, so correctness, honesty about numbers, and interview-defensibility matter more than raw model performance.

## Source of truth documents

- `AML_Transaction_Monitoring_Project_Brief.md` — the original technical brief: dataset, feature engineering, modelling, evaluation, explainability, monitoring requirements. Do not deviate from its framing without checking with the user.
- `PLAN.md` — the execution plan: phased build order, repo layout, deployment strategy, cost constraints. Update it as phases complete or the approach changes; keep it current, not aspirational.
- `reports/results.md` — once it exists, the single source of truth for all reported metrics. Never hand-type a number into the README that isn't traceable here.
- `reports/challenges.md` — narrative log of non-obvious problems hit during the build (root cause, fix, why), kept for interview storytelling. Add an entry whenever a real bug or performance problem gets diagnosed and fixed — not routine implementation work, and not until the actual root cause is understood (see its own entries for the level of specificity expected).
- `portfolio/business_overview.md` and `portfolio/technical_overview.md` — living portfolio write-ups for non-technical and technical readers respectively, feeding the eventual README (Phase 11). Update both at the end of every phase that changes what's built or what's known (new stage summary in the business doc, new technical section/numbers in the technical doc) — same "keep it current, not aspirational" rule as `PLAN.md`. Every number in either file must trace to `reports/results.md` or `reports/challenges.md`; never state a forward-looking result (e.g. the headline FP-reduction number) before it's actually been computed.
- `investigations/` — standalone, runnable scripts behind the non-obvious findings in `reports/challenges.md` (root-cause profiling, ablations, correlation checks, verification of a number before trusting it). Excluded from `ruff` (see `pyproject.toml`) since these are one-off diagnostics, not maintained pipeline code — but every script must actually run and print real, current numbers, not be a stale historical artifact. When an ad-hoc analysis surfaces a finding worth a `challenges.md` entry, save the script here (organized by phase, see `investigations/README.md`) rather than leaving it in a scratch/temp directory — it's what makes a claimed number re-verifiable instead of just asserted.

## Hard constraints

- **Total cost: $0.** Every tool/service choice must have a free tier that covers this project's scale (see the cost ledger in `PLAN.md`). Flag it before introducing anything that could incur cost.
- **Must end in a live, deployed demo** (Streamlit Community Cloud target — see `PLAN.md`), not just notebooks.
- **No leakage.** Rolling/window features must use only information available strictly before each transaction's timestamp. Train/val/test splits are time-ordered, never random k-fold.
- **Accuracy is not a headline metric** at ~0.10% prevalence (confirmed real number, see Dataset section — far more extreme than the brief's ~2% estimate). Use precision@k, recall-per-typology, PR-AUC, and FP-reduction-at-equal-recall-vs-baseline instead.
- **Every reported number must be regenerable from code** (a script or notebook cell), not manually computed once and pasted.

## Repository conventions

- `src/` is the source of truth; notebooks in `notebooks/` are thin callers into `src/` for narrative/EDA purposes, not where logic lives.
- All paths, seeds, window sizes, and thresholds live in `config.yaml` — no hardcoded magic numbers scattered in scripts.
- Everything seeded and deterministic.
- `data/`, `models/` are gitignored (large/binary/regenerable). The only bundled data artifact committed to the repo is the small precomputed `app/data/demo_alerts.parquet` used by the live demo (target <20MB — see Phase 8 in `PLAN.md`).
- Tests live in `tests/`, run via `pytest`, and CI (`.github/workflows/ci.yml`) runs ruff + pytest on every push.
- Commit per phase (see `PLAN.md`) with a clear message describing what became runnable.

## Environment

- Python 3.12, conda env `venve` in the project root. `requirements.txt` is fully installed into it already (including `kaggle` for data acquisition) — don't reinstall from scratch, just add new packages to `requirements.txt` and `pip install` the delta if a phase needs something new.
- Core deps: pandas, numpy, scikit-learn, xgboost, lightgbm, networkx, shap, matplotlib, seaborn, pyyaml, jupyter, streamlit, pyarrow, pytest, ruff, kaggle.
- No GPU required anywhere in this project.
- **How to invoke tools in `venve`** (Windows conda layout, not obvious): `python.exe` lives at the env root (`venve/python.exe`), but console-script tools (`pytest`, `ruff`, `kaggle`, `jupyter-nbconvert`, etc.) live in `venve/Scripts/`. There is no plain `jupyter.exe` — use `venve/Scripts/jupyter-nbconvert.exe` directly. C: drive has previously run low on space; if `pip install` fails with "No space left on device", redirect cache/temp to D: (`--cache-dir` flag / `TMPDIR`/`TEMP`/`TMP` env vars pointed at a folder under the project), don't assume D: itself is full.
- `.vscode/settings.json` (gitignored, local only) points the IDE's Python interpreter at `venve/python.exe` for autocomplete/lint in-editor — recreate it if it's missing and imports show as unresolved in the editor.

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

## Git workflow

`main` has GitHub branch protection (set up 2026-07-20) — this is a hard requirement, not a suggestion:
- No direct pushes to `main`, **including from admins** (`enforce_admins: true`) — every change goes through a PR.
- The `lint-and-test` CI check (ruff + pytest, from `.github/workflows/ci.yml`) must pass before a PR can merge.
- `required_approving_review_count` is `0` — a human approval is *not* required to merge (would deadlock a solo repo, since GitHub blocks self-approval). CI passing is the actual gate.
- Force-pushes and branch deletion are disabled on `main`.

Practical loop for every phase (this is how Phases 2+ were actually done, follow the same shape):
1. `git checkout -b phase-N-<short-name>` from an up-to-date `main`.
2. Implement, test locally (`pytest -v`, `ruff check .`) until clean.
3. Commit, `git push -u origin phase-N-<short-name>`.
4. `gh pr create` with a summary of what changed and the real numbers/results produced.
5. Wait for CI (`gh pr checks <number>`), fix forward on the same branch if it fails.
6. **Let the user review and merge themselves** rather than auto-merging — they've been doing this via the GitHub UI ("Files changed" tab, then the merge button). Don't merge on their behalf unless they explicitly ask you to.
7. After merge: `git checkout main && git pull`, delete the local and remote feature branch (`git branch -d <name>`, `git push origin --delete <name>`).

## Current status

Phases 0-5 are done (Phase 5 in PR review as of this writing — see `PLAN.md`'s Progress section for the authoritative per-phase checklist and what each phase actually produced, including deviations from the original plan). The project's headline result now exists: **at equal recall (68.8%), the model raises 96.6% fewer alerts than the rules baseline on the test split** — see `reports/results.md`'s Phase 5 section. Next step is Phase 6 (`src/explain.py`) in `PLAN.md`, on a new feature branch per the Git workflow above.
