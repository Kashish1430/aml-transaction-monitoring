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
- Tests live in `tests/`, run via `pytest`, and CI (`.github/workflows/ci.yml`) runs ruff + pytest on pushes to `main` and on every pull request. 147 tests as of Phase 9.
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

CI (`.github/workflows/ci.yml`) runs on pushes to `main` and on **pull requests to any base branch** — the `pull_request` trigger is deliberately unfiltered (changed in Phase 7). It was previously restricted to `[main, master]`, which meant a PR based on anything other than `main` got no checks at all and the "wait for CI" step below had nothing to wait for.

**Always branch from an up-to-date `main` and target `main`.** Do not open a phase PR against another phase's branch. This was tried once in Phase 7 and went wrong: PR #7 was opened against `phase-6-explainability` after PR #6 had *already* merged, so GitHub never re-targeted it (auto-retargeting only fires when the base PR merges while the stacked PR is open). Merging #7 wrote Phase 7 onto a dead branch instead of `main`, and it took a third PR (#8) to land it. If a previous phase's PR is still open and the next phase genuinely depends on it, wait for the merge rather than stacking.

Practical loop for every phase (this is how Phases 2+ were actually done, follow the same shape):
1. `git checkout -b phase-N-<short-name>` from an up-to-date `main` (`git checkout main && git pull` first — verify the previous phase's PR is actually merged, don't assume).
2. Implement, test locally (`pytest -v`, `ruff check .`) until clean.
3. Commit, `git push -u origin phase-N-<short-name>`.
4. `gh pr create` with a summary of what changed and the real numbers/results produced.
5. Wait for CI (`gh pr checks <number>`), fix forward on the same branch if it fails.
6. **Let the user review and merge themselves** rather than auto-merging — they've been doing this via the GitHub UI ("Files changed" tab, then the merge button). Don't merge on their behalf unless they explicitly ask you to.
7. After merge: `git checkout main && git pull`, delete the local and remote feature branch (`git branch -d <name>`, `git push origin --delete <name>`).

## Current status

Phases 0-9 are done (Phases 0-8 merged to `main` via PRs #6, #7, #8, #10; Phase 9 on `phase-9-streamlit-app` — see `PLAN.md`'s Progress section for the authoritative per-phase checklist and what each phase actually produced, including deviations from the original plan). The project's headline result exists: **at equal recall (68.8%), the model raises 96.6% fewer alerts than the rules baseline on the test split** — see `reports/results.md`'s Phase 5 section. Phase 7 re-derived it excluding the dataset's anomalous tail (96.2%), so it's now known not to depend on that tail.

Phase 8 built `scripts/build_demo_artifact.py`, which writes the only two data files the deployed app reads: `app/data/demo_alerts.parquet` (15.3 MB, 30,000 rows x 126 cols) and `app/data/demo_metrics.json` (36 KB). Three things from it constrain Phase 9:

- **Never re-rank the artifact.** It is a *sample* whose `rank`/`model_score` are true values computed over the complete 1,015,669-row test split before sampling. The app must sort by the shipped `rank`, and must treat row counts below the alert queue as sample counts, not population counts (only the 10,011-row queue and the 1,797 positives are complete). CI asserts the queue is exactly ranks 1..N with no holes.
- **The waterfall is precomputed — don't add `shap` to `app/requirements.txt`.** Every row ships all 54 raw feature values as `feat_<name>` and their 54 SHAP contributions as `shap_<name>`. Base value is in the metrics JSON (0.123039, from `explain.shap_base_value`); base + contributions reconstruct the shipped score to 5.2e-07, asserted in CI. The `feat_`/`shap_` prefixes exist because `amount_paid_usd` is both a display field and a model feature — an unprefixed concat is a duplicate-column crash.
- **`is_tail_population` is a filter, not a defect.** The Phase 7 tail is deliberately over-represented (3.097% of the artifact vs. 0.109% of the test split) because all positives are kept; both shares and the tail-excluded headline are in the metrics JSON. Expose it, don't silently include or drop it.

Rebuilding the artifact takes one ~15-minute local run (`python -m scripts.build_demo_artifact`) and needs `data/processed/features.parquet`, `data/raw/`, and `models/` — none of which are in the repo. The committed artifact is therefore the only copy CI ever sees, which is why the integrity tests in `tests/test_build_demo_artifact.py` read the real file rather than a fixture.

Phase 7 added `src/monitoring.py` (PSI/CSI drift). Four things from it are load-bearing:

- **The 2022-09-11+ tail is a different population** — daily volume collapses from 654,467 rows to 11, laundering rate goes from ~0.09% to 59.12%, payment format to 100% ACH. It sits inside the test split and holds 36.4% of its positives in 0.109% of its rows. Any test-split number should be sanity-checked against it (Phase 7 did: the headline moves 96.6% → 96.2% without it). Phase 8's demo artifact sampling should not accidentally over- or under-represent it.
- **Never quantile-bin a binary feature for drift.** It collapses to one bin and reports PSI 0.0000 regardless of what happened — it did exactly that for `payment_format_ACH`, the model's top feature. `characteristic_stability` routes anything with ≤10 distinct reference values through `category_share_psi`; don't "simplify" that branch away.
- **Drift on this dataset is mostly artifact, and the writeups say so.** The 09-09 score-PSI step is the 7-day graph lookback evicting HI-Small's largest day for the first time, not a data change; most feature CSI is 7/30-day windows still warming up on a 17-18 day dataset. Don't quote "23 of 54 features drifted significantly" without that context.
- **`min_slice_size` (1,000) means the tail's daily slices are never scored** — reported as `insufficient_data`, deliberately. The fix is a volume monitor alongside PSI, not a lower floor.

Phase 6 added `src/explain.py` (SHAP + per-alert reason codes); two things from it are load-bearing for later phases:

- **Never read `shap.TreeExplainer.expected_value`** — on shap 0.45.1 + xgboost 2.0.3 it returns `logit(base_score)` until the first `shap_values()` call, then is silently replaced with a different, correct value. Use `explain.shap_base_value`, which derives the intercept and asserts additivity. Don't "simplify" it back.
- **Reason codes come in two variants** (`build_reason_code`'s `exclude_features`): faithful to raw SHAP, and behavioural with `PAYMENT_FORMAT_FEATURES` excluded from the *sentence only*. This exists because `payment_format_ACH` leads the faithful code on 99.86% of the alert queue, making it useless for triage. Phase 8's demo artifact should carry both; don't silently drop one.

Phase 9 built the app itself, as **two** modules rather than the one `PLAN.md` originally sketched:

- **`app/artifact.py` holds all the logic; `app/streamlit_app.py` is layout only.** A Streamlit script executes top-to-bottom on import, so logic living in the entrypoint can't be imported by a test without starting the UI. Keep new data-shaping code in `artifact.py`, where `tests/test_app_artifact.py` can reach it.
- **Never let the app import `src/`.** `app/requirements.txt` is streamlit/pandas/pyarrow/plotly/numpy only — no shap, xgboost or scikit-learn, none of which exist on Streamlit Cloud and none of which the app needs.
- **The `sys.path.insert` at the top of `streamlit_app.py` is load-bearing, not clutter.** `streamlit run app/streamlit_app.py` puts `app/` on the path, *not* the repo root, so `from app import artifact` raises `ModuleNotFoundError` on the real server. Both pytest (`pythonpath = ["."]`) and `AppTest` hide this by running with the repo root already present — the test suite structurally cannot catch it. Verify path-dependent changes by emulating the deployment path, not by running the suite.
- **`tests/test_app_smoke.py` runs the real app headless** through `streamlit.testing.v1.AppTest`, which is what makes PLAN.md's "click through all four tabs" check enforceable in CI. When adding a widget with a `format_func`, note that `AppTest` reports `.options` already formatted but `set_value` takes the *raw* value — passing `options[i]` double-formats and raises inside AppTest, not the app.

Next step is Phase 10 (deploy to Streamlit Community Cloud, get the public URL) in `PLAN.md`, on a new feature branch per the Git workflow above. Phase 10 is largely a hosting-console task rather than a code one; the repo side is already done.
