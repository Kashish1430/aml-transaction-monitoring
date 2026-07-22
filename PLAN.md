# AML Alert-Triage System — Build & Deployment Plan

## Context

This project turns `AML_Transaction_Monitoring_Project_Brief.md` into a deployed, CV-ready portfolio piece for banks/financial institutions. The brief specifies the ML pipeline: an alert-triage system that re-ranks/filters the alerts a rules-based AML engine would raise, using the IBM AML Kaggle dataset (HI-Small), with graph/motif features, a tuned gradient-boosting model, analyst-style evaluation (precision@k, recall-per-typology, FP-reduction-at-equal-recall), SHAP reason codes, and PSI drift monitoring.

The brief does not cover deployment. This plan adds a deployment layer on top of the brief's pipeline and sequences the whole build into phases, each ending in something runnable/checkable, so work can resume across sessions without losing the thread. Two hard constraints from the project owner: the finished system must be **actually deployed with a live link**, and total cost must be **$0**.

Starting state (2026-07-19, historical): empty directory except the brief and a local conda env (`venve`, Python 3.12, no packages installed). No git repo yet. See **Progress** below for current state — this section is left as-is for context on where the project started.

## Progress (keep this current — see CLAUDE.md's instruction not to let this doc go stale)

- **Phase 0 — DONE.** Repo scaffolded, pushed to `github.com/Kashish1430/aml-transaction-monitoring` (public), CI green.
- **Phase 1 — DONE.** `src/data_loader.py` loads/validates all three HI-Small files. Real numbers differ from this plan's original estimates: **prevalence is ~0.10%** (not ~2%), **date span is ~17-18 days**, and only 62% of laundering-labelled transactions map to a named typology in `Patterns.txt` (the rest are laundering but unpatterned — expected, not a bug). Full detail in `CLAUDE.md`'s Dataset section.
- **Phase 2 — DONE.** `src/rules_baseline.py` built with one addition beyond the original plan: transaction amounts are normalized to USD via a static FX table (`config.yaml`'s `fx_rates_to_usd`) before thresholding, because the dataset's 15 currencies operate at wildly different numeric scales (checked empirically before building the rule — a flat threshold would have been silently wrong). Results in `reports/results.md`: **36.3% alert rate, 0.17% precision, 60.6% recall** on the full HI-Small dataset — this is the number Phase 4's model must beat.
- **Phase 3 — DONE.** `src/features.py` (account/window velocity, volume, structuring score, pass-through ratio across [1,7,30]-day windows) and `src/graph_features.py` (directed account graph via networkx: degree, fan-in/fan-out score, short-cycle membership) joined via `build_modelling_table` into one 52-column table, run end-to-end in `notebooks/02_feature_engineering.ipynb`, saved to `data/processed/features.parquet`. Full HI-Small run: 5,078,345 rows in ~1532s (~25.5 min), no NaNs. Two performance findings that changed the design from the original plan:
  - `_rolling_distinct_counterparties`'s first implementation used `DataFrame.groupby` per account and was 30-40x slower than necessary — most accounts have only 1-2 transactions in an 18-day window, so groupby's per-group object-construction overhead dominated over the actual O(1)-per-row two-pointer work. Rewritten with a single global sort + raw numpy array slicing per account-group boundary (no `DataFrame.groupby` in the hot path).
  - The graph features were originally planned as "all history to date" per day (per this doc's Phase 3 description), but HI-Small's first 2 days alone contain ~1.9M of its 5.08M transactions, so an unbounded accumulating graph never stopped growing across the 18-day span — profiling showed 521s to process just the first 9 days on a still-worsening curve. Fixed by bounding each day's graph to a trailing 7-day lookback (`graph_features.lookback_days` in `config.yaml`) with incremental edge pruning (add each edge once, remove once when it ages out — not a rebuild-from-scratch per day, which was tried and was *slower* due to redundant re-insertion). Also reduced `cycle_max_length` from 4 to 3. Both are documented as deliberate, defensible simplifications (recent structure matters more than 15-day-old edges for a typology playing out over hours-to-days) in `src/graph_features.py`'s module docstring, not silent workarounds.
- **Phase 4 — DONE.** `src/model.py`: XGBoost (`tree_method="hist"`, early-stopped on PR-AUC against a validation split) plus an Isolation Forest unsupervised layer, on a strictly time-ordered 60/20/20 split (`time_ordered_split`) with `scale_pos_weight` (1,324.94) computed from the training split only. Run end-to-end in `notebooks/03_modelling.ipynb`, saved to `models/` (gitignored). Sanity-check numbers (not the Phase 5 headline): test PR-AUC 0.3967, ROC-AUC 0.9828, precision@100 92% — full numbers and caveats in `reports/results.md`. Two findings investigated in depth, not just reported, full writeups in `reports/challenges.md`:
  - `payment_format_ACH` dominates feature importance because 86.6% of all laundering transactions use ACH format (its laundering rate is ~7.3x the dataset's overall prevalence) — an ablation (retraining without `payment_format`) shows PR-AUC falls to 0.0705, confirming a large share of the headline numbers depend on this one categorical correlation, which may be real behavior or a synthetic-generator artifact. The features that rise to the top without it are the Phase 3 account/window and graph features, not noise — the more important result for this project's thesis.
  - The Isolation Forest layer has 0% top-1000 overlap with XGBoost and catches only 1/1,797 test positives — diagnosed as rediscovering extreme-volume hub accounts (raw count/amount features are heavily right-skewed, and isolation-based detection can't distinguish "extreme but legitimate" from "extreme and suspicious"). Log-transforming heavy-tailed features before fitting is noted as future work, not implemented, to keep this phase's scope disciplined.
- **Phase 5 — DONE.** `src/evaluate.py`: precision@k, recall-per-typology, PR-AUC/ROC-AUC, calibration, and the headline false-positive-reduction-vs-rules-baseline-at-equal-recall number. Run end-to-end in `notebooks/04_evaluation_explainability.ipynb`. **HEADLINE RESULT: at 68.8% recall (matched to the rules baseline's own test-split recall), the model raises 10,011 alerts vs. the rules baseline's 297,564 on the same test population — a 96.6% reduction.** Full numbers, recall-per-typology table (67.6%-89.8% across all 8 patterns, no typology dramatically weaker), and caveats in `reports/results.md`. Three things built or found along the way, full writeups in `reports/challenges.md`:
  - The typology-label join (`data_loader.join_pattern_types`) had never actually been implemented — `load_patterns`'s docstring had pointed at "PLAN.md Phase 3" since Phase 1, but Phase 3 never needed it and never built it. Implemented and verified (zero duplicate join keys, recovers exactly the previously-documented 62%/3,209-of-5,177 match rate) — the stale docstring reference is corrected too.
  - The rules-baseline comparison had to be re-derived on the test split specifically, not reused from Phase 2's full-dataset number (different, non-comparable population) and not recomputed on an isolated test-only slice (would cold-start the structuring/pass-through rules' rolling windows right at the split boundary). Computed on the full dataset first, then restricted to the test split's rows by index — the test-split rules baseline recall (68.8%) differs from Phase 2's full-dataset figure (60.6%) as a result, expected and disclosed, not a discrepancy.
  - The model's raw scores are not calibrated probabilities (mean predicted score ~41x actual test prevalence) — the direct, expected consequence of `scale_pos_weight`. Diagnosed and disclosed; doesn't affect any reported number since precision@k/recall-per-typology/the headline all depend only on ranking, not the literal score value. Post-hoc calibration (Platt/isotonic) noted as future work, not implemented.
- **Phase 6 onward — NOT STARTED.** Next up: `src/explain.py` — SHAP TreeExplainer, global feature importance, per-alert plain-English reason codes.
- **Process addition not in the original plan:** GitHub branch protection was added to `main` after Phase 2 (2026-07-20) — PR + passing CI required for every change from here on, including admins. Exact rules and the working loop are documented in `CLAUDE.md`'s Git workflow section; follow that for Phase 3 onward.

## Key design decisions

**Deployment target: Streamlit Community Cloud (free tier).** Deploys straight from a public GitHub repo, zero infra to manage, the standard portfolio format for ML/data projects — a bank risk/compliance reviewer will recognize it instantly. Fallback if it has issues: Hugging Face Spaces (also free, supports Streamlit).

**The live demo is an "Analyst Alert-Triage Dashboard,"** not a bare model endpoint — matches the brief's own framing ("a working demo... beats any vanity metric"):
- A ranked alert queue (score, account, amount, typology, SHAP reason code) with typology/bank/date filters.
- A single-alert drill-down showing the SHAP waterfall + the plain-English reason code.
- A results/metrics tab reproducing `results.md`: PR curve, recall-per-typology bar chart, and the headline "false-positive reduction at equal recall vs. rules baseline" comparison.

**Free-tier resource handling.** Streamlit Community Cloud gives ~1 CPU / ~1GB RAM. HI-Small has millions of rows — too much to load raw in the deployed app. Strategy: the **full pipeline runs locally/offline** against the complete HI-Small dataset for all reported numbers (`results.md`, README headline stat, notebooks). For the **live app**, we precompute a bundled demo artifact — a scored, feature-joined, SHAP-explained sample of a few tens of thousands of transactions (all positives + a stratified sample of negatives, target <20MB as parquet) — checked into the repo. The app loads only this precomputed table; it never re-runs feature engineering or SHAP live. This is standard practice and will be disclosed honestly in the README.

**Data acquisition is one-time and local** via the Kaggle API — not needed at deploy time since only the precomputed demo artifact ships to Streamlit Cloud.

## Cost ledger (target: $0)

| Item | Choice | Cost |
|---|---|---|
| Dataset | Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml, HI-Small | Free |
| Compute (pipeline dev) | Local laptop / conda env `venve` | Free |
| Source control | GitHub public repo | Free |
| CI (lint/tests on push) | GitHub Actions (free minutes on public repos) | Free |
| Live demo hosting | Streamlit Community Cloud | Free |
| Explainability | SHAP (local library, TreeExplainer) | Free |
| Domain | `*.streamlit.app` subdomain | Free |

No paid APIs — reason codes come from SHAP + a template, not an LLM call.

## Repository layout

Extends the brief's scaffold with what deployment and engineering hygiene require:

```
aml-transaction-monitoring/
├── README.md
├── PLAN.md                     # this file
├── CLAUDE.md                   # orientation for AI-assisted sessions
├── requirements.txt            # full/dev deps (pipeline + notebooks)
├── app/
│   ├── requirements.txt        # slim deps for Streamlit Cloud
│   ├── streamlit_app.py        # entrypoint: Alert Queue / Alert Detail / Model Performance / About
│   └── data/
│       └── demo_alerts.parquet # precomputed, bundled scored sample
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/                    # gitignored
│   └── processed/              # gitignored
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_evaluation_explainability.ipynb
├── src/
│   ├── data_loader.py
│   ├── rules_baseline.py
│   ├── features.py
│   ├── graph_features.py
│   ├── model.py
│   ├── evaluate.py
│   ├── explain.py
│   └── monitoring.py
├── scripts/
│   └── build_demo_artifact.py  # produces app/data/demo_alerts.parquet from the trained pipeline
├── tests/
│   ├── test_data_loader.py
│   ├── test_rules_baseline.py
│   ├── test_features.py        # esp. leakage: rolling windows use only past data
│   └── test_evaluate.py
├── models/                      # gitignored; only demo parquet ships to the app
├── reports/
│   ├── figures/
│   └── results.md
├── .github/workflows/ci.yml    # lint (ruff) + pytest on push/PR
├── .gitignore
├── config.yaml
└── LICENSE                     # MIT
```

## Execution phases

Each phase ends with something runnable/checkable — no phase depends on trusting an earlier phase blindly.

**Phase 0 — Project scaffolding & environment [DONE]**
- `git init`, create the folder tree above, `.gitignore` (data/, models/, `__pycache__`, `.ipynb_checkpoints`, venv).
- `requirements.txt` pinned: pandas, numpy, scikit-learn, xgboost, lightgbm, networkx, shap, matplotlib, seaborn, pyyaml, jupyter, streamlit, pyarrow, pytest, ruff.
- Install into the existing `venve` conda env.
- `config.yaml` with paths, seeds, window sizes, thresholds.
- Push scaffolded repo to a new public GitHub repo.
- `.github/workflows/ci.yml` running ruff + pytest on push.
- **Check:** installed packages match requirements; CI runs green on the initial commit.

**Phase 1 — Data acquisition, validation, EDA [DONE]**
- Kaggle API download of HI-Small into `data/raw/` (gitignored).
- `src/data_loader.py`: explicit schema validation (fail loudly on missing columns), timestamp parsing, unified account key (bank+account).
- `notebooks/01_eda.ipynb` calling into `data_loader`: class balance, amount distributions (log scale) legit vs. laundering, volume over time, most-active accounts, distribution across the 8 typologies. Figures saved to `reports/figures/`.
- **Check:** dataset stats logged (row count, class balance, date span, unique accounts); figures render and look sane. Actual: ~0.10% prevalence, not the ~2% originally estimated here — see Progress section above.

**Phase 2 — Rules baseline [DONE]**
- `src/rules_baseline.py`: large-amount threshold, structuring (many sub-threshold txns in a short window), rapid pass-through.
- Record alert count, precision, recall against labels — the number every later step must beat.
- **Check:** `tests/test_rules_baseline.py` on synthetic mini-cases (a hand-built structuring pattern must fire).

**Phase 3 — Feature engineering [DONE]**
- `src/features.py`: account/window (1d/7d/30d) velocity & volume, structuring signals, layering/pass-through, counterparty features — all leakage-safe (past-only per transaction).
- `src/graph_features.py`: directed account graph via networkx; in/out-degree, distinct counterparties, fan-in/fan-out/cycle motif detection mapped to the 8 typologies.
- Join into one modelling table in `data/processed/`.
- `notebooks/02_feature_engineering.ipynb` as the runnable walkthrough calling into `src/`.
- **Check:** `tests/test_features.py` asserts a feature computed "as of" time T never uses rows with timestamp > T.

**Phase 4 — Modelling [DONE]**
- `src/model.py`: XGBoost/LightGBM with `scale_pos_weight`/class weights (document exact values), time-aware train/val/test split (no random k-fold).
- Optional: Isolation Forest unsupervised layer, compared against supervised scores.
- `notebooks/03_modelling.ipynb`.
- **Check:** split boundaries are strictly time-ordered (`max(train.ts) <= min(test.ts)`); model serializes to `models/` (gitignored).

**Phase 5 — Evaluation [DONE]**
- `src/evaluate.py`: precision@k (several k), recall-per-typology (all 8 patterns), FP-reduction-vs-rules-baseline-at-equal-recall (the headline number), PR-AUC + ROC-AUC, calibration curve.
- `notebooks/04_evaluation_explainability.ipynb` + `reports/results.md` with tables and plots.
- **Check:** the headline sentence ("at equal recall, X% fewer alerts than the rules baseline") is computed from code, not eyeballed, and reproducible by rerunning the script.

**Phase 6 — Explainability**
- `src/explain.py`: SHAP TreeExplainer, global feature importance, per-alert local explanation converted to a plain-English reason-code string via a small template.
- **Check:** spot-check 5 flagged alerts' reason codes against their raw feature values for correctness.

**Phase 7 — Monitoring**
- `src/monitoring.py`: PSI/CSI on feature and score distributions across time slices of the dataset.
- One plot/table in `results.md` showing score PSI over time.
- **Check:** PSI values compute without NaN/inf on real data; a deliberately shuffled "future" slice shows elevated PSI vs. a stable slice.

**Phase 8 — Demo artifact build**
- `scripts/build_demo_artifact.py`: runs the trained pipeline over a bounded, stratified sample (all positives + sampled negatives, target ~20-50k rows) and writes `app/data/demo_alerts.parquet` with transaction/account fields, model score, rank, typology label, and precomputed SHAP reason-code string per row.
- Also export the numbers needed for the Model Performance tab (PR curve points, recall-per-typology table, FP-reduction headline, PSI series) as a small JSON/parquet the app reads directly.
- **Check:** artifact size <20MB; if it ever exceeds GitHub's 100MB hard limit, fall back to Git LFS or shrink the sample.

**Phase 9 — Streamlit app**
- `app/streamlit_app.py` with tabs: **Alert Queue** (sortable/filterable, top-k slider), **Alert Detail** (SHAP bar chart + reason-code sentence), **Model Performance** (PR curve, recall-per-typology, FP-reduction headline callout, PSI-over-time), **About** (framing, links to repo/results.md).
- Slim `app/requirements.txt` (streamlit, pandas, pyarrow, plotly/matplotlib; drop shap since values are precomputed).
- **Check:** `streamlit run app/streamlit_app.py` locally, click through all four tabs, no exceptions, fast load.

**Phase 10 — Deploy**
- Push repo to GitHub (public).
- Connect Streamlit Community Cloud to the repo, entrypoint `app/streamlit_app.py`, deps from `app/requirements.txt`.
- Deploy, get the public `*.streamlit.app` URL.
- **Check:** open the live URL from a fresh/incognito session, click through every tab, confirm it matches local behavior. Add the link + a screenshot/GIF to the README.

**Phase 11 — README & portfolio polish**
- README opens with the false-positive-reduction framing and headline result, then: live demo link (prominent, near the top), business problem, why network features matter, evaluation-choice rationale (why not accuracy), SHAP reason codes, drift monitoring, how to reproduce locally, simple architecture diagram, honest caveats (synthetic data, demo app uses a precomputed sample not live inference).
- Business problem section should fold in the batch-vs-real-time architecture reasoning already written up in `reports/challenges.md`'s "Architecture — why this is a batch pipeline, not a real-time scoring service" entry: sanctions screening is real-time/blocking (a lookup problem), AML typology detection is batch/periodic by nature (patterns only visible across a rolling window of history, matching the 30-60 day SAR filing timeline) — this is *why* the whole system is architected as offline batch scoring feeding an analyst queue, not a simplification being apologized for.
- MIT license, clean commit history, CI badge in README.
- **Check:** read the README fresh, top to bottom; every numeric claim traces to `results.md` or the notebooks.

**Phase 12 (optional stretch, only after core is solid)**
- Isolation Forest agreement/disagreement analysis if not already done in Phase 4.
- Elliptic dataset / GNN — explicitly out of scope per the brief unless requested later.

## Verification approach

- Unit tests (`pytest tests/`) for schema validation, leakage-safety of rolling features, and rules-baseline logic — run in CI on every push.
- Each notebook is a thin caller into `src/`, so re-running notebooks top-to-bottom is itself an integration check.
- `reports/results.md` numbers must be regenerable by rerunning `src/evaluate.py` against the saved model — no hand-typed metrics.
- Final end-to-end check before calling it done: fresh clone of the repo, follow the README's own "how to run" steps for the core pipeline, and separately open the live Streamlit URL cold and exercise all four tabs.

## Suggested order of work across sessions

Phase 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11, each phase committed separately with a clear commit message, so progress is visible and any phase can be resumed independently in a future session.

## Future work: Project 2 (separate repo, after this project ships)

Decided 2026-07-19: this alert-triage project ships first and completely (through Phase 11, live demo deployed) before any work starts on a second, complementary CV project. Do not pull work forward from the item below into this repo.

**Project 2 concept — mule-account / network fraud detection.** Sourced from a real interview problem statement (not this brief): a bank's transaction-level fraud models score mule-account activity as clean because each individual transaction looks legitimate; the signal is in account-to-account structure (mule rings), not any single transaction. Given: 2 years of transactions (sender, receiver, amount, timestamp, channel), account metadata (KYC date, device IDs, login patterns), and 3,000 confirmed mule accounts out of 4M customers.

Why it's a good second project, and why it's a *different* problem from this one (not a harder version of it):
- **PU-learning framing, not standard supervised.** The 3,000 mules are confirmed positives; the remaining ~4M are unlabeled, not confirmed-clean — regulators didn't clear them, they just weren't caught. Treating the rest as negatives is the naive mistake to explicitly avoid and call out.
- **Structural/graph-native, not per-transaction.** Requires community detection (mule rings sharing device IDs/login fingerprints), network centrality, entity resolution across accounts, graph embeddings (Node2Vec) or label propagation — a distinct skillset from this project's window-based feature engineering.
- **No public dataset matches it.** Will require building a synthetic generator: extend the IBM AML transaction graph (or a fresh agent-based simulation) with synthetic device/login/KYC layers and deliberately injected mule-ring communities as ground truth. The simulator design is itself a talking point (demonstrates understanding of mule-ring structure) but must be disclosed honestly in that project's README as simulated data built to match the interview scenario, not real bank data.

Two projects covering different AML disciplines (transaction-level alert triage here vs. network/entity-level mule detection there) is the intended CV story — start scoping Project 2 only once this repo is fully deployed.
