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
- **Phase 6 — DONE.** `src/explain.py`: SHAP TreeExplainer over a bounded sample (all test positives + the Phase 5 equal-recall alert queue + 20,000 seeded background negatives — `select_explanation_sample`), global importance as mean |SHAP|, and per-alert plain-English reason codes built from the top *positive* contributors rendered against each row's own raw feature values (`describe_feature` → `build_reason_code`). Run end-to-end in `notebooks/04_evaluation_explainability.ipynb` (sections 7.x). Full numbers in `reports/results.md`, three writeups in `reports/challenges.md`, backing scripts in `investigations/phase6_explainability/`. Three things worth carrying forward:
  - **The headline Phase 6 finding is a negative one, and it's reported as such.** Phase 4's `payment_format_ACH` dominance wrecks the naive explanation: across the 10,011-alert queue it is the single largest positive contributor on 9,997 alerts (99.86%), with only 4 distinct features ever leading the sentence. A lead clause identical on ~99.9% of the queue carries no triage information however faithful it is. Response: `build_reason_code` takes `exclude_features`, and a second **behavioural** variant is produced with the payment-format one-hots excluded from the *sentence only* (never the model, so no metric changes) — that variant spreads across 11 distinct leading features, led by `receiver_in_7d_distinct_counterparties` (41.0%), `amount_paid_usd` (27.8%) and `sender_graph_fan_in_score` (15.5%). Both variants ship; the gap is documented, not smoothed over.
  - **`shap.TreeExplainer.expected_value` is not safe to read.** On shap 0.45.1 + xgboost 2.0.3 it returns `logit(base_score)` until the first `shap_values()` call, then is silently replaced by a different, correct scalar — a constant log-odds offset that makes reconstructed margins wrong while looking entirely plausible. `shap_base_value` derives the intercept as `margin - shap_values.sum(axis=1)` instead and asserts it's constant across rows. Contributions themselves are exact (bit-identical to XGBoost's native `pred_contribs`), so no reported number was affected — this is a trap defused before Phase 8/9 builds a waterfall on top of it. Don't "simplify" this back to `expected_value`.
  - **Groundedness of the generated text is an automated assertion**, because the real risk with a reason code is a fluent sentence quoting the wrong figure, which no eyeball check reliably catches. `describe_feature` is a pure, separately-tested mapping, and `investigations/phase6_explainability/02_reason_code_spot_check_vs_raw_values.py` parses the number back out of each clause and asserts it round-trips to that row's modelling-table value (PLAN.md's Phase 6 check, executed rather than eyeballed).
- **Phase 7 — DONE.** `src/monitoring.py`: PSI (score) and CSI (per-feature) against the **training split** as a fixed reference — never slice-to-neighbouring-slice, which hides gradual drift — plus `psi_table` per-bin attribution, `classify_psi` bands, and `psi_over_time`. Run via `python -m src.monitoring`, which also writes `reports/figures/07_score_psi_over_time.png`. Full numbers in `reports/results.md`, four writeups in `reports/challenges.md`, backing scripts in `investigations/phase7_monitoring/`. The phase's headline is that **most of what the monitor reports is not drift in the data**, and it took real work to establish that rather than reporting the raw table:
  - **The largest drift signal is self-inflicted.** Daily score PSI is flat (~0.07) for three days then steps 4x to 0.2845 on 2022-09-09 with no ramp. Cause: `graph_features.lookback_days` is 7 and the dataset starts 09-01, so 09-09 is the first day the trailing graph window evicts anything — and it evicts 09-01, HI-Small's largest day (1,114,921 rows, 22.0%). 8 of the top 10 CSI drivers across the step are `*_graph_*`. A production monitor would have paged someone for a data incident that doesn't exist. Generalisable lesson: **a drift monitor watches the features, not the world**, so anything on a schedule inside the feature pipeline is indistinguishable from real drift.
  - **Most feature CSI is rolling-window warm-up, not drift.** 23 of 54 features are "significant" (led by `receiver_in_30d_count` at 3.08), but the 30-day window is longer than the 17-18 day dataset so those counts can only accumulate. The discriminating check: non-windowed features don't move (7 stable / 0 moderate / 1 significant; `amount_paid_usd` at CSI 0.0669) while windowed ones do (14 / 10 / 22).
  - **A defect in this module, found and fixed:** quantile binning a binary feature collapses to one bin, so the first version reported CSI **0.0000** for `payment_format_Reinvestment` in the exact comparison where it went from 15.8% of transactions to zero — and 0.0000 for `payment_format_ACH`, the model's top feature. Fixed with `category_share_psi` for low-cardinality features (`method` column records which path each took); 20 → 23 significant. It then surfaced a real finding: all 481,056 Reinvestment transactions fall on 2022-09-01 alone.
  - **The one unambiguous population change is the one the monitor misses.** From 09-11 daily volume collapses (654,467 → 11 rows), laundering rate goes to 59.12%, format to 100% ACH; aggregated PSI is 10.40 but every day is under `min_slice_size` and correctly reported `insufficient_data`. The answer isn't a lower floor (PSI on 46 rows is noise) but a **volume monitor** alongside. Because that tail holds 36.4% of test positives in 0.109% of its rows, the Phase 5 headline was re-derived without it: **96.2% vs. 96.6%** — the headline does not depend on the tail.
- **Phase 8 — DONE.** `scripts/build_demo_artifact.py` writes both artifacts the deployed app reads: `app/data/demo_alerts.parquet` (**15.3 MB**, under the 20MB target — 30,000 rows x 126 columns) and `app/data/demo_metrics.json` (36 KB). One offline run reproduces every headline number from the pipeline rather than copying it (96.6% / 96.2% ex-tail / PR-AUC 0.3967 / precision@100 92%), so the artifact can't drift from `results.md` without a test failing. Full numbers in `reports/results.md`, four writeups in `reports/challenges.md`, backing script in `investigations/phase8_demo_artifact/`. Three things worth carrying into Phase 9:
  - **The artifact is a sample; its ranks are not.** Scores, ranks and the equal-recall queue are computed over the complete 1,015,669-row test split *before* sampling, so rank 1 means "highest-scored transaction in the test split", not "best survivor of sampling". The 10,011-alert queue ships whole and unsampled (it *is* the headline result); only negatives outside it are sampled, stratified across score deciles. Both properties are asserted in CI against the committed artifact — the queue must be exactly ranks 1..N with no holes, and score must be monotone in rank. **Phase 9 must not re-rank the artifact**; sort by the shipped `rank`/`model_score` and treat counts below the queue as sample counts, not population counts.
  - **The Phase 7 tail is deliberately over-represented and flagged, not corrected** — 3.097% of the artifact vs. 0.109% of the test split, because keeping every positive necessarily drags it in. Every row carries `is_tail_population`, both shares are in the metrics JSON, and the tail-excluded headline (96.2%) ships alongside the headline. Phase 9 should expose the flag as a filter rather than silently including or excluding it.
  - **The waterfall is precomputed and verified.** All 54 raw feature values (`feat_*`) and their 54 SHAP contributions (`shap_*`) ship per row, so the app draws a real waterfall with no `shap` dependency. Base value 0.123039 (from `explain.shap_base_value`, never `TreeExplainer.expected_value`) plus the shipped contributions reconstruct the shipped score to max abs error 5.2e-07 — asserted in CI.
- **Phase 9 — DONE.** The Analyst Alert-Triage Dashboard: `app/streamlit_app.py` (UI) plus `app/artifact.py` (pure data layer). Four tabs — Alert Queue, Alert Detail, Model Performance, About — reading only the Phase 8 artifacts, computing nothing about the model. Full detail in `reports/results.md`, two writeups in `reports/challenges.md`. **147 tests** (123 → 147; 16 on the data layer, 8 end-to-end). Three things worth carrying into Phase 10:
  - **Deviation from this doc's repo layout: the app is two modules, not one.** A Streamlit script executes top-to-bottom on import, so anything living in `streamlit_app.py` cannot be imported by a test without starting the UI. All filtering/waterfall/loading logic moved to `app/artifact.py`, which is unit-tested directly; `streamlit_app.py` is layout and charts only. `app/requirements.txt` also now declares `numpy` explicitly (imported directly by `artifact.py`, previously relied on as a pandas transitive).
  - **PLAN.md's Phase 9 check is executed, not remembered.** `tests/test_app_smoke.py` runs the real script headless through `streamlit.testing.v1.AppTest` and asserts on rendered elements, so "click through all four tabs, no exceptions" is enforced in CI. It found two defects that would otherwise have reached the public URL: `st.slider` raising whenever a filter narrowed the view to ≤10 rows, and `from app import artifact` failing under the real `streamlit run` import path.
  - **The `sys.path` fix at the top of `streamlit_app.py` is load-bearing for Phase 10** — `streamlit run app/streamlit_app.py` puts `app/` on the path, not the repo root, and both pytest and AppTest hide this by running with the repo root already there. Don't "tidy" that insert away.
- **Phase 10 onward — NOT STARTED.** Next up: connect the repo to Streamlit Community Cloud and get the public URL.
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

**Phase 6 — Explainability [DONE]**
- `src/explain.py`: SHAP TreeExplainer, global feature importance, per-alert local explanation converted to a plain-English reason-code string via a small template.
- **Check:** spot-check 5 flagged alerts' reason codes against their raw feature values for correctness. Done, and made an assertion rather than an eyeball check — see `investigations/phase6_explainability/02_reason_code_spot_check_vs_raw_values.py` and the Progress notes above.

**Phase 7 — Monitoring [DONE]**
- `src/monitoring.py`: PSI/CSI on feature and score distributions across time slices of the dataset.
- One plot/table in `results.md` showing score PSI over time. Done — `reports/figures/07_score_psi_over_time.png`, written by `python -m src.monitoring`.
- **Check:** PSI values compute without NaN/inf on real data; a deliberately shuffled "future" slice shows elevated PSI vs. a stable slice. Done, with one correction to the check itself: **a shuffled slice is a valid negative control, not a positive one** — a row shuffle preserves the marginal distribution exactly, so its PSI is ~0 by construction and it would have passed no matter how broken the metric was. It's kept as the negative control (`test_shuffled_slice_is_a_negative_control_not_a_positive_one`) and a genuinely perturbed slice is used as the positive one. The NaN/inf half did fire on real data and is handled by the epsilon floor.

**Phase 8 — Demo artifact build [DONE]**
- `scripts/build_demo_artifact.py`: runs the trained pipeline over a bounded, stratified sample (all positives + sampled negatives, target ~20-50k rows) and writes `app/data/demo_alerts.parquet` with transaction/account fields, model score, rank, typology label, and precomputed SHAP reason-code string per row. Done, at 30,000 rows — and carrying more than originally specified: **both** reason-code variants (Phase 6's faithful and behavioural), the full `feat_*`/`shap_*` matrix so the app can draw a real waterfall, and an `is_tail_population` flag for the Phase 7 tail.
- Also export the numbers needed for the Model Performance tab (PR curve points, recall-per-typology table, FP-reduction headline, PSI series) as a small JSON/parquet the app reads directly. Done — `app/data/demo_metrics.json`, 36 KB, which also carries the tail-excluded headline, calibration, global SHAP importance and the SHAP base value.
- **Check:** artifact size <20MB; if it ever exceeds GitHub's 100MB hard limit, fall back to Git LFS or shrink the sample. **Actual: 15.3 MB**, no fallback needed. The check was extended beyond size, since size was never the way this artifact would go wrong: CI now also asserts against the committed file that ranks are true ranks over the full test split, that the alert queue ships with no holes, that the shipped SHAP columns reconstruct the shipped scores, and that the headline in the metrics JSON still matches `results.md`.

**Phase 9 — Streamlit app [DONE]**
- `app/streamlit_app.py` with tabs: **Alert Queue** (sortable/filterable, top-k slider), **Alert Detail** (SHAP bar chart + reason-code sentence), **Model Performance** (PR curve, recall-per-typology, FP-reduction headline callout, PSI-over-time), **About** (framing, links to repo/results.md). Done, with the logic split into `app/artifact.py` so it can be tested without starting the UI (see the Progress note above), and the Alert Detail tab drawing a full SHAP **waterfall** rather than a bar chart, since Phase 8 ships every contribution per row.
- Slim `app/requirements.txt` (streamlit, pandas, pyarrow, plotly/matplotlib; drop shap since values are precomputed). Done — streamlit/pandas/pyarrow/plotly/numpy, pinned to the same versions as the root `requirements.txt` so what runs locally is what deploys. No shap, xgboost or scikit-learn.
- **Check:** `streamlit run app/streamlit_app.py` locally, click through all four tabs, no exceptions, fast load. Done, and automated: `tests/test_app_smoke.py` executes the app headless via `streamlit.testing.v1.AppTest` and asserts across all four tabs plus the empty/near-empty filter edge cases. The server was also started for real (`--server.headless`) and confirmed serving. Load is ~2.7s cold for a 15.3MB artifact.

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
