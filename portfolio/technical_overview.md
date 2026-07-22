# AML Alert-Triage System — For Technical Readers

*A living document, updated as each build phase completes. This is a summary and
navigation aid — `PLAN.md` is the authoritative phase-by-phase build log,
`reports/results.md` is the source of truth for metrics, and `reports/challenges.md`
has the full engineering post-mortems referenced here. Nothing below should be taken as
more precise than those source docs; if they ever disagree, the source docs win.*

## What this is (and isn't)

An **alert-triage / re-ranking system**, not a "detect money laundering" classifier.
Given the alerts a naive rules engine raises, the model re-ranks and filters them so a
compliance analyst reviews fewer false positives at equal-or-better recall of known
laundering typologies. The evaluation is framed accordingly: precision@k,
recall-per-typology, and false-positive-reduction-at-equal-recall-vs-baseline are the
headline metrics — not accuracy, which is meaningless at this dataset's prevalence (see
below).

## Dataset

IBM's "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models"
(AMLworld generator; [arXiv:2306.16424](https://arxiv.org/abs/2306.16424), NeurIPS 2023
Datasets & Benchmarks), **HI-Small** split specifically, via Kaggle
(`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`). Confirmed properties
(measured, not assumed — see `notebooks/01_eda.ipynb`):

- 5,078,345 transactions, prevalence ~0.10% (5,177 positives) — the brief's original
  ~2% estimate was wrong by ~20x; every evaluation choice downstream accounts for the
  real number.
- Date span ~17-18 days (2022-09-01 to 2022-09-18), and heavily front-loaded: the first
  2 days alone account for ~37% of all transactions. This skew is not just a data
  curiosity — it directly caused a Phase 3 performance bug (see below).
- 8 labelled laundering typologies (fan-in, fan-out, gather-scatter, scatter-gather,
  cycle, bipartite, stack, random), matched via `HI-Small_Patterns.txt`'s block format
  against shared field values (no transaction ID exists in this dataset) — this
  recovers typology labels for 62% of laundering transactions; the remainder are
  laundering but unpatterned (a data limitation, disclosed, not silently dropped).

## Architecture

- `src/` is the source of truth; `notebooks/` are thin callers into `src/` for
  narrative/EDA, not where logic lives.
- All paths, seeds, window sizes, and thresholds live in `config.yaml` — no scattered
  magic numbers.
- Deployment target is Streamlit Community Cloud (free tier). The live app **never**
  runs feature engineering, model inference, or SHAP live — it reads a precomputed,
  bundled artifact (`app/data/demo_alerts.parquet`, target <20MB) built offline by a
  script from the fully-trained pipeline. Full-dataset numbers come from running the
  real pipeline locally, never from the demo app. This is a deliberate cost/resource
  constraint (Streamlit's free tier is ~1 CPU/1GB RAM), disclosed honestly rather than
  papered over.
- Total cost target: **$0** — every tool choice (GitHub Actions CI, Streamlit Community
  Cloud, Kaggle, SHAP as a local library) has a free tier sized for this project.
- CI: ruff + pytest on every push (`.github/workflows/ci.yml`), branch-protected `main`
  (PR + green CI required, no direct pushes including from admins).

## Pipeline stages completed so far

### Phase 1 — Data loading & validation (`src/data_loader.py`)
Explicit schema validation (fails loudly on missing columns, doesn't silently coerce),
timestamp parsing, and a unified `{bank_id}_{account_number}` account key so the
transactions and accounts tables join cleanly. Handles two dataset-specific quirks:
`Trans.csv`'s duplicate `Account` column header (pandas auto-renames the second to
`Account.1`), and `Patterns.txt` not being a CSV at all (custom block parser).

### Phase 2 — Rules baseline (`src/rules_baseline.py`)
Three rules (large-amount, structuring, rapid pass-through), all operating on a
USD-normalized amount — a static FX table converts the dataset's 15 payment currencies
(spanning USD to Bitcoin, wildly different numeric scales) to a common unit before
thresholding, checked empirically before being built, not discovered as a bug after.
**Result: 36.3% alert rate, 0.17% precision, 60.6% recall** on the full dataset — the
number every later stage must beat, per `reports/results.md`.

### Phase 3 — Feature engineering (`src/features.py`, `src/graph_features.py`)
- **Account/window features:** rolling velocity, volume, distinct-counterparty, and
  structuring-score features across 1/7/30-day windows, all leakage-safe by
  construction (a feature at transaction time T only ever sees rows with
  timestamp <= T — asserted directly in `tests/test_features.py`, not just assumed).
  Count/sum features are vectorized via a `merge_asof` cumulative-sum trick (no
  per-account Python loop); distinct-counterparty features use a genuine two-pointer
  sliding window per account, since distinct-count isn't a simple prefix-sum
  difference.
- **Graph features:** a directed account graph (networkx) rebuilt per calendar day from
  a bounded trailing lookback window (not unbounded history — see below), giving
  in/out-degree, fan-in/fan-out score, and short-cycle membership.
- **Joined** via `build_modelling_table` into one 52-column table, run end-to-end
  against the full dataset in `notebooks/02_feature_engineering.ipynb`: 5,078,345 rows,
  ~1,532s (~25.5 min), no NaNs, saved to `data/processed/features.parquet` (gitignored,
  regenerable).
- **Two real performance bugs were found and fixed here, not just planned around** —
  full root-cause writeups in `reports/challenges.md`:
  1. The distinct-counterparty rolling window's first implementation used
     `DataFrame.groupby` per account and was 30-40x slower than necessary (most
     accounts have only 1-2 transactions in an 18-day window, so per-group object
     construction dominated the actual O(1)-per-row work). Fixed by sorting once and
     slicing raw numpy arrays at account-group boundaries instead.
  2. The graph features were originally "all history to date" per day, but the
     dataset's front-loaded volume meant that never stopped growing across the 18-day
     span (521s to process just the first 9 of 18 days on a still-worsening curve).
     Fixed by bounding each day's graph to a trailing 7-day lookback with **incremental
     edge pruning** (add each edge once, remove it once when it ages out of the
     window) — a naive "rebuild the graph from scratch each day" version of the same
     bound was tried first and was *slower* than the unbounded version, due to
     redundant re-insertion of overlapping-window edges.

### Phase 4 — Modelling (`src/model.py`)
- **Split:** strictly time-ordered 60/20/20 (`time_ordered_split`) — row-position split
  on time-sorted data, not a fixed calendar cutoff (HI-Small's daily volume is too
  uneven for that) and never a random shuffle. Surfaces a real distribution shift
  honestly rather than hiding it: train/val/test prevalence is 0.075%/0.107%/0.177%
  respectively, since the dataset's low-volume tail days carry disproportionately more
  laundering activity.
- **Imbalance handling:** `scale_pos_weight` (negatives/positives = **1,324.94**),
  computed from the training split only — never val/test, which would leak
  split-specific class balance into training.
- **Model:** XGBoost, `tree_method="hist"`, early-stopped on validation PR-AUC
  (`eval_metric="aucpr"` — the right curve under this prevalence, not accuracy or plain
  ROC-AUC). Reasonable documented defaults, not a grid/random search (disclosed as an
  explicit scope boundary, not silently skipped).
- **Unsupervised layer:** Isolation Forest (`contamination="auto"`, deliberately not
  set to the true positive rate, which would smuggle label information into an
  "unsupervised" model) — see the two investigated findings below.
- **Sanity-check results** (not the Phase 5 headline): test PR-AUC 0.3967, ROC-AUC
  0.9828, precision@100 92%. Full table and caveats in `reports/results.md`.
- **Two findings investigated in depth, not just reported** — full root-cause writeups
  in `reports/challenges.md`:
  1. `payment_format_ACH` dominates feature importance (86.6% of all laundering
     transactions use ACH format, ~43x the dataset's overall ACH-specific rate vs.
     baseline prevalence). An ablation — retraining with every `payment_format_*`
     column dropped — shows PR-AUC falling from 0.3967 to 0.0705, confirming this one
     categorical correlation carries a large share of the headline numbers (real
     behavior, synthetic-generator artifact, or both — undetermined from the data
     alone, disclosed as a caveat rather than hidden). The features that rise to the
     top without it — `receiver_in_30d_distinct_counterparties`,
     `sender_graph_in_cycle`, `sender_out_7d_distinct_counterparties` — are exactly the
     Phase 3 account/window and graph features, which is the more important result for
     this project's actual thesis.
  2. The Isolation Forest layer has **0%** top-1000 overlap with XGBoost and catches
     only 1 of 1,797 test-split laundering cases. Diagnosed (not just reported): its
     top-1000 sit at the extreme tail of raw volume features (mean
     `sender_out_30d_count` of 149,867 vs. an overall mean of 7,890, near the dataset's
     actual maximum) — it's rediscovering high-throughput hub accounts, not laundering
     behavior, because isolation-based detection can't distinguish "extreme but
     legitimate" from "extreme and suspicious" on heavily right-skewed raw features.
     Log-transforming heavy-tailed features before fitting is noted as future work,
     not implemented, to keep this phase's scope disciplined.
- **Serialization:** XGBoost's native `save_model` (version-portable, not pickle) plus
  `joblib` for the Isolation Forest, with a `model_metadata.json` sidecar capturing
  split sizes, `scale_pos_weight`, and the sanity-check metrics — all to `models/`
  (gitignored, regenerable by rerunning `notebooks/03_modelling.ipynb`).

## Evaluation philosophy (Phase 5, not yet built)

Accuracy is explicitly rejected as a metric — at 0.10% prevalence a model that flags
nothing is already 99.9% "accurate." Planned metrics: precision@k (multiple k),
recall-per-typology (all 8 patterns), PR-AUC, calibration, and the headline
false-positive-reduction-at-equal-recall-vs-rules-baseline number — computed from code
(`src/evaluate.py`) against the saved model, never hand-typed into `results.md` or the
README.

## Tech stack

Python 3.12 · pandas / numpy · scikit-learn · xgboost / lightgbm · networkx · shap ·
matplotlib / seaborn · pyyaml · jupyter · streamlit · pyarrow · pytest · ruff.
No GPU required anywhere in this project.

## Current status

Phases 0-4 of 11 complete (Phase 4 in PR review as of this writing — see `PLAN.md`'s
Progress section for the authoritative per-phase state). Next: Phase 5
(`src/evaluate.py` — precision@k, recall-per-typology, PR-AUC/calibration as reported
metrics, and the headline false-positive-reduction-vs-rules-baseline-at-equal-recall
number, computed against the model saved in Phase 4).
