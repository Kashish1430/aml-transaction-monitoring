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

Phases 0-3 of 11 complete and merged (or in PR review — see `PLAN.md`'s Progress
section for the authoritative per-phase state). Next: Phase 4 (`src/model.py` —
XGBoost/LightGBM with `scale_pos_weight`, strictly time-ordered train/val/test split,
no random k-fold), then Phase 5 (evaluation against the Phase 2 baseline above).
