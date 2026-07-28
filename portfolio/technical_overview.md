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
     transactions use ACH format; ACH's laundering rate is ~7.3x the dataset's overall
     prevalence). An ablation — retraining with every `payment_format_*`
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

### Phase 5 — Evaluation (`src/evaluate.py`) — the project's headline result

Accuracy is explicitly rejected as a metric — at 0.10% prevalence a model that flags
nothing is already 99.9% "accurate." Computed instead, on the model's held-out test
split only:

**>>> HEADLINE: at equal recall (68.8%), the model raises 96.6% fewer alerts than the
rules baseline — 10,011 vs. 297,564, on the same test population. <<<**

Methodology matters here as much as the number: the rules-baseline comparison is
**not** Phase 2's full-dataset figure (36.3% alert rate, 60.6% recall — a different,
non-comparable population), and it's **not** the rules re-run on an isolated test-only
slice either (would cold-start the structuring/pass-through rules' rolling windows
right at the split boundary — the same leakage failure mode Phase 3 was built to
avoid, reappearing at a different boundary). Instead: `apply_rules_baseline` runs on
the full dataset first (keeping every rule's complete rolling-window history), then
the result is restricted to the test split's rows by index — the correct
apples-to-apples comparison. The rules baseline's test-split recall (68.8%) differs
from Phase 2's full-dataset number (60.6%) as a direct, expected result of this, and
is disclosed as such rather than reconciled away.

**Recall per typology** at the same operating point: 67.6% (`BIPARTITE`) to 89.8%
(`FAN-IN`) across all 8 patterns — no typology dramatically weaker, including patterns
without a dedicated graph feature (`BIPARTITE`, `STACK`, `RANDOM`), which still get
caught at a broadly similar rate to `FAN-IN`/`CYCLE` (which the Phase 3 graph features
specifically target).

**Supporting metrics:** PR-AUC 0.3967, ROC-AUC 0.9828 (reference only), precision@100
92.0%, precision@1000 60.6%.

**A calibration finding, diagnosed not just plotted:** the model's raw score is not a
calibrated probability — mean predicted score on the test split (7.31%) is ~41x the
actual prevalence (0.177%). This is the direct, expected consequence of
`scale_pos_weight=1324.94` (the same weighting that lets the model rank rare positives
at all inflates scores for anything positive-like). Doesn't affect any number above —
precision@k, recall-per-typology, and the headline reduction depend only on ranking,
not the literal score — but would matter if this project ever showed an analyst a
literal "X% chance" figure; post-hoc calibration (Platt/isotonic) is noted as future
work, not implemented.

A typology-label join (`data_loader.join_pattern_types`, matching `load_transactions`
and `load_patterns` on every shared field since the dataset has no transaction ID) had
to be built for the recall-per-typology table — `load_patterns`'s docstring had
pointed at "Phase 3" for this since Phase 1, but Phase 3 never actually needed or built
it. Verified against the previously-documented 62%/3,209-of-5,177 match rate before
trusting it. Full writeups of all three findings above in `reports/challenges.md`.

### Phase 6 — Explainability (`src/explain.py`)

SHAP `TreeExplainer` over the XGBoost model, exposed three ways: global feature
importance (mean |SHAP|), per-alert local contributions, and a plain-English **reason
code** per alert. AML is regulated, so this isn't optional flavour — an analyst has to
justify escalating a case and an auditor has to follow that justification afterwards.

**Scope choice.** SHAP is computed over a bounded sample, not all 1,015,669 test rows:
all test positives (at ~0.10% prevalence a uniform sample would contain almost none,
leaving global importance describing only what the model thinks about legitimate
traffic), plus the Phase 5 equal-recall alert queue, plus 20,000 seeded background
negatives that make mean |SHAP| a population statistic rather than an alerts-only one.
`TreeExplainer` is exact per row, so this is a runtime/memory bound, not an
approximation.

**Reason-code construction.** The top positive SHAP contributors are rendered against
that row's own raw feature values via a per-family template map (`describe_feature`),
e.g. `receiver_in_7d_distinct_counterparties = 11` → "the receiver received from 11
distinct counterparties in the past 7 days". Three rules, each guarding a specific
failure mode:
- **Positive contributions only.** A negative SHAP value means the feature argued
  *against* the alert; citing it as a reason would be actively misleading in a case note.
- **Binary features render by direction**, so a zero-valued one-hot carrying positive
  SHAP becomes "the payment was *not* made via X" rather than a clause implying the
  opposite.
- **Unknown feature names degrade to `name = value` instead of raising** — a degraded
  reason code is recoverable in production; an exception in the explanation layer takes
  down an otherwise healthy alert queue.

**Faithfulness is asserted, not assumed.** `shap_base_value` derives SHAP's additive
intercept as `margin - shap_values.sum(axis=1)` and raises if it isn't constant across
rows, verifying the additivity identity the explanations depend on. It deliberately does
*not* read `TreeExplainer.expected_value`, which on shap 0.45.1 + xgboost 2.0.3 returns
`logit(base_score)` until the first `shap_values()` call and is then silently replaced
with a different, correct value — a constant offset that produces plausible-looking but
wrong reconstructions. Contributions themselves were verified bit-identical to XGBoost's
native `pred_contribs`, so no reported number was ever affected; this was a latent trap
fixed before Phase 8/9 could build a waterfall on the wrong intercept.

**Groundedness is a test, not an eyeball check.** The risk with generated justifications
isn't a crash, it's a fluent sentence quoting the wrong figure. So `describe_feature` is
a pure, separately-tested value-to-words mapping, and
`investigations/phase6_explainability/02_reason_code_spot_check_vs_raw_values.py` parses
the number back out of each generated clause and asserts it round-trips to that row's
value in the modelling table (also cross-checking against the Phase 3 feature table and
the `payment_format` one-hot mapping).

**The main finding, and the design response.** Phase 4's `payment_format_ACH` dominance
shows up under SHAP too, and it degrades the explanation layer badly: across the
10,011-alert queue, `payment_format_ACH` is the single largest positive contributor on
**9,997 alerts (99.86%)**, with only 4 distinct features ever leading the sentence. A
lead clause identical on ~99.9% of the queue carries no triage information however
faithful it is. `build_reason_code` therefore takes `exclude_features`, and a second
**behavioural** variant is produced with `PAYMENT_FORMAT_FEATURES` excluded from the
sentence only — never from the model, so no metric changes. That variant spreads across
11 distinct leading features (`receiver_in_7d_distinct_counterparties` 41.0%,
`amount_paid_usd` 27.8%, `sender_graph_fan_in_score` 15.5%, `receiver_in_1d_count`
12.5%), i.e. the Phase 3 account/window and graph features. Both variants are reported,
because they answer different questions: what the model is actually doing versus what an
analyst should look at. Full numbers in `reports/results.md`; writeups of all three
findings in `reports/challenges.md`.

## Tech stack

Python 3.12 · pandas / numpy · scikit-learn · xgboost / lightgbm · networkx · shap ·
matplotlib / seaborn · pyyaml · jupyter · streamlit · pyarrow · pytest · ruff.
No GPU required anywhere in this project.

## Current status

Phases 0-6 of 11 complete — see `PLAN.md`'s Progress section for the authoritative
per-phase state. The project's headline result exists, and every alert now carries a
grounded, plain-English reason code. Next: Phase 7 (`src/monitoring.py` — PSI/CSI drift
checks on feature and score distributions across the dataset's time span), then the
precomputed demo artifact and Streamlit app (Phases 8-10).
