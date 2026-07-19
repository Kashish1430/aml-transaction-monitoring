# AML Transaction Monitoring — Project Brief

## One-line pitch
An anti-money-laundering (AML) transaction-monitoring system that triages the alerts a rules-based engine would generate, cutting false positives while preserving recall of known laundering typologies — with per-alert explanations suitable for a compliance analyst.

## Why this framing matters (read before building)
Do not build this as "a classifier that detects money laundering." Real production AML is rules-based and buries analysts in alerts, the overwhelming majority of which are false positives; the twin failure is false negatives, where genuine laundering goes undetected. The valuable, hireable framing is therefore alert triage / false-positive reduction: given the alerts a naive rules engine raises, rank and filter them so an analyst reviews far fewer cases at the same or better recall. Every design decision below serves that framing. The headline result we want to be able to state at the end is: "at equal recall of labelled laundering, this model reduces the alert volume an analyst must review by X% versus the rules baseline." Keep that sentence in mind the whole way through.

## Dataset
Use the **IBM Transactions for Anti-Money Laundering (AML)** dataset on Kaggle (publisher: ealtman2019, "ibm-transactions-for-anti-money-laundering-aml"). It is fully synthetic — a multi-agent virtual world of individuals, companies and banks where a small fraction of agents launder funds — so there are no privacy concerns, and unlike real data every transaction carries a reliable `laundering` label. It ships as two groups, HI (higher illicit ratio) and LI (lower illicit ratio), each in small/medium/large sizes. **Start with HI-Small** so everything runs on a laptop; it is heavily imbalanced (roughly 2% illicit), which is realistic and is the whole point. Alongside the transaction CSVs there is a companion file listing transactions that follow one of **eight laundering patterns** from the AMLSim simulator: fan-in, fan-out, bipartite, stack, random, cycle, scatter-gather, and gather-scatter. Preserve that pattern labelling — it lets us engineer typology-aware features and, more importantly, report recall broken down per pattern, which is a strong differentiator. If a graph/crypto extension is wanted later, the **Elliptic** Bitcoin dataset (~200k nodes, 166 features, ~2% illicit, mostly unlabelled) is the other field-standard and suits unsupervised and graph methods; treat it as optional phase-two, not core.

The transaction fields are roughly: timestamp, sender bank/account, receiver bank/account, amount paid, payment currency, amount received, receiving currency, payment format, and the `Is Laundering` label. Confirm exact column names on load rather than assuming them, and log basic dataset stats (row count, class balance, date span, number of unique accounts) as the first artifact.

## Repository scaffold
Please create a clean, reproducible layout:

```
aml-transaction-monitoring/
├── README.md                  # project story, framing, how to run, headline results
├── requirements.txt           # pinned deps
├── data/
│   ├── raw/                   # downloaded CSVs (gitignored)
│   └── processed/             # engineered feature tables (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modelling.ipynb
│   └── 04_evaluation_explainability.ipynb
├── src/
│   ├── data_loader.py         # load, validate schema, basic cleaning
│   ├── rules_baseline.py      # the naive rules engine we must beat
│   ├── features.py            # all feature engineering (account/window/graph)
│   ├── graph_features.py      # networkx motif + centrality features
│   ├── model.py               # train/predict wrappers (XGBoost/LightGBM, IsolationForest)
│   ├── evaluate.py            # precision@k, recall-per-typology, FP-reduction, PR-AUC, calibration
│   ├── explain.py             # SHAP reason codes per alert
│   └── monitoring.py          # PSI/CSI drift checks
├── reports/
│   ├── figures/
│   └── results.md             # tables + plots, final numbers
└── config.yaml                # thresholds, window sizes, paths, seeds
```

Keep everything seeded and deterministic, put paths and hyperparameters in `config.yaml`, and gitignore the data. Favour scripts in `src/` as the source of truth with notebooks calling into them, so the work is reproducible rather than notebook-only.

## Environment
Python 3.11. Core dependencies: pandas, numpy, scikit-learn, xgboost, lightgbm, networkx, shap, matplotlib, seaborn, pyyaml, and jupyter. Pin versions in `requirements.txt`. No GPU is required for the core project.

## Step 1 — Load, validate, EDA
Load HI-Small, validate the schema explicitly (fail loudly if expected columns are missing), parse the timestamp to datetime, and standardise account identifiers into a single unique key per account (bank + account number). Produce EDA that a reviewer will actually care about: class balance, amount distributions for legitimate vs laundering (log scale), transaction volume over time, the most active accounts, and the distribution of the eight laundering patterns. Save every figure to `reports/figures/`. The EDA should make the imbalance and the pattern structure visible — those two facts drive the rest of the design.

## Step 2 — Rules baseline (build this before any ML)
Implement a deliberately simple rules engine in `rules_baseline.py` that mimics what a bank might deploy: flag any transaction above a fixed large-amount threshold; flag accounts making many transactions just under a reporting threshold within a short window (classic structuring); flag rapid pass-through (funds in and out of an account within a short time window). This baseline exists to reproduce the false-positive problem: record how many alerts it raises and its recall/precision against the labels. Every later model is judged by how much it improves on this. Do not skip it — the comparison to this baseline is the project's main result.

## Step 3 — Feature engineering (the core of the project)
A single transaction is nearly uninformative; laundering lives in patterns across accounts and time. Engineer features at the account level over rolling time windows (e.g. 1-day, 7-day, 30-day):

- **Velocity and volume:** count, sum, mean, std, min, max of amounts per account per window; the account's activity relative to its own historical baseline (z-score of today vs trailing mean) to catch sudden spikes.
- **Structuring signals:** counts of transactions falling just below round/reporting thresholds; proportion of round-number amounts; number of distinct small transfers aggregating to a large sum.
- **Layering / pass-through:** ratio of inflow to outflow per window, and "funds that don't rest" — money received and sent onward within a short window; time-to-forward statistics.
- **Counterparty features:** number of distinct senders/receivers per account per window; share of activity with new (first-seen) counterparties; cross-currency and cross-bank transfer ratios.
- **Graph / network features (the standout, in `graph_features.py`):** build a directed graph with accounts as nodes and transactions as edges using networkx, then compute in-degree, out-degree, number of distinct counterparties, and detect the laundering motifs directly — fan-in (many senders → one account), fan-out (one account → many receivers), and cycles (funds returning toward their origin). These motif features map straight onto the eight labelled typologies and are the part most likely to impress. Note that graph-derived features fed into a gradient-boosting model capture most of the value without needing a graph neural network — do it this way first.

Write features to `data/processed/` as a single modelling table keyed by transaction (with the account/window aggregates joined on), and be careful to compute all rolling features using only past information relative to each transaction to avoid leakage.

## Step 4 — Modelling
Progress through three layers so there is a clear story of improvement:

1. **Rules baseline** (already built) — the number to beat.
2. **Supervised gradient boosting** — XGBoost and/or LightGBM on the engineered features. This is the workhorse and plays to strengths. Handle the extreme imbalance explicitly with `scale_pos_weight` / class weights (prefer this over naive oversampling; if resampling is used, only on the training fold, never the test fold) and document exactly what was done. Tune with a time-aware split (train on earlier transactions, validate/test on later ones) rather than random k-fold, because random splitting leaks future information in a temporal problem.
3. **Unsupervised anomaly layer (optional but realistic)** — an Isolation Forest (or autoencoder) scoring transactions with no reliance on labels, reflecting the reality that launderers invent new patterns the labels don't cover. Show where it agrees and disagrees with the supervised model.

A graph neural network (e.g. GraphSAGE) is a legitimate **phase-two stretch**, not a requirement; the graph *features* above already deliver most of the benefit in a far more interpretable, defensible form.

## Step 5 — Evaluation (evaluate like an analyst, not a Kaggler)
Accuracy is meaningless at ~2% prevalence — do not report it as a headline. In `evaluate.py` compute and tabulate:

- **Precision@k** — precision within the top-k highest-scored transactions, because an analyst can only review a fixed number of alerts per day; report at several realistic k values.
- **Recall per laundering typology** — recall broken out across the eight patterns, so it's visible which schemes the model catches and which it misses.
- **False-positive reduction versus the rules baseline at equal recall** — the money metric. Fix recall at the baseline's level, then show how many fewer alerts the model raises to achieve it. This is the headline sentence.
- **PR-AUC** (precision-recall AUC), which is the right summary curve under heavy imbalance, alongside ROC-AUC for reference.
- **Calibration** — a reliability check on predicted scores, since analysts triage by score.

Put all of this, with plots, into `reports/results.md`.

## Step 6 — Explainability (AML is regulated — this is not optional flavour)
A black box won't pass an auditor. In `explain.py`, use SHAP to generate a per-alert, plain-English reason code — e.g. "flagged: 14 sub-threshold deposits from 9 first-seen counterparties in 48h, then forwarded within 2h." Provide both global feature importance and local explanations for individual flagged transactions. Framing the model as auditable and analyst-ready is a genuine differentiator for compliance-oriented roles.

## Step 7 — Monitoring (ties in existing PSI/CSI skills)
In `monitoring.py`, add population and characteristic stability checks (PSI/CSI) comparing feature and score distributions across time periods, to demonstrate awareness that AML models drift as behaviour changes and must be monitored in production. A short section showing PSI on the score across the dataset's time span is enough to make the point.

## Deliverables and portfolio framing
The finished repo should include: a clear README that opens with the false-positive-reduction framing and the headline result, reproducible scripts, the four notebooks, saved figures, and `results.md` with the metric tables. In the README, tell the story a hiring manager wants — the business problem, why network features matter, the evaluation choices and why accuracy was rejected, the SHAP reason codes, and the drift monitoring — and be honest and specific with every number so each one is defensible under interview follow-up (how it was measured, the baseline, the caveats). A working demo or a couple of annotated example alerts beats any vanity metric.

## Suggested build order (milestones)
Start with data loading, validation and EDA; then the rules baseline and its recorded alert volume; then core account/window feature engineering; then the graph/motif features; then the XGBoost/LightGBM model with imbalance handling and a time-aware split; then the full evaluation suite with the false-positive-reduction headline; then SHAP explanations; then PSI monitoring; then write the README and results. Treat the unsupervised layer and any GNN work as clearly-scoped stretch goals only after the core pipeline produces honest, defensible numbers end to end.

## Guardrails on scope
Resist boiling the ocean. HI-Small, solid feature engineering including graph motifs, a rules baseline plus one well-tuned boosting model, an honest analyst-style evaluation, SHAP reason codes, and a PSI check constitute a complete and credible project. Everything else is optional. A tight, well-evaluated, well-explained pipeline is worth far more than a sprawling one with inflated claims.
