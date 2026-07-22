# Investigations

Standalone, runnable scripts behind the findings written up in `reports/challenges.md`
and cited in `reports/results.md`. Every non-obvious number in this project's
challenges/results docs was surfaced by actually running something against the real
data, not eyeballed or estimated — this folder is where that "actually running
something" lives, so the process is reproducible and inspectable, not just asserted.

**These are diagnostic scripts, not maintained pipeline code.** `src/` is still the
single source of truth for the pipeline itself (per `CLAUDE.md`); nothing here is
imported by `src/`, `notebooks/`, or `tests/`. They're excluded from `ruff` (see
`pyproject.toml`) since they're one-off investigations, not production code held to
the same style bar — but every one of them runs and every number it prints is real,
checked at the time this was written.

Each script is self-contained, prints its findings to stdout, and has a module
docstring naming the specific `reports/challenges.md` entry it supports. Run from the
repo root, e.g.:

```
venve/python.exe investigations/phase3_feature_engineering/01_groupby_vs_vectorized_distinct_counterparties.py
```

## Index

**Phase 3 — feature engineering** (see `reports/challenges.md`'s Phase 3 section)
- `01_groupby_vs_vectorized_distinct_counterparties.py` — reproduces the 30-40x
  speedup from replacing `DataFrame.groupby` with sorted numpy slicing in the
  distinct-counterparty rolling window.
- `02_graph_growth_unbounded_vs_bounded_lookback.py` — reproduces why an unbounded
  "all history to date" account graph never stopped growing across HI-Small's 18 days,
  and confirms the bounded-lookback-with-incremental-pruning fix plateaus instead.
- `03_single_day_cycle_detection_is_not_the_bottleneck.py` — isolates that
  `nx.simple_cycles` itself is fast even on a ~680K-edge single-day graph; the growth
  problem in script 02 was graph *accumulation*, not cycle enumeration.

**Phase 4 — modelling** (see `reports/challenges.md`'s Phase 4 section)
- `01_payment_format_ach_correlation.py` — the 86.6%-of-laundering-uses-ACH finding
  that explained why `payment_format_ACH` dominated feature importance.
- `02_payment_format_ablation.py` — retrains without `payment_format` to quantify how
  much of the model's performance depends on that one correlation.
- `03_isolation_forest_hub_account_diagnosis.py` — inspects what Isolation Forest's
  top-1000 "anomalous" test transactions actually look like, explaining the 0%
  overlap with XGBoost.

**Phase 5 — evaluation** (see `reports/challenges.md`'s Phase 5 section)
- `01_pattern_type_join_verification.py` — verifies `join_pattern_types` against the
  previously-documented 62%/3,209-of-5,177 typology match rate before trusting it.
- `02_score_calibration_distortion_check.py` — quantifies how far the model's raw
  scores drift from calibrated probabilities, and ties it to `scale_pos_weight`.
- `03_transaction_vs_account_level_alerts.py` — checks how many distinct accounts sit
  behind the model's transaction-level alert set, since the model scores individual
  transactions using account-level features, not accounts directly — see the
  "transaction-level vs. account-level" entry in `reports/challenges.md`.
