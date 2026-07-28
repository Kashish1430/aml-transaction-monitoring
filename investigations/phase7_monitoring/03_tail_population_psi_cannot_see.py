"""The drift a daily PSI monitor structurally cannot see, and whether it moves the headline.

Supports the "the one real population change is the one the distribution monitor misses"
entry in reports/challenges.md's Phase 7 section, and the tail-robustness numbers in
reports/results.md's Phase 7 section.

HI-Small's last eight days are not the same dataset as its first ten. From 2022-09-11
onward, daily volume collapses from hundreds of thousands of transactions to double
digits, the laundering rate goes from ~0.1% to ~59%, and the payment format becomes 100%
ACH. That is a real population change of a magnitude no monitor should miss.

The Phase 7 daily monitor misses it anyway — not through a bug, but structurally. Every
one of those days is smaller than `monitoring.min_slice_size` (1,000), so each is
correctly reported as `insufficient_data` rather than being scored on 46 rows of noise.
The floor is right and the outcome is still a blind spot. This script quantifies both
halves of that, and then asks the question that actually matters for the project: since
this tail sits inside the test split, does the project's headline result depend on it?

Three things checked:
1. what the tail looks like vs. the body, and why each day is unscoreable,
2. what the PSI is once the tail is aggregated into one slice big enough to score,
3. whether the Phase 5 headline (96.6% fewer alerts at equal recall) survives removing
   the tail entirely — recomputed with the same `evaluate.py` machinery, not asserted.

Run from the repo root (needs data/processed/features.parquet, models/, data/raw/):
    venve/python.exe investigations/phase7_monitoring/03_tail_population_psi_cannot_see.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data_loader import load_config, load_transactions
from src.evaluate import alerts_for_recall, false_positive_reduction
from src.model import load_xgboost_model, prepare_feature_matrix, time_ordered_split
from src.monitoring import classify_psi, population_stability_index, psi_over_time
from src.rules_baseline import apply_rules_baseline

TAIL_START = pd.Timestamp("2022-09-11")


def main() -> None:
    config = load_config()
    split_cfg = config["split"]
    mon_cfg = config["monitoring"]
    n_bins = mon_cfg["psi_bins"]
    epsilon = float(mon_cfg["psi_epsilon"])

    print("Loading modelling table and model ...")
    df = pd.read_parquet(config["paths"]["modelling_table"])
    train_df, val_df, test_df = time_ordered_split(
        df, split_cfg["train_frac"], split_cfg["val_frac"]
    )
    del df

    model = load_xgboost_model(config["paths"]["models_dir"])
    x_train, _ = prepare_feature_matrix(train_df)
    train_scores = model.predict_proba(x_train)[:, 1]
    x_test, y_test_series = prepare_feature_matrix(test_df)
    test_scores = model.predict_proba(x_test)[:, 1]
    y_test = y_test_series.to_numpy()

    tail_mask = (test_df["timestamp"] >= TAIL_START).to_numpy()

    # --- 1. What the tail is -----------------------------------------------------------
    print(f"\n=== 1. The {TAIL_START.date()}+ tail vs. the rest of the test split ===")
    print(
        f"  test split : {len(test_df):>9,} rows, {int(y_test.sum()):>5,} positives, "
        f"prevalence {y_test.mean():.4%}\n"
        f"  body       : {int((~tail_mask).sum()):>9,} rows, "
        f"{int(y_test[~tail_mask].sum()):>5,} positives, prevalence {y_test[~tail_mask].mean():.4%}\n"
        f"  tail       : {int(tail_mask.sum()):>9,} rows, "
        f"{int(y_test[tail_mask].sum()):>5,} positives, prevalence {y_test[tail_mask].mean():.4%}"
    )
    print(
        f"\n  {y_test[tail_mask].sum() / y_test.sum():.1%} of all test-split positives live "
        f"in {tail_mask.sum() / len(test_df):.3%} of its rows."
    )

    print("\n  Per-day size of the tail (vs. min_slice_size = "
          f"{mon_cfg['min_slice_size']:,}):")
    tail_days = test_df.loc[tail_mask, "timestamp"].dt.date.value_counts().sort_index()
    for day, n in tail_days.items():
        print(f"    {day}: {n:>6,} rows   {'SCOREABLE' if n >= mon_cfg['min_slice_size'] else 'too thin to score'}")

    # --- 2. The PSI the daily monitor never gets to report ------------------------------
    print("\n=== 2. What the monitor reports, and what it would report if it could ===")
    daily = psi_over_time(
        train_scores,
        test_scores,
        test_df["timestamp"].reset_index(drop=True),
        freq=mon_cfg["time_slice_freq"],
        n_bins=n_bins,
        epsilon=epsilon,
        min_slice_size=mon_cfg["min_slice_size"],
        thresholds=mon_cfg["psi_thresholds"],
    )
    print(daily.to_string(index=False))

    psi_tail = population_stability_index(train_scores, test_scores[tail_mask], n_bins, epsilon)
    psi_body = population_stability_index(train_scores, test_scores[~tail_mask], n_bins, epsilon)
    print(
        f"\n  aggregated tail ({tail_mask.sum():,} rows): PSI {psi_tail:.4f} "
        f"({classify_psi(psi_tail, mon_cfg['psi_thresholds'])})\n"
        f"  body            ({(~tail_mask).sum():,} rows): PSI {psi_body:.4f} "
        f"({classify_psi(psi_body, mon_cfg['psi_thresholds'])})"
    )
    print(
        "\n  A PSI of ~10 is off any conventional scale - and every daily slice of it is\n"
        "  correctly reported as insufficient_data. The lesson is not 'lower the floor':\n"
        "  PSI on 46 rows would be noise. It is that a distribution monitor needs a\n"
        "  VOLUME monitor beside it. Daily row count falling 654,467 -> 11 is trivially\n"
        "  detectable and needs no binning at all."
    )

    # --- 3. Does the headline depend on the tail? --------------------------------------
    print("\n=== 3. Robustness: does the Phase 5 headline survive dropping the tail? ===")
    raw = load_transactions(f"{config['paths']['raw_dir']}/HI-Small_Trans.csv")
    baseline_test = apply_rules_baseline(raw, config).loc[test_df.index]
    b_alert = baseline_test["alert"].to_numpy()
    b_laundering = baseline_test["is_laundering"].astype(bool).to_numpy()

    for label, mask in (
        ("FULL test", np.ones(len(y_test), dtype=bool)),
        ("BODY only", ~tail_mask),
        ("TAIL only", tail_mask),
    ):
        n_baseline_alerts = int(b_alert[mask].sum())
        baseline_recall = float(
            (b_alert[mask] & b_laundering[mask]).sum() / b_laundering[mask].sum()
        )
        alert_idx = alerts_for_recall(y_test[mask], test_scores[mask], baseline_recall)
        reduction = false_positive_reduction(len(alert_idx), n_baseline_alerts)
        print(
            f"  {label}: baseline {n_baseline_alerts:>7,} alerts @ {baseline_recall:.1%} recall "
            f"-> model {len(alert_idx):>6,} alerts = {reduction:.1%} reduction"
        )

    print(
        "\n  The headline is not an artifact of the tail: removing it entirely moves the\n"
        "  reduction by well under a percentage point. The tail-only number is low for a\n"
        "  reason that is not a model failure - at ~59% prevalence there are barely any\n"
        "  false positives left for the model to remove, so the rules baseline is already\n"
        "  close to right and there is little headroom."
    )


if __name__ == "__main__":
    main()
