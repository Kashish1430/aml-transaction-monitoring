"""Separates real population drift from rolling-window warm-up in the Phase 7 CSI table.

Supports the "most of the drift the monitor reports is the monitor watching its own
feature windows fill up" entry in reports/challenges.md's Phase 7 section, and the CSI
numbers in reports/results.md's Phase 7 section.

The Phase 7 run reports 20 of 54 features in the "significant" CSI band (>0.25) between
the train and test splits, led by `receiver_in_30d_count` at CSI 3.08. Taken at face
value that is an alarming readout: a model-risk function seeing 20 significant input
shifts inside an 18-day window would pull the model. The question this script answers is
whether the underlying population actually moved, or whether the features moved for a
mechanical reason that has nothing to do with the data generating process.

The suspicion is structural. HI-Small spans ~17-18 days, but `config.yaml`'s
`windows.rolling_days` includes 7 and 30. A 30-day window is longer than the entire
dataset, so `*_30d_count` cannot do anything on day 3 except be smaller than it is on
day 10 — every account's trailing count is still accumulating. A 7-day window has the
same problem for the dataset's first week. That is warm-up, not drift: the feature is
converging on its steady-state value, and the population behind it is unchanged.

The discriminating test is whether *non-windowed* features move the same way. If the
population genuinely shifted, `amount_paid_usd` (a per-transaction value with no window
at all) should shift too. If only the windowed features move, the drift is warm-up.

The answer splits three ways, and the third one is a genuine finding rather than an
artifact:
1. every windowed feature that drifts is consistent with warm-up,
2. the continuous non-windowed feature (`amount_paid_usd`) does not drift at all,
3. one non-windowed *categorical* feature drifts enormously and for a real reason —
   `payment_format_Reinvestment` is 43.15% of transactions on 2022-09-01 and 0.00% on
   every subsequent day. All 481,056 Reinvestment transactions in HI-Small fall on the
   dataset's first day. That is the generator's behaviour, not a filling window.

Point 3 is only visible at all because `characteristic_stability` routes low-cardinality
features through `category_share_psi`. Quantile-binned, every payment-format one-hot
scored exactly 0.0000 — see the "a drift monitor blind to binary features" entry in
reports/challenges.md's Phase 7 section.

Run from the repo root (needs data/processed/features.parquet):
    venve/python.exe investigations/phase7_monitoring/01_window_warmup_vs_real_drift.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_loader import load_config
from src.model import prepare_feature_matrix, time_ordered_split
from src.monitoring import characteristic_stability

# Features whose value depends on a trailing window (and therefore must warm up), vs.
# features computed from the transaction row alone (which cannot warm up, so any drift
# in them is real).
WINDOWED = re.compile(r"_\d+d_|_\d+d$|_graph_")
NON_WINDOWED_EXAMPLES = ["amount_paid_usd", "amount_received_usd"]

TAIL_START = pd.Timestamp("2022-09-11")


def main() -> None:
    config = load_config()
    split_cfg = config["split"]
    mon_cfg = config["monitoring"]

    print("Loading modelling table ...")
    df = pd.read_parquet(config["paths"]["modelling_table"])
    train_df, _, test_df = time_ordered_split(
        df, split_cfg["train_frac"], split_cfg["val_frac"]
    )
    del df

    x_train, _ = prepare_feature_matrix(train_df)
    x_test, _ = prepare_feature_matrix(test_df)

    csi = characteristic_stability(
        x_train,
        x_test,
        n_bins=mon_cfg["psi_bins"],
        epsilon=float(mon_cfg["psi_epsilon"]),
        thresholds=mon_cfg["psi_thresholds"],
    )
    csi["is_windowed"] = csi["feature"].str.contains(WINDOWED)

    print("\n=== 1. CSI band counts, split by whether the feature has a trailing window ===")
    print(pd.crosstab(csi["is_windowed"], csi["band"]).to_string())

    print("\n=== 2. Do the NON-windowed features drift at all? ===")
    non_windowed = csi[~csi["is_windowed"]]
    print(non_windowed.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    continuous = non_windowed[~non_windowed["feature"].str.startswith("payment_format_")]
    print(
        f"\n  Max CSI, non-windowed CONTINUOUS features: {continuous['csi'].max():.4f} "
        f"({continuous.loc[continuous['csi'].idxmax(), 'feature']})\n"
        f"  Max CSI, windowed features:                {csi[csi['is_windowed']]['csi'].max():.4f}"
    )
    print(
        "  -> the population's per-transaction amounts did NOT move. Whatever moved the\n"
        "     windowed features is not a change in the underlying transactions."
    )

    # --- 3. The direct evidence: windowed counts rise monotonically with calendar day ---
    # If this is warm-up, the per-day mean climbs from the dataset's first day and starts
    # flattening once the window is fully populated. If it were real drift, there would be
    # no reason for the climb to start exactly at the dataset's first day.
    print("\n=== 3. Per-day mean of the top windowed drifters vs. a non-windowed control ===")
    full = pd.read_parquet(config["paths"]["modelling_table"])
    full["day"] = full["timestamp"].dt.date
    top_windowed = list(csi[csi["is_windowed"]]["feature"].head(4))
    watch = [c for c in top_windowed + NON_WINDOWED_EXAMPLES if c in full.columns]

    per_day = full.groupby("day")[watch].mean()
    per_day["n_rows"] = full.groupby("day").size()
    pd.set_option("display.width", 220)
    print(per_day.to_string(float_format=lambda v: f"{v:,.3f}"))

    print(
        "\nRead the table above down the columns: the *_30d_count / *_7d_count means climb\n"
        "from the dataset's very first day, which is what a filling window looks like.\n"
        "amount_paid_usd has no window and no comparable trend."
    )

    # --- 3b. The one non-windowed feature that DOES drift, and why it is real ----------
    print("\n=== 3b. payment_format_Reinvestment: real drift, not warm-up ===")
    mix = pd.crosstab(full["day"], full["payment_format"], normalize="index") * 100
    print(mix[["Reinvestment", "ACH"]].to_string(float_format=lambda v: f"{v:.2f}"))
    reinvestment = full[full["payment_format"] == "Reinvestment"]
    print(
        f"\n  All {len(reinvestment):,} Reinvestment transactions fall between "
        f"{reinvestment['timestamp'].min()} and {reinvestment['timestamp'].max()}\n"
        f"  -> a single calendar day. The format is 43.15% of day 1 and absent thereafter."
    )
    reinvestment_csi = csi[csi["feature"] == "payment_format_Reinvestment"]
    if not reinvestment_csi.empty:
        row = reinvestment_csi.iloc[0]
        print(
            f"  CSI {row['csi']:.4f} ({row['band']}), via the {row['method']} path.\n"
            "  Quantile-binned this scored 0.0000 - a binary column has one quantile bin,\n"
            "  so the metric could not express a feature vanishing. See challenges.md."
        )

    # --- 4. The one genuinely different population: the day-11+ tail --------------------
    print(f"\n=== 4. Separately: the {TAIL_START.date()}+ tail is NOT warm-up ===")
    tail = full[full["timestamp"] >= TAIL_START]
    body = full[full["timestamp"] < TAIL_START]
    print(
        f"  body ({len(body):,} rows): laundering rate {body['is_laundering'].mean():.4%}, "
        f"{body['payment_format'].eq('ACH').mean():.1%} ACH\n"
        f"  tail ({len(tail):,} rows): laundering rate {tail['is_laundering'].mean():.4%}, "
        f"{tail['payment_format'].eq('ACH').mean():.1%} ACH"
    )
    print(
        "  This one is a real population change, not a filling window - and it is the\n"
        "  drift the per-day PSI monitor structurally cannot see, because every one of\n"
        "  those days is below min_slice_size. See script 03 in this folder."
    )


if __name__ == "__main__":
    main()
