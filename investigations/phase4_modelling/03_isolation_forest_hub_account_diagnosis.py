"""Reproduces the Phase 4 finding in reports/challenges.md: Isolation Forest's top-1000
"anomalous" test transactions have zero overlap with XGBoost's top-1000, and this
script shows why -- they sit at the extreme tail of raw volume features (mean
sender_out_30d_count near the dataset's actual maximum), i.e. Isolation Forest is
rediscovering high-throughput hub accounts, not laundering-specific behavior.

Run from the repo root: venve/python.exe investigations/phase4_modelling/03_isolation_forest_hub_account_diagnosis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data_loader import load_config
from src.model import prepare_feature_matrix, time_ordered_split, train_isolation_forest


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    modelling_table_path = Path(__file__).resolve().parents[2] / config["paths"]["modelling_table"]

    df = pd.read_parquet(modelling_table_path)
    train_df, _, test_df = time_ordered_split(
        df, config["split"]["train_frac"], config["split"]["val_frac"]
    )
    del df

    x_train, _ = prepare_feature_matrix(train_df)
    x_test, y_test = prepare_feature_matrix(test_df)

    iso = train_isolation_forest(x_train, config["seed"])
    iso_scores = -iso.decision_function(x_test)

    top_1000_idx = np.argsort(-iso_scores)[:1000]
    top_1000 = x_test.iloc[top_1000_idx]

    print("Isolation Forest top-1000 flagged transactions -- mean feature value vs. overall test set:")
    cols_to_check = ["amount_paid_usd", "sender_out_30d_count", "sender_out_30d_distinct_counterparties"]
    for col in cols_to_check:
        print(
            f"  {col:40s} top1000_mean={top_1000[col].mean():>15,.0f}  "
            f"overall_mean={x_test[col].mean():>12,.0f}  overall_max={x_test[col].max():>15,.0f}"
        )

    n_true_positives_caught = y_test.to_numpy()[top_1000_idx].sum()
    print(f"\nActual laundering transactions among the top-1000: {n_true_positives_caught} / 1000")
    print(f"Total laundering transactions in the test split: {int(y_test.sum())}")


if __name__ == "__main__":
    main()
