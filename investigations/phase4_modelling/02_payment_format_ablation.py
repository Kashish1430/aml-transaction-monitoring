"""Reproduces the Phase 4 ablation in reports/challenges.md: retrains the identical
XGBoost model with every `payment_format_*` column dropped, to quantify -- not just
assert -- how much of the model's performance depends on the ACH correlation found in
01_payment_format_ach_correlation.py.

Run from the repo root: venve/python.exe investigations/phase4_modelling/02_payment_format_ablation.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data_loader import load_config
from src.model import compute_scale_pos_weight, prepare_feature_matrix, time_ordered_split, train_xgboost


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    modelling_table_path = Path(__file__).resolve().parents[2] / config["paths"]["modelling_table"]

    df = pd.read_parquet(modelling_table_path)
    train_df, val_df, test_df = time_ordered_split(
        df, config["split"]["train_frac"], config["split"]["val_frac"]
    )
    del df

    x_train, y_train = prepare_feature_matrix(train_df)
    x_val, y_val = prepare_feature_matrix(val_df)
    x_test, y_test = prepare_feature_matrix(test_df)

    payment_cols = [c for c in x_train.columns if c.startswith("payment_format_")]
    print(f"Dropping columns: {payment_cols}\n")

    x_train_ablated = x_train.drop(columns=payment_cols)
    x_val_ablated = x_val.drop(columns=payment_cols)
    x_test_ablated = x_test.drop(columns=payment_cols)

    scale_pos_weight = compute_scale_pos_weight(y_train)

    t0 = time.time()
    model_ablated = train_xgboost(x_train_ablated, y_train, x_val_ablated, y_val, scale_pos_weight, config["seed"])
    print(
        f"Trained (no payment_format) in {time.time() - t0:.1f}s, "
        f"best_iteration={model_ablated.best_iteration}, best_val_aucpr={model_ablated.best_score:.4f}"
    )

    test_scores = model_ablated.predict_proba(x_test_ablated)[:, 1]
    print(f"Test PR-AUC (no payment_format):  {average_precision_score(y_test, test_scores):.4f}")
    print(f"Test ROC-AUC (no payment_format): {roc_auc_score(y_test, test_scores):.4f}")

    order = np.argsort(-test_scores)
    y_sorted = y_test.to_numpy()[order]
    print("\nPrecision@k (no payment_format):")
    for k in config["evaluation"]["precision_at_k"]:
        hits = y_sorted[:k].sum()
        print(f"  precision@{k}: {hits}/{k} = {hits / k:.2%}")

    importances = model_ablated.get_booster().get_score(importance_type="gain")
    top = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop features (no payment_format):")
    for name, gain in top:
        print(f"  {name:45s} {gain:>10.1f}")


if __name__ == "__main__":
    main()
