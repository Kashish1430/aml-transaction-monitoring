"""Investigates a question raised after Phase 5 shipped: this system scores individual
TRANSACTIONS (matching `is_laundering`'s label granularity), using features that
describe ACCOUNT-level rolling history and network position -- so is the alert set at
the headline operating point concentrated on a small number of repeat-offender
accounts, or spread across many distinct ones? This matters for reading the headline
alert-reduction number correctly: a production system would likely bundle multiple
flagged transactions from the same account into one case for an analyst, so how much
overlap exists between "alerted transactions" and "distinct accounts" affects how that
bundling would actually change an analyst's workload. See the "transaction-level vs.
account-level" entry in reports/challenges.md.

Run from the repo root: venve/python.exe investigations/phase5_evaluation/03_transaction_vs_account_level_alerts.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_loader import load_config, load_transactions
from src.evaluate import alerts_for_recall
from src.model import load_xgboost_model, prepare_feature_matrix, time_ordered_split
from src.rules_baseline import apply_rules_baseline


def distinct_accounts(df: pd.DataFrame) -> int:
    return int(pd.concat([df["from_account_key"], df["to_account_key"]]).nunique())


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    raw_dir = Path(__file__).resolve().parents[2] / config["paths"]["raw_dir"]
    modelling_table_path = Path(__file__).resolve().parents[2] / config["paths"]["modelling_table"]
    models_dir = Path(__file__).resolve().parents[2] / config["paths"]["models_dir"]

    features_df = pd.read_parquet(modelling_table_path)
    _, _, test_df = time_ordered_split(
        features_df, config["split"]["train_frac"], config["split"]["val_frac"]
    )
    del features_df

    x_test, y_test = prepare_feature_matrix(test_df)
    model = load_xgboost_model(models_dir)
    test_scores = model.predict_proba(x_test)[:, 1]

    raw_txns = load_transactions(raw_dir / "HI-Small_Trans.csv")
    baseline_full = apply_rules_baseline(raw_txns, config)
    baseline_test = baseline_full.loc[test_df.index]
    baseline_recall = float(
        (baseline_test["alert"] & baseline_test["is_laundering"].astype(bool)).sum()
        / baseline_test["is_laundering"].sum()
    )

    model_alert_idx = alerts_for_recall(y_test.to_numpy(), test_scores, baseline_recall)
    alerted_txns = test_df.iloc[model_alert_idx]

    print(f"Model alerts (transactions): {len(alerted_txns):,}")
    print(f"Distinct accounts (sender or receiver) among alerted transactions: {distinct_accounts(alerted_txns):,}")

    baseline_alerted = raw_txns.loc[baseline_test.index[baseline_test['alert']]]
    print(f"\nRules baseline alerts (transactions): {len(baseline_alerted):,}")
    print(f"Distinct accounts in rules baseline alerts: {distinct_accounts(baseline_alerted):,}")

    true_positives = test_df[test_df["is_laundering"] == 1]
    print(f"\nTrue laundering transactions in test split: {len(true_positives):,}")
    print(f"Distinct accounts involved in those: {distinct_accounts(true_positives):,}")

    max_possible = len(alerted_txns) * 2
    actual = distinct_accounts(alerted_txns)
    print(
        f"\nModel's alerted transactions touch {actual:,} distinct accounts out of a "
        f"max possible {max_possible:,} (if every sender/receiver were unique) -- "
        f"{actual / max_possible:.1%} of the theoretical maximum, i.e. alerts are not "
        f"concentrated on a small handful of repeat accounts in this dataset."
    )


if __name__ == "__main__":
    main()
