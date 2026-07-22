"""Account/window feature engineering checks on hand-built synthetic cases, plus the
leakage assertion required by PLAN.md Phase 3: a feature computed "as of" time T never
uses rows with timestamp > T.
"""

import pandas as pd
import pytest

from src.features import (
    assemble_feature_table,
    build_modelling_table,
    compute_account_features,
    compute_pass_through_ratios,
    compute_structuring_scores,
)


def _txns(rows):
    columns = ["timestamp", "from_account_key", "to_account_key", "amount_paid_usd"]
    df = pd.DataFrame(rows, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_laundering"] = 0
    df["payment_format"] = "ACH"
    return df


def test_sender_out_count_and_sum_within_window():
    # A sends to B three times within 1 day, then a 4th time 2 days later (outside the
    # 1d window relative to the 4th txn, since window looks backward from each row).
    df = _txns(
        [
            ("2022-09-01 00:00", "A", "B", 100.0),
            ("2022-09-01 06:00", "A", "B", 200.0),
            ("2022-09-01 12:00", "A", "B", 300.0),
            ("2022-09-03 00:00", "A", "B", 400.0),
        ]
    )
    features = compute_account_features(df, windows_days=[1])

    assert list(features["sender_out_1d_count"]) == [1, 2, 3, 1]
    assert list(features["sender_out_1d_amount_usd"]) == [100.0, 300.0, 600.0, 400.0]


def test_sender_in_counts_receiving_history_not_sending():
    # A receives money from X at 00:00, then sends to B at 06:00 same day — the sender_in
    # feature on A's outgoing txn should reflect the inbound txn from X.
    df = _txns(
        [
            ("2022-09-01 00:00", "X", "A", 500.0),
            ("2022-09-01 06:00", "A", "B", 100.0),
        ]
    )
    features = compute_account_features(df, windows_days=[1])

    # Row 0 is X->A: X's own out-history is what sender_in/out_* describe for row 0's
    # sender (X), not A. Check row 1 (A->B) instead, where the sender is A.
    assert features.loc[1, "sender_in_1d_count"] == 1
    assert features.loc[1, "sender_in_1d_amount_usd"] == 500.0
    assert features.loc[1, "sender_out_1d_count"] == 1  # A's own out-history: this txn


def test_receiver_in_distinct_counterparties_counts_fan_in():
    # Three distinct senders (X, Y, Z) all pay account C within a day -> fan-in of 3
    # on C's own third inbound txn (window is backward-looking, inclusive of self).
    df = _txns(
        [
            ("2022-09-01 00:00", "X", "C", 100.0),
            ("2022-09-01 01:00", "Y", "C", 100.0),
            ("2022-09-01 02:00", "Z", "C", 100.0),
        ]
    )
    features = compute_account_features(df, windows_days=[1])

    assert list(features["receiver_in_1d_distinct_counterparties"]) == [1, 2, 3]


def test_sender_out_distinct_counterparties_counts_fan_out():
    # One sender (A) pays three distinct receivers (X, Y, Z) within a day -> fan-out.
    df = _txns(
        [
            ("2022-09-01 00:00", "A", "X", 100.0),
            ("2022-09-01 01:00", "A", "Y", 100.0),
            ("2022-09-01 02:00", "A", "Z", 100.0),
        ]
    )
    features = compute_account_features(df, windows_days=[1])

    assert list(features["sender_out_1d_distinct_counterparties"]) == [1, 2, 3]


def test_distinct_counterparties_drops_out_of_window():
    # A pays X, then Y two days later (outside a 1d window) -> distinct count resets,
    # doesn't accumulate forever, proving this is a genuine sliding window not a
    # cumulative-since-start count.
    df = _txns(
        [
            ("2022-09-01 00:00", "A", "X", 100.0),
            ("2022-09-03 00:00", "A", "Y", 100.0),
        ]
    )
    features = compute_account_features(df, windows_days=[1])

    assert list(features["sender_out_1d_distinct_counterparties"]) == [1, 1]


def test_structuring_score_counts_only_subthreshold_history():
    # Two sub-threshold (9000-9999) txns from A within the window, one at-or-above
    # threshold txn that must not count.
    df = _txns(
        [
            ("2022-09-01 00:00", "A", "B", 9500.0),
            ("2022-09-01 01:00", "A", "B", 9600.0),
            ("2022-09-01 02:00", "A", "B", 15000.0),
        ]
    )
    scores = compute_structuring_scores(
        df, windows_days=[1], threshold_usd=10000, threshold_fraction=0.9
    )
    assert list(scores["structuring_score_1d"]) == [1, 2, 2]


def test_pass_through_ratio_near_one_when_inflow_equals_outflow():
    df = _txns(
        [
            ("2022-09-01 00:00", "X", "A", 1000.0),
            ("2022-09-01 01:00", "A", "B", 999.0),
        ]
    )
    account_features = compute_account_features(df, windows_days=[1])
    ratios = compute_pass_through_ratios(account_features, windows_days=[1])

    # Row 1: A's sender_in (1000) / (sender_out (999) + eps=1) == 1.0
    assert ratios.loc[1, "sender_pass_through_ratio_1d"] == pytest.approx(1.0)


def test_no_feature_uses_future_transactions():
    # The core leakage guarantee: change a transaction's amount/timestamp-adjacent
    # behaviour in the future and confirm every earlier row's features are unaffected.
    base = _txns(
        [
            ("2022-09-01 00:00", "A", "B", 100.0),
            ("2022-09-01 06:00", "A", "B", 200.0),
            ("2022-09-02 00:00", "A", "B", 300.0),
        ]
    )
    features_before = compute_account_features(base, windows_days=[1, 7, 30])

    mutated = base.copy()
    mutated.loc[2, "amount_paid_usd"] = 999_999.0
    features_after = compute_account_features(mutated, windows_days=[1, 7, 30])

    pd.testing.assert_frame_equal(
        features_before.iloc[:2].reset_index(drop=True),
        features_after.iloc[:2].reset_index(drop=True),
    )


def test_assemble_feature_table_runs_end_to_end():
    config = {
        "fx_rates_to_usd": {"US Dollar": 1.0},
        "windows": {"rolling_days": [1, 7]},
        "rules_baseline": {
            "large_amount_threshold_usd": 10000,
            "structuring_threshold_fraction": 0.9,
        },
    }
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-09-01 00:00", "2022-09-01 01:00"]),
            "from_account_key": ["A", "B"],
            "to_account_key": ["B", "A"],
            "amount_paid": [100.0, 200.0],
            "payment_currency": ["US Dollar", "US Dollar"],
            "payment_format": ["ACH", "ACH"],
            "is_laundering": [0, 0],
        }
    )
    result = assemble_feature_table(df, config)

    assert len(result) == 2
    assert "sender_out_1d_count" in result.columns
    assert "structuring_score_7d" in result.columns
    assert "sender_pass_through_ratio_1d" in result.columns
    assert not result.isna().any().any()


def test_build_modelling_table_joins_account_and_graph_features():
    config = {
        "fx_rates_to_usd": {"US Dollar": 1.0},
        "windows": {"rolling_days": [1]},
        "rules_baseline": {
            "large_amount_threshold_usd": 10000,
            "structuring_threshold_fraction": 0.9,
        },
        "graph_features": {"cycle_max_length": 4},
    }
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-09-01 00:00", "2022-09-01 01:00"]),
            "from_account_key": ["A", "B"],
            "to_account_key": ["B", "A"],
            "amount_paid": [100.0, 200.0],
            "payment_currency": ["US Dollar", "US Dollar"],
            "payment_format": ["ACH", "ACH"],
            "is_laundering": [0, 0],
        }
    )
    result = build_modelling_table(df, config)

    assert len(result) == 2
    assert "sender_out_1d_count" in result.columns  # account feature
    assert "sender_graph_out_degree" in result.columns  # graph feature
    assert not result.isna().any().any()


def test_compute_account_features_rejects_missing_columns():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2022-09-01"])})
    with pytest.raises(ValueError, match="missing columns"):
        compute_account_features(df, windows_days=[1])
