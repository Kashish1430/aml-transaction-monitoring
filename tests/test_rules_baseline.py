"""Rules-baseline logic checks on hand-built synthetic cases (PLAN.md Phase 2)."""

import pandas as pd
import pytest

from src.rules_baseline import (
    add_usd_amount,
    apply_rules_baseline,
    flag_large_amount,
    flag_pass_through,
    flag_structuring,
    summarize_baseline,
)


def test_add_usd_amount_converts_known_currencies():
    df = pd.DataFrame(
        {
            "amount_paid": [100.0, 50.0],
            "payment_currency": ["US Dollar", "Euro"],
        }
    )
    result = add_usd_amount(df, {"US Dollar": 1.0, "Euro": 2.0})
    assert list(result["amount_paid_usd"]) == [100.0, 100.0]


def test_add_usd_amount_fails_loudly_on_unknown_currency():
    df = pd.DataFrame({"amount_paid": [100.0], "payment_currency": ["Klingon Credit"]})
    with pytest.raises(ValueError, match="No FX rate configured"):
        add_usd_amount(df, {"US Dollar": 1.0})


def test_flag_large_amount_uses_strict_threshold():
    df = pd.DataFrame({"amount_paid_usd": [5000.0, 15000.0, 9999.99, 10000.01]})
    flags = flag_large_amount(df, threshold_usd=10000)
    assert list(flags) == [False, True, False, True]


def test_flag_structuring_fires_on_clustered_subthreshold_txns():
    # Account A: 3 sub-threshold (9000 <= x < 10000) transactions within 2 hours. The window
    # is backward-looking (leakage-safe), so only the 3rd transaction — the one that actually
    # completes the pattern — should flag; the first two couldn't have known more were coming.
    # Account B: only 2 such transactions ever — must never fire (min_txn_count=3).
    times = pd.to_datetime(
        [
            "2022-09-01 00:00",
            "2022-09-01 00:30",
            "2022-09-01 01:00",
            "2022-09-01 00:00",
            "2022-09-01 00:30",
        ]
    )
    df = pd.DataFrame(
        {
            "from_account_key": ["A", "A", "A", "B", "B"],
            "timestamp": times,
            "amount_paid_usd": [9500.0, 9500.0, 9500.0, 9500.0, 9500.0],
        }
    )

    flags = flag_structuring(
        df, threshold_usd=10000, threshold_fraction=0.9, window_hours=24, min_txn_count=3
    )

    assert list(flags[df["from_account_key"] == "A"]) == [False, False, True]
    assert not flags[df["from_account_key"] == "B"].any()


def test_flag_structuring_respects_window_boundary():
    # Same account, same sub-threshold amounts, but spread across 3 days — window is 24h,
    # so these must never cluster together.
    times = pd.to_datetime(["2022-09-01 00:00", "2022-09-02 00:00", "2022-09-03 00:00"])
    df = pd.DataFrame(
        {
            "from_account_key": ["A", "A", "A"],
            "timestamp": times,
            "amount_paid_usd": [9500.0, 9500.0, 9500.0],
        }
    )

    flags = flag_structuring(
        df, threshold_usd=10000, threshold_fraction=0.9, window_hours=24, min_txn_count=3
    )

    assert not flags.any()


def test_flag_pass_through_fires_on_quick_in_and_out():
    # Account X: inbound at 00:00, outbound at 00:30 — within the 2h window, must fire.
    # Account Y: inbound at 00:00, outbound 5h later — outside the window, must not fire.
    # Account Z: outbound with no inbound ever — must not fire.
    df = pd.DataFrame(
        {
            "from_account_key": ["OTHER", "X", "OTHER", "Y", "Z"],
            "to_account_key": ["X", "OTHER", "Y", "OTHER", "OTHER"],
            "timestamp": pd.to_datetime(
                [
                    "2022-09-01 00:00",  # inbound to X
                    "2022-09-01 00:30",  # X -> OTHER (outbound, quick pass-through)
                    "2022-09-01 00:00",  # inbound to Y
                    "2022-09-01 05:00",  # Y -> OTHER (outbound, too late)
                    "2022-09-01 00:00",  # Z -> OTHER, no prior inbound to Z
                ]
            ),
        }
    )

    flags = flag_pass_through(df, window_hours=2)

    assert flags.loc[1]  # X's outbound
    assert not flags.loc[3]  # Y's outbound, outside window
    assert not flags.loc[4]  # Z's outbound, no inbound at all


def test_apply_rules_baseline_and_summarize_baseline():
    config = {
        "rules_baseline": {
            "large_amount_threshold_usd": 10000,
            "structuring_window_hours": 24,
            "structuring_min_txn_count": 3,
            "structuring_threshold_fraction": 0.9,
            "pass_through_window_hours": 2,
        },
        "fx_rates_to_usd": {"US Dollar": 1.0},
    }
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-09-01 00:00", "2022-09-01 00:10"]),
            "from_account_key": ["A", "C"],
            "to_account_key": ["B", "D"],
            "amount_paid": [20000.0, 100.0],
            "payment_currency": ["US Dollar", "US Dollar"],
            "is_laundering": [1, 0],
        }
    )

    result = apply_rules_baseline(df, config)
    summary = summarize_baseline(result)

    assert result.loc[0, "alert"]  # large-amount rule fires on the 20,000 txn
    assert not result.loc[1, "alert"]  # unrelated C->D txn, no rule should touch it
    assert summary["n_alerts"] == 1
    assert summary["true_positives"] == 1
    assert summary["precision"] == 1.0
    assert summary["recall"] == 1.0
