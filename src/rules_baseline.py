"""Naive rules engine (large-amount, structuring, pass-through) — the baseline to beat.

Implemented in PLAN.md Phase 2. Every model built later is judged by how much it improves
on this: fewer false positives at equal or better recall of `is_laundering`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_config, load_transactions


def add_usd_amount(df: pd.DataFrame, fx_rates: dict[str, float]) -> pd.DataFrame:
    """Convert `amount_paid` to a USD-equivalent column so amount-based thresholds are
    comparable across the dataset's 15 payment currencies (see config.yaml's
    fx_rates_to_usd for why a static table is used and what that assumes).
    """
    missing = set(df["payment_currency"].unique()) - set(fx_rates)
    if missing:
        raise ValueError(f"No FX rate configured for currencies: {sorted(missing)}")

    df = df.copy()
    df["amount_paid_usd"] = df["amount_paid"] * df["payment_currency"].map(fx_rates)
    return df


def flag_large_amount(df: pd.DataFrame, threshold_usd: float) -> pd.Series:
    """Flag any transaction whose USD-equivalent paid amount exceeds a fixed threshold —
    the classic CTR-style large-amount rule.
    """
    return df["amount_paid_usd"] > threshold_usd


def _rolling_window_counts(timestamps: pd.Series, window: pd.Timedelta) -> np.ndarray:
    """For each timestamp, count how many timestamps (including itself) fall within
    `window` ending at that timestamp. O(n log n) via searchsorted, not O(n^2).
    """
    values = np.sort(timestamps.values.astype("datetime64[ns]"))
    window_start = values - window.to_timedelta64()
    left = np.searchsorted(values, window_start, side="right")
    right = np.searchsorted(values, values, side="right")
    counts_by_sorted_pos = right - left

    order = np.argsort(timestamps.values.astype("datetime64[ns]"), kind="stable")
    counts = np.empty_like(counts_by_sorted_pos)
    counts[order] = counts_by_sorted_pos
    return counts


def flag_structuring(
    df: pd.DataFrame,
    threshold_usd: float,
    threshold_fraction: float,
    window_hours: int,
    min_txn_count: int,
) -> pd.Series:
    """Flag accounts making several transactions just under `threshold_usd` within a
    short rolling window — classic structuring/smurfing.

    A transaction is "sub-threshold" if its USD-equivalent amount falls in
    [threshold_usd * threshold_fraction, threshold_usd). A transaction is flagged once
    `min_txn_count` or more sub-threshold transactions from the same sending account
    land within any `window_hours`-wide window ending at that transaction.
    """
    lower = threshold_usd * threshold_fraction
    sub_mask = df["amount_paid_usd"].between(lower, threshold_usd, inclusive="left")
    window = pd.Timedelta(hours=window_hours)

    flags = pd.Series(False, index=df.index)
    sub_df = df.loc[sub_mask, ["from_account_key", "timestamp"]]
    for _, group in sub_df.groupby("from_account_key"):
        counts = _rolling_window_counts(group["timestamp"], window)
        flagged_idx = group.index[counts >= min_txn_count]
        flags.loc[flagged_idx] = True

    return flags


def flag_pass_through(df: pd.DataFrame, window_hours: int) -> pd.Series:
    """Flag outgoing transactions where the sending account received funds within the
    preceding `window_hours` — "funds that don't rest".

    Known simplification: a same-account self-transaction (from_account_key ==
    to_account_key, which does occur in this dataset) can trivially satisfy its own
    inbound/outbound check. Left as-is per the brief's "deliberately simple" baseline —
    this is exactly the kind of blunt behaviour the ML model is meant to improve on.
    """
    window = pd.Timedelta(hours=window_hours)
    flags = pd.Series(False, index=df.index)

    inbound_by_account = {
        key: np.sort(group["timestamp"].values.astype("datetime64[ns]"))
        for key, group in df.groupby("to_account_key")
    }

    for account_key, out_group in df.groupby("from_account_key"):
        in_times = inbound_by_account.get(account_key)
        if in_times is None or len(in_times) == 0:
            continue

        out_times = out_group["timestamp"].values.astype("datetime64[ns]")
        window_start = out_times - window.to_timedelta64()
        left = np.searchsorted(in_times, window_start, side="right")
        right = np.searchsorted(in_times, out_times, side="right")
        has_recent_inbound = (right - left) > 0

        flagged_idx = out_group.index[has_recent_inbound]
        flags.loc[flagged_idx] = True

    return flags


def apply_rules_baseline(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run all three rules and combine into a single per-transaction alert flag."""
    rules_cfg = config["rules_baseline"]
    df = add_usd_amount(df, config["fx_rates_to_usd"])

    result = df[
        ["timestamp", "from_account_key", "to_account_key", "amount_paid_usd", "is_laundering"]
    ].copy()
    result["flag_large_amount"] = flag_large_amount(df, rules_cfg["large_amount_threshold_usd"])
    result["flag_structuring"] = flag_structuring(
        df,
        rules_cfg["large_amount_threshold_usd"],
        rules_cfg["structuring_threshold_fraction"],
        rules_cfg["structuring_window_hours"],
        rules_cfg["structuring_min_txn_count"],
    )
    result["flag_pass_through"] = flag_pass_through(df, rules_cfg["pass_through_window_hours"])
    result["alert"] = result[
        ["flag_large_amount", "flag_structuring", "flag_pass_through"]
    ].any(axis=1)

    return result


def summarize_baseline(result: pd.DataFrame) -> dict:
    """Alert count, precision, recall against `is_laundering` — the numbers every later
    model in this project must beat.
    """
    alerts = result["alert"]
    labels = result["is_laundering"].astype(bool)
    n_alerts = int(alerts.sum())
    n_positives = int(labels.sum())
    true_positives = int((alerts & labels).sum())

    return {
        "n_alerts": n_alerts,
        "alert_rate": float(alerts.mean()),
        "true_positives": true_positives,
        "precision": true_positives / n_alerts if n_alerts else 0.0,
        "recall": true_positives / n_positives if n_positives else 0.0,
        "per_rule_alert_counts": {
            "large_amount": int(result["flag_large_amount"].sum()),
            "structuring": int(result["flag_structuring"].sum()),
            "pass_through": int(result["flag_pass_through"].sum()),
        },
    }


if __name__ == "__main__":
    config = load_config()
    txns = load_transactions(f"{config['paths']['raw_dir']}/HI-Small_Trans.csv")

    result = apply_rules_baseline(txns, config)
    summary = summarize_baseline(result)

    print("Rules baseline summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
