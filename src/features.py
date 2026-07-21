"""Account/window feature engineering: velocity, volume, structuring, layering,
counterparty signals — all leakage-safe (every feature at a transaction's timestamp T
uses only transactions with timestamp <= T, per CLAUDE.md's no-leakage constraint).

Implemented in PLAN.md Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.graph_features import build_daily_graph_features
from src.rules_baseline import add_usd_amount

REQUIRED_COLUMNS = {"timestamp", "from_account_key", "to_account_key", "amount_paid_usd"}

# (role, direction) combos to compute count/sum for. "sender"/"receiver" pick which
# account of the transaction is of interest; "out"/"in" pick which side of *that
# account's own history* to look at (e.g. sender_in = has the sender been receiving
# money recently — a pass-through/layering signal, not the sender's outgoing activity).
_COUNT_SUM_COMBOS = [
    ("sender", "from_account_key", "out"),
    ("sender", "from_account_key", "in"),
    ("receiver", "to_account_key", "out"),
    ("receiver", "to_account_key", "in"),
]

# Distinct-counterparty counts are only computed for the two classic typology signals:
# fan-out (sender -> many distinct receivers) and fan-in (receiver <- many distinct
# senders). The other two combos are covered by the count/sum features above and add
# limited marginal signal for meaningfully more compute (two-pointer, not vectorizable
# via the merge_asof trick used for count/sum).
_DISTINCT_COMBOS = [
    ("sender", "from_account_key", "out"),
    ("receiver", "to_account_key", "in"),
]


def _build_account_events(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format table: one row per (account, direction) leg of each transaction.

    Every transaction contributes two events — the sender's outgoing leg and the
    receiver's incoming leg — so per-account rolling stats can be computed uniformly
    against either leg regardless of which account/direction a query is about.
    """
    sent = df[["from_account_key", "timestamp", "amount_paid_usd", "to_account_key"]].copy()
    sent.columns = ["account_key", "timestamp", "amount_usd", "counterparty_key"]
    sent["direction"] = "out"

    received = df[["to_account_key", "timestamp", "amount_paid_usd", "from_account_key"]].copy()
    received.columns = ["account_key", "timestamp", "amount_usd", "counterparty_key"]
    received["direction"] = "in"

    return pd.concat([sent, received], ignore_index=True)


def _rolling_count_sum(
    events: pd.DataFrame, query: pd.DataFrame, window: pd.Timedelta
) -> pd.DataFrame:
    """Count and USD sum of `events` (columns: account_key, timestamp, amount_usd) within
    (query_time - window, query_time] for each row of `query` (columns: account_key,
    timestamp, row_pos).

    Vectorized via the merge_asof cumulative-sum trick — no Python-level loop over
    accounts: sort events per account and take a running cumulative count/sum, then
    `merge_asof(direction="backward")` looks up "cumulative value as of time X" for any
    query time X (need not coincide with an actual event), and the window count/sum is
    just the difference of two such lookups (at query_time and at query_time - window).
    Query and event timestamps need not coincide — this generalizes rules_baseline's
    `_rolling_window_counts`, which only handled query points == event points.
    """
    events = events.sort_values("timestamp").reset_index(drop=True)
    events["_cum_count"] = events.groupby("account_key").cumcount() + 1
    events["_cum_amount"] = events.groupby("account_key")["amount_usd"].cumsum()
    cum_lookup = events[["account_key", "timestamp", "_cum_count", "_cum_amount"]]

    query = query.sort_values("timestamp").reset_index(drop=True)
    query["_window_start"] = query["timestamp"] - window

    right = pd.merge_asof(
        query[["account_key", "timestamp", "row_pos"]],
        cum_lookup,
        on="timestamp",
        by="account_key",
        direction="backward",
    ).set_index("row_pos")
    left = pd.merge_asof(
        query[["account_key", "_window_start", "row_pos"]].rename(
            columns={"_window_start": "timestamp"}
        ),
        cum_lookup,
        on="timestamp",
        by="account_key",
        direction="backward",
    ).set_index("row_pos")

    count = (right["_cum_count"].fillna(0) - left["_cum_count"].fillna(0)).astype("int64")
    amount_sum = right["_cum_amount"].fillna(0) - left["_cum_amount"].fillna(0)
    return pd.DataFrame({"count": count, "amount_sum": amount_sum}).sort_index()


def _rolling_distinct_counterparties(
    events: pd.DataFrame, query: pd.DataFrame, window: pd.Timedelta
) -> pd.Series:
    """Distinct counterparties among `events` within (query_time - window, query_time]
    for each row of `query`, grouped per account.

    Unlike count/sum, distinct-count is not a simple prefix-sum difference (a
    counterparty can leave and re-enter the window), so this uses a genuine two-pointer
    sliding window per account. Both frames are sorted once by (account_key, timestamp)
    and sliced with raw numpy arrays per account-group boundary — deliberately avoiding
    `DataFrame.groupby`, whose per-group object construction dominated runtime at
    HI-Small's scale (most accounts have only 1-2 transactions in an 18-day window, so
    groupby's fixed per-group overhead swamped the O(1)-per-row two-pointer work: an
    early profiling pass showed ~1ms/row from groupby object creation alone, which would
    have made the full 5M-row run take hours).
    """
    window_td = np.timedelta64(window)

    events_sorted = events.sort_values(["account_key", "timestamp"])
    e_accounts = events_sorted["account_key"].to_numpy()
    e_times = events_sorted["timestamp"].to_numpy()
    e_cps = events_sorted["counterparty_key"].to_numpy()

    query_sorted = query.sort_values(["account_key", "timestamp"])
    q_accounts = query_sorted["account_key"].to_numpy()
    q_times = query_sorted["timestamp"].to_numpy()
    q_positions = query_sorted["row_pos"].to_numpy()

    out = np.zeros(len(q_accounts), dtype="int64")
    unique_q_accounts, q_group_starts = np.unique(q_accounts, return_index=True)
    q_group_starts = np.append(q_group_starts, len(q_accounts))

    for gi, account in enumerate(unique_q_accounts):
        q_lo, q_hi = q_group_starts[gi], q_group_starts[gi + 1]
        e_lo = np.searchsorted(e_accounts, account, side="left")
        e_hi = np.searchsorted(e_accounts, account, side="right")
        if e_hi == e_lo:
            continue  # no history for this account -> distinct count stays 0

        et, ecp = e_times[e_lo:e_hi], e_cps[e_lo:e_hi]
        qt = q_times[q_lo:q_hi]
        n = len(et)

        left = right = 0
        counts: dict = {}
        distinct = 0
        local_out = np.empty(len(qt), dtype="int64")

        for i, q_time in enumerate(qt):
            while right < n and et[right] <= q_time:
                cp = ecp[right]
                c = counts.get(cp, 0)
                if c == 0:
                    distinct += 1
                counts[cp] = c + 1
                right += 1
            window_start = q_time - window_td
            while left < right and et[left] <= window_start:
                cp = ecp[left]
                counts[cp] -= 1
                if counts[cp] == 0:
                    distinct -= 1
                left += 1
            local_out[i] = distinct

        out[q_lo:q_hi] = local_out

    result = pd.Series(out, index=q_positions, name="distinct_counterparties")
    return result.sort_index()


def compute_account_features(df: pd.DataFrame, windows_days: list[int]) -> pd.DataFrame:
    """Build the per-transaction account/window feature table.

    For each (role, direction) combo in `_COUNT_SUM_COMBOS` and each window in
    `windows_days`, adds a transaction-count and USD-amount-sum column. Adds
    distinct-counterparty columns for the fan-out/fan-in combos in `_DISTINCT_COMBOS`.
    All columns are named `{role}_{direction}_{window}d_{stat}`.

    Leakage-safe: every rolling stat for row i is bounded above by df.loc[i, "timestamp"]
    — see tests/test_features.py for the explicit leakage assertion.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"compute_account_features missing columns: {sorted(missing)}")

    events = _build_account_events(df)
    out_events = events[events["direction"] == "out"]
    in_events = events[events["direction"] == "in"]
    direction_events = {"out": out_events, "in": in_events}

    features = pd.DataFrame(index=df.index)

    for role, account_col, direction in _COUNT_SUM_COMBOS:
        query = pd.DataFrame(
            {
                "account_key": df[account_col].values,
                "timestamp": df["timestamp"].values,
                "row_pos": np.arange(len(df)),
            }
        )
        for window_days in windows_days:
            window = pd.Timedelta(days=window_days)
            stats = _rolling_count_sum(direction_events[direction], query, window)
            features[f"{role}_{direction}_{window_days}d_count"] = stats["count"].to_numpy()
            features[f"{role}_{direction}_{window_days}d_amount_usd"] = stats[
                "amount_sum"
            ].to_numpy()

    for role, account_col, direction in _DISTINCT_COMBOS:
        query = pd.DataFrame(
            {
                "account_key": df[account_col].values,
                "timestamp": df["timestamp"].values,
                "row_pos": np.arange(len(df)),
            }
        )
        for window_days in windows_days:
            window = pd.Timedelta(days=window_days)
            distinct = _rolling_distinct_counterparties(direction_events[direction], query, window)
            features[f"{role}_{direction}_{window_days}d_distinct_counterparties"] = (
                distinct.to_numpy()
            )

    return features


def compute_structuring_scores(
    df: pd.DataFrame,
    windows_days: list[int],
    threshold_usd: float,
    threshold_fraction: float,
) -> pd.DataFrame:
    """Rolling count of the sender's own sub-threshold outgoing transactions in each
    window — a continuous generalisation of rules_baseline's fixed-24h binary
    structuring flag onto the same [1, 7, 30]-day windows as the other features.
    """
    lower = threshold_usd * threshold_fraction
    sub_mask = df["amount_paid_usd"].between(lower, threshold_usd, inclusive="left")

    events = df.loc[
        sub_mask, ["from_account_key", "timestamp", "amount_paid_usd"]
    ].copy()
    events.columns = ["account_key", "timestamp", "amount_usd"]

    query = pd.DataFrame(
        {
            "account_key": df["from_account_key"].values,
            "timestamp": df["timestamp"].values,
            "row_pos": np.arange(len(df)),
        }
    )

    scores = pd.DataFrame(index=df.index)
    for window_days in windows_days:
        window = pd.Timedelta(days=window_days)
        stats = _rolling_count_sum(events, query, window)
        scores[f"structuring_score_{window_days}d"] = stats["count"].to_numpy()
    return scores


def compute_pass_through_ratios(
    account_features: pd.DataFrame, windows_days: list[int], eps: float = 1.0
) -> pd.DataFrame:
    """Ratio of the sender's recent inflow to recent outflow, per window — near 1 means
    funds are flowing straight through the sender (classic layering signature) rather
    than accumulating. Derived from `compute_account_features`'s output, so call that
    first and pass its result in.
    """
    ratios = pd.DataFrame(index=account_features.index)
    for window_days in windows_days:
        inflow = account_features[f"sender_in_{window_days}d_amount_usd"]
        outflow = account_features[f"sender_out_{window_days}d_amount_usd"]
        ratios[f"sender_pass_through_ratio_{window_days}d"] = inflow / (outflow + eps)
    return ratios


def assemble_feature_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Full Phase 3 feature table: USD amounts + account/window features + structuring
    scores + pass-through ratios, joined onto the original transaction key columns.
    """
    df = add_usd_amount(df, config["fx_rates_to_usd"])
    windows_days = config["windows"]["rolling_days"]
    rules_cfg = config["rules_baseline"]

    account_features = compute_account_features(df, windows_days)
    structuring_scores = compute_structuring_scores(
        df,
        windows_days,
        rules_cfg["large_amount_threshold_usd"],
        rules_cfg["structuring_threshold_fraction"],
    )
    pass_through_ratios = compute_pass_through_ratios(account_features, windows_days)

    key_columns = df[
        [
            "timestamp",
            "from_account_key",
            "to_account_key",
            "amount_paid_usd",
            "payment_format",
            "is_laundering",
        ]
    ]
    return pd.concat(
        [key_columns, account_features, structuring_scores, pass_through_ratios], axis=1
    )


def build_modelling_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """The single Phase 3 deliverable: account/window features (this module) joined with
    graph/motif features (src/graph_features.py) into one row-aligned modelling table.

    Both halves are computed independently against the same `df` and concatenated on
    the shared index — safe because both preserve `df`'s original row order/index and
    neither drops rows.
    """
    account_table = assemble_feature_table(df, config)
    graph_cfg = config.get("graph_features", {})
    graph_table = build_daily_graph_features(
        df,
        cycle_max_length=graph_cfg.get("cycle_max_length", 3),
        lookback_days=graph_cfg.get("lookback_days", 7),
    )
    return pd.concat([account_table, graph_table], axis=1)


if __name__ == "__main__":
    from src.data_loader import load_config, load_transactions

    config = load_config()
    txns = load_transactions(f"{config['paths']['raw_dir']}/HI-Small_Trans.csv")

    features = build_modelling_table(txns, config)
    print(f"Modelling table: {features.shape[0]} rows, {features.shape[1]} columns")
    print(features.columns.tolist())

    out_path = config["paths"]["modelling_table"]
    features.to_parquet(out_path)
    print(f"Saved to {out_path}")
