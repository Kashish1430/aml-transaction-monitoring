"""Reproduces the Phase 3 finding in reports/challenges.md: the first implementation
of the distinct-counterparty rolling window used `DataFrame.groupby` per account and
was 30-40x slower than necessary, because most accounts have only 1-2 transactions in
an 18-day window, so groupby's per-group object-construction overhead dominated the
actual O(1)-per-row two-pointer work.

The old `groupby`-based implementation no longer exists in src/features.py (it was
replaced, not kept around as dead code) -- it's reproduced standalone here, faithful to
the original approach, so the "before" side of the comparison is a real measurement
against real data, not a number quoted from memory. The "after" side imports and times
the actual current implementation from src/features.py.

Run from the repo root: venve/python.exe investigations/phase3_feature_engineering/01_groupby_vs_vectorized_distinct_counterparties.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.data_loader import load_config, load_transactions
from src.features import _build_account_events, _rolling_distinct_counterparties
from src.rules_baseline import add_usd_amount


def old_groupby_based_distinct_counterparties(
    events: pd.DataFrame, query: pd.DataFrame, window: pd.Timedelta
) -> pd.Series:
    """The original (replaced) implementation: a two-pointer sliding window per
    account, but with both `events` and `query` grouped via `DataFrame.groupby` --
    which, for HI-Small's data, constructs one small DataFrame object per account for
    accounts that mostly only have 1-2 transactions total.
    """
    result = pd.Series(0, index=query["row_pos"], dtype="int64")
    window_td = np.timedelta64(window)

    events_by_account = {acct: grp for acct, grp in events.groupby("account_key", sort=False)}

    for account, qgroup in query.groupby("account_key", sort=False):
        acct_events = events_by_account.get(account)
        q_sorted = qgroup.sort_values("timestamp")
        q_times = q_sorted["timestamp"].values.astype("datetime64[ns]")
        q_positions = q_sorted["row_pos"].values

        if acct_events is None or len(acct_events) == 0:
            continue

        e_sorted = acct_events.sort_values("timestamp")
        e_times = e_sorted["timestamp"].values.astype("datetime64[ns]")
        e_cps = e_sorted["counterparty_key"].values
        n = len(e_times)

        left = 0
        right = 0
        counts: dict = {}
        distinct = 0
        out = np.empty(len(q_times), dtype=np.int64)

        for i, q_time in enumerate(q_times):
            while right < n and e_times[right] <= q_time:
                cp = e_cps[right]
                if counts.get(cp, 0) == 0:
                    distinct += 1
                counts[cp] = counts.get(cp, 0) + 1
                right += 1
            window_start = q_time - window_td
            while left < right and e_times[left] <= window_start:
                cp = e_cps[left]
                counts[cp] -= 1
                if counts[cp] == 0:
                    distinct -= 1
                left += 1
            out[i] = distinct

        result.loc[q_positions] = out

    return result.sort_index()


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    raw_dir = Path(__file__).resolve().parents[2] / config["paths"]["raw_dir"]
    txns = load_transactions(raw_dir / "HI-Small_Trans.csv")

    for n in [5_000, 20_000]:
        subset = add_usd_amount(txns.iloc[:n].copy(), config["fx_rates_to_usd"])
        events = _build_account_events(subset)
        out_events = events[events["direction"] == "out"]

        query = pd.DataFrame(
            {
                "account_key": subset["from_account_key"].values,
                "timestamp": subset["timestamp"].values,
                "row_pos": np.arange(len(subset)),
            }
        )
        window = pd.Timedelta(days=1)

        t0 = time.time()
        old_groupby_based_distinct_counterparties(out_events, query, window)
        old_elapsed = time.time() - t0

        t0 = time.time()
        _rolling_distinct_counterparties(out_events, query, window)
        new_elapsed = time.time() - t0

        print(
            f"n={n:>6}: groupby-based={old_elapsed:6.2f}s  "
            f"vectorized={new_elapsed:6.2f}s  speedup={old_elapsed / new_elapsed:5.1f}x"
        )


if __name__ == "__main__":
    main()
