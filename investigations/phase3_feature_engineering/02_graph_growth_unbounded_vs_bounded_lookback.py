"""Reproduces the Phase 3 finding in reports/challenges.md: an "all history to date"
account graph never stopped growing across HI-Small's 18-day span (its first 2 days
alone hold ~1.9M of its 5.08M transactions), and the fix -- bounding each day's graph
to a trailing 7-day lookback with incremental edge pruning -- keeps per-day cost
roughly flat instead.

The old unbounded implementation no longer exists in src/graph_features.py -- it's
reproduced standalone here (a naive MultiDiGraph that only ever grows) so the "before"
number is a real measurement, not a quoted figure. The "after" side calls the actual
current `build_daily_graph_features` from src/graph_features.py.

This re-runs the actual profiling process that led to the fix, at a smaller day-count
range than the original investigation (which went up to 9 of 18 days and took ~500s at
the worst point) so this script finishes in a reasonable time -- the growth trend is
already unambiguous by day 6.

Run from the repo root: venve/python.exe investigations/phase3_feature_engineering/02_graph_growth_unbounded_vs_bounded_lookback.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src.data_loader import load_config, load_transactions
from src.graph_features import build_daily_graph_features, graph_snapshot_features


def unbounded_graph_features(df, cycle_max_length=3):
    """The original (replaced) design: one running graph, edges added once per day and
    never removed -- "all history to date" per the original PLAN.md Phase 3 wording.
    """
    df = df.copy()
    df["_date"] = df["timestamp"].dt.floor("D")
    dates = sorted(df["_date"].unique())

    running_graph = nx.MultiDiGraph()
    for date in dates:
        day_df = df.loc[df["_date"] == date]
        accounts_today = set(day_df["from_account_key"]) | set(day_df["to_account_key"])
        graph_snapshot_features(running_graph, accounts_today, cycle_max_length)
        for row in day_df.itertuples(index=False):
            running_graph.add_edge(row.from_account_key, row.to_account_key, timestamp=row.timestamp)
    return running_graph


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    raw_dir = Path(__file__).resolve().parents[2] / config["paths"]["raw_dir"]
    txns = load_transactions(raw_dir / "HI-Small_Trans.csv")
    txns["_date"] = txns["timestamp"].dt.floor("D")
    dates = sorted(txns["_date"].unique())

    graph_cfg = config["graph_features"]
    print(f"cycle_max_length={graph_cfg['cycle_max_length']}  lookback_days={graph_cfg['lookback_days']}\n")

    for n_days in [3, 6, 9]:
        cutoff = dates[n_days - 1]
        subset = txns[txns["_date"] <= cutoff].drop(columns=["_date"]).copy()

        t0 = time.time()
        unbounded_graph_features(subset, graph_cfg["cycle_max_length"])
        unbounded_elapsed = time.time() - t0

        t0 = time.time()
        build_daily_graph_features(
            subset,
            cycle_max_length=graph_cfg["cycle_max_length"],
            lookback_days=graph_cfg["lookback_days"],
        )
        bounded_elapsed = time.time() - t0

        print(
            f"first {n_days} days ({len(subset):>9,} rows): "
            f"unbounded={unbounded_elapsed:7.1f}s   bounded-lookback={bounded_elapsed:7.1f}s"
        )

    print(
        "\nExpect: unbounded's per-day cost keeps climbing (the growth that made the "
        "full 18-day unbounded run infeasible); bounded-lookback's cost stays roughly "
        "flat once past the lookback window."
    )


if __name__ == "__main__":
    main()
