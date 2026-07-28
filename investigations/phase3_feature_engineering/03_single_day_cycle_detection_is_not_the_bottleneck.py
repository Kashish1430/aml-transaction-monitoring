"""Reproduces the diagnostic step in reports/challenges.md's Phase 3 section that
clarified what was actually driving the graph-features slowdown: cycle detection
itself (`nx.simple_cycles`) is fast even on a large single-day graph -- the real cost
was letting the graph accumulate unboundedly across many days (see
02_graph_growth_unbounded_vs_bounded_lookback.py), not cycle enumeration being slow
per se.

Run from the repo root: venve/python.exe investigations/phase3_feature_engineering/03_single_day_cycle_detection_is_not_the_bottleneck.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import networkx as nx

from src.data_loader import load_config, load_transactions
from src.graph_features import _accounts_on_short_cycles


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    raw_dir = Path(__file__).resolve().parents[2] / config["paths"]["raw_dir"]
    txns = load_transactions(raw_dir / "HI-Small_Trans.csv")
    txns["_date"] = txns["timestamp"].dt.floor("D")
    dates = sorted(txns["_date"].unique())

    day1 = txns[txns["_date"] == dates[0]]
    print(f"Day 1: {len(day1):,} transactions")

    g = nx.MultiDiGraph()
    t0 = time.time()
    for row in day1.itertuples(index=False):
        g.add_edge(row.from_account_key, row.to_account_key, timestamp=row.timestamp)
    print(
        f"Build graph: {time.time() - t0:.2f}s, "
        f"nodes={g.number_of_nodes():,}, edges={g.number_of_edges():,}"
    )

    simple = nx.DiGraph(g)
    print(f"Simple graph: nodes={simple.number_of_nodes():,}, edges={simple.number_of_edges():,}")

    for max_len in [2, 3, 4]:
        t0 = time.time()
        cyc = _accounts_on_short_cycles(simple, max_len)
        print(
            f"cycle_max_length={max_len}: {time.time() - t0:.2f}s, "
            f"{len(cyc):,} accounts on a cycle"
        )

    print(
        "\nCycle detection alone is a few seconds even on a ~680K-edge graph -- the "
        "growth problem in script 02 comes from letting this graph accumulate across "
        "many days without bound, not from nx.simple_cycles being inherently slow."
    )


if __name__ == "__main__":
    main()
