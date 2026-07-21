"""Directed account graph via networkx: degree, distinct counterparties, motif
detection (fan-in/fan-out/cycle).

Implemented in PLAN.md Phase 3. Leakage-safe by construction: features for any
transaction on calendar day D come from a graph built only from transactions on days
strictly before D (see `build_daily_graph_features`) — never same-day or future edges.
Day-granularity (not per-transaction incremental graphs) is a deliberate simplification:
HI-Small only spans ~17-18 days (see CLAUDE.md), so this bounds graph rebuilds to ~17
snapshots instead of one per transaction, at the cost of same-day edges never
contributing to same-day features (acceptable — same-day velocity is already covered by
the 1-day window in src/features.py).

The graph for each day is built from a trailing `lookback_days` window, not all history
since day 1. This is both a performance necessity and a defensible modelling choice:
graph-build and cycle-detection cost both scale with edge count, and HI-Small's first
2 days alone contribute ~1.9M of its 5.08M transactions (see CLAUDE.md's date-span
note), so an unbounded "all history to date" graph never stops growing across the
dataset's 18-day span — an early version of this function took 521s to process just the
first 9 days on a clearly worsening curve (profiled during Phase 3 development, see
PLAN.md). Bounding to a trailing window keeps per-day cost roughly constant once the
window fills, and a fan-in/fan-out/cycle motif from 15 days ago is stale signal for a
typology playing out over hours-to-days anyway.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import networkx as nx
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "from_account_key", "to_account_key"}


def graph_snapshot_features(
    g: nx.MultiDiGraph, accounts: Iterable[str], cycle_max_length: int = 4
) -> pd.DataFrame:
    """Per-account degree/motif features from a single, already time-bounded graph
    snapshot.

    - graph_in_degree / graph_out_degree: distinct-counterparty in/out edge counts.
      Computed on a simple DiGraph collapse of `g` so repeated transactions between the
      same pair count once, matching PLAN.md's "distinct counterparties" requirement.
    - graph_fan_in_score / graph_fan_out_score: in_degree / out_degree share of total
      degree — fan-in and fan-out typology signatures respectively.
    - graph_in_cycle: whether the account sits on a directed cycle of length
      <= cycle_max_length — the cycle typology signature (funds returning toward their
      origin).

    `build_daily_graph_features` calls this once per transaction and applies both a
    `sender_` and `receiver_` prefix to every column below, so each of Features 9-13
    here becomes 2 actual columns in the final modelling table.
    """
    simple = nx.DiGraph(g)
    cycle_accounts = _accounts_on_short_cycles(simple, cycle_max_length)

    records = []
    for account in accounts:
        in_deg = simple.in_degree(account) if simple.has_node(account) else 0
        out_deg = simple.out_degree(account) if simple.has_node(account) else 0
        total = in_deg + out_deg
        records.append(
            {
                "account_key": account,
                # Feature 9 — {role}_graph_in_degree: how many DISTINCT accounts has
                # this account received from within the lookback window? A graph-native
                # version of fan-in, robust to one counterparty sending many small
                # transactions (that would inflate a raw transaction count but not
                # this — collapsed to a simple DiGraph first).
                "graph_in_degree": in_deg,
                # Feature 10 — {role}_graph_out_degree: how many DISTINCT accounts has
                # this account sent to within the lookback window? Graph-native fan-out.
                "graph_out_degree": out_deg,
                # Feature 11 — {role}_graph_fan_in_score: in_degree's share of this
                # account's total degree. Close to 1.0 means the account is
                # structurally a "collector" (mostly receiving from many sources)
                # rather than a balanced participant or a distributor.
                "graph_fan_in_score": in_deg / total if total else 0.0,
                # Feature 12 — {role}_graph_fan_out_score: out_degree's share of total
                # degree — the mirror image of Feature 11; close to 1.0 means the
                # account is structurally a "distributor."
                "graph_fan_out_score": out_deg / total if total else 0.0,
                # Feature 13 — {role}_graph_in_cycle: does this account sit on a
                # directed cycle of length <= cycle_max_length within the lookback
                # window — i.e. do funds from this account eventually loop back to it
                # within a few hops? The cycle typology's signature: money "returning"
                # rather than moving in one direction, a pattern a legitimate business
                # relationship essentially never produces by accident.
                "graph_in_cycle": account in cycle_accounts,
            }
        )
    return pd.DataFrame.from_records(records)


def _accounts_on_short_cycles(g: nx.DiGraph, max_length: int) -> set[str]:
    """Accounts sitting on at least one directed cycle of length <= max_length.

    Uses `nx.simple_cycles` with `length_bound` (networkx >= 3.2) so cycle enumeration
    stays tractable on large graphs — unbounded simple-cycle enumeration is infeasible.
    """
    cycle_accounts: set[str] = set()
    for cycle in nx.simple_cycles(g, length_bound=max_length):
        cycle_accounts.update(cycle)
    return cycle_accounts


def build_daily_graph_features(
    df: pd.DataFrame, cycle_max_length: int = 4, lookback_days: int = 7
) -> pd.DataFrame:
    """As-of graph features for every transaction in `df`.

    Each transaction's sender_*/receiver_* columns come from `graph_snapshot_features`
    computed over the graph of transactions in the `lookback_days` strictly before that
    transaction's calendar date (see module docstring for why a bounded lookback, not
    all history, is used). Accounts with no prior history in the window get all-zero/
    False features (cold start, not missing data).
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"build_daily_graph_features missing columns: {sorted(missing)}")

    df = df.copy()
    df["_date"] = df["timestamp"].dt.floor("D")
    dates = sorted(df["_date"].unique())
    lookback = pd.Timedelta(days=lookback_days)

    day_frames = []
    running_graph = nx.MultiDiGraph()
    # Edges currently in `running_graph`, grouped by the day they were added — a FIFO of
    # (date, [(u, v, key), ...]) so edges can be pruned in O(1) amortized per edge as they
    # age out of the lookback window, instead of rebuilding the graph from scratch every
    # day (which would re-insert overlapping-window edges up to `lookback_days` times).
    pending_removals: deque = deque()

    for date in dates:
        cutoff = date - lookback
        while pending_removals and pending_removals[0][0] < cutoff:
            _, edges = pending_removals.popleft()
            for u, v, key in edges:
                running_graph.remove_edge(u, v, key=key)

        day_df = df.loc[df["_date"] == date]
        accounts_today = pd.concat(
            [day_df["from_account_key"], day_df["to_account_key"]]
        ).unique()

        snapshot = graph_snapshot_features(
            running_graph, accounts_today, cycle_max_length
        ).set_index("account_key")

        sender_feats = snapshot.reindex(day_df["from_account_key"]).add_prefix("sender_")
        sender_feats.index = day_df.index
        receiver_feats = snapshot.reindex(day_df["to_account_key"]).add_prefix("receiver_")
        receiver_feats.index = day_df.index

        day_frames.append(pd.concat([sender_feats, receiver_feats], axis=1))

        # Add today's edges only *after* computing today's features (leakage-safe), and
        # remember their keys so they can be pruned once they age out of the window.
        added = []
        for row in day_df.itertuples(index=False):
            key = running_graph.add_edge(
                row.from_account_key, row.to_account_key, timestamp=row.timestamp
            )
            added.append((row.from_account_key, row.to_account_key, key))
        pending_removals.append((date, added))

    result = pd.concat(day_frames).reindex(df.index)
    bool_cols = [c for c in result.columns if c.endswith("_in_cycle")]
    result[bool_cols] = result[bool_cols].fillna(False).astype(bool)
    numeric_cols = [c for c in result.columns if c not in bool_cols]
    result[numeric_cols] = result[numeric_cols].fillna(0)
    return result


if __name__ == "__main__":
    from src.data_loader import load_config, load_transactions

    config = load_config()
    txns = load_transactions(f"{config['paths']['raw_dir']}/HI-Small_Trans.csv")

    graph_cfg = config.get("graph_features", {})
    features = build_daily_graph_features(
        txns,
        cycle_max_length=graph_cfg.get("cycle_max_length", 3),
        lookback_days=graph_cfg.get("lookback_days", 7),
    )
    print(f"Graph feature table: {features.shape[0]} rows, {features.shape[1]} columns")
    print(features.columns.tolist())
