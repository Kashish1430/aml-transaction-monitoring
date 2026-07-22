"""Graph feature checks on hand-built synthetic cases (PLAN.md Phase 3), including the
day-boundary leakage guarantee: a transaction's graph features must never reflect
same-day or future edges.
"""

import networkx as nx
import pandas as pd
import pytest

from src.graph_features import build_daily_graph_features, graph_snapshot_features


def _txns(rows):
    df = pd.DataFrame(rows, columns=["timestamp", "from_account_key", "to_account_key"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def test_graph_snapshot_features_degree_and_fan_scores():
    g = nx.MultiDiGraph()
    g.add_edge("A", "X")
    g.add_edge("A", "Y")
    g.add_edge("A", "Z")  # A fans out to 3 distinct receivers
    g.add_edge("W", "C")
    g.add_edge("V", "C")  # C fans in from 2 distinct senders

    result = graph_snapshot_features(g, ["A", "C", "isolated"]).set_index("account_key")

    assert result.loc["A", "graph_out_degree"] == 3
    assert result.loc["A", "graph_in_degree"] == 0
    assert result.loc["A", "graph_fan_out_score"] == pytest.approx(1.0)

    assert result.loc["C", "graph_in_degree"] == 2
    assert result.loc["C", "graph_fan_in_score"] == pytest.approx(1.0)

    assert result.loc["isolated", "graph_in_degree"] == 0
    assert result.loc["isolated", "graph_out_degree"] == 0
    assert result.loc["isolated", "graph_fan_in_score"] == 0.0


def test_graph_snapshot_features_collapses_repeated_edges():
    # A pays B three times -> distinct-counterparty degree is 1, not 3.
    g = nx.MultiDiGraph()
    g.add_edge("A", "B")
    g.add_edge("A", "B")
    g.add_edge("A", "B")

    result = graph_snapshot_features(g, ["A"]).set_index("account_key")
    assert result.loc["A", "graph_out_degree"] == 1


def test_graph_snapshot_features_flags_short_cycles():
    # A -> B -> C -> A is a 3-cycle; D is unrelated and must not be flagged.
    g = nx.MultiDiGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "A")
    g.add_edge("D", "E")

    result = graph_snapshot_features(g, ["A", "B", "C", "D"], cycle_max_length=4).set_index(
        "account_key"
    )
    assert result.loc["A", "graph_in_cycle"]
    assert result.loc["B", "graph_in_cycle"]
    assert result.loc["C", "graph_in_cycle"]
    assert not result.loc["D", "graph_in_cycle"]


def test_graph_snapshot_features_respects_cycle_length_bound():
    # A 5-hop cycle must not be flagged when cycle_max_length=3.
    g = nx.MultiDiGraph()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "D")
    g.add_edge("D", "E")
    g.add_edge("E", "A")

    result = graph_snapshot_features(g, ["A"], cycle_max_length=3).set_index("account_key")
    assert not result.loc["A", "graph_in_cycle"]


def test_build_daily_graph_features_excludes_same_day_edges():
    # A->B and B->A both happen on 2022-09-01 (would form a 2-cycle if same-day edges
    # counted), then A->C happens on 2022-09-02. Day-1 features must show zero history
    # (cold start); the day-2 transaction may see day-1's edges but not create a
    # same-day cycle from A->B/B->A.
    df = _txns(
        [
            ("2022-09-01 00:00", "A", "B"),
            ("2022-09-01 01:00", "B", "A"),
            ("2022-09-02 00:00", "A", "C"),
        ]
    )
    result = build_daily_graph_features(df, cycle_max_length=4)

    # Day 1: no prior history at all -> cold start.
    assert result.loc[0, "sender_graph_out_degree"] == 0
    assert result.loc[1, "sender_graph_out_degree"] == 0
    assert not result.loc[0, "sender_graph_in_cycle"]
    assert not result.loc[1, "sender_graph_in_cycle"]

    # Day 2: graph now includes both day-1 edges (A->B, B->A), so A has out_degree 1
    # (to B) from prior history, and sits on the 2-cycle A<->B.
    assert result.loc[2, "sender_graph_out_degree"] == 1
    assert result.loc[2, "sender_graph_in_cycle"]


def test_build_daily_graph_features_excludes_edges_older_than_lookback():
    # A->B happens on day 1. A transaction involving A on day 4 must NOT see it when
    # lookback_days=2 (day 1 is 3 days before day 4), but MUST see it when
    # lookback_days=7 (day 1 is within a 7-day trailing window of day 4).
    df = _txns(
        [
            ("2022-09-01 00:00", "A", "B"),
            ("2022-09-04 00:00", "A", "C"),
        ]
    )
    short_lookback = build_daily_graph_features(df, cycle_max_length=4, lookback_days=2)
    long_lookback = build_daily_graph_features(df, cycle_max_length=4, lookback_days=7)

    assert short_lookback.loc[1, "sender_graph_out_degree"] == 0
    assert long_lookback.loc[1, "sender_graph_out_degree"] == 1


def test_build_daily_graph_features_rejects_missing_columns():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2022-09-01"])})
    with pytest.raises(ValueError, match="missing columns"):
        build_daily_graph_features(df)
