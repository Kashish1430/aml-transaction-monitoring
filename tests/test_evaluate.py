"""Evaluation-metric checks on hand-built synthetic cases (PLAN.md Phase 5)."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    alerts_for_recall,
    false_positive_reduction,
    pr_roc_auc,
    precision_at_k,
    recall_per_typology,
)


def test_precision_at_k_counts_hits_in_top_k_by_score():
    y_true = np.array([1, 0, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])  # top-3 by score: indices 0,1,2 -> labels 1,0,1

    assert precision_at_k(y_true, y_scores, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(y_true, y_scores, k=1) == pytest.approx(1.0)


def test_alerts_for_recall_returns_exact_count_no_ties():
    # 4 positives total, at positions 0, 2, 4, 5 (scores strictly descending by
    # position). The 3rd positive lands at position 4, so hitting 75% recall (3/4)
    # requires exactly the top 5 alerts (positions 0-4) -- not 6.
    y_true = np.array([1, 0, 1, 0, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    alert_idx = alerts_for_recall(y_true, y_scores, target_recall=0.75)
    assert len(alert_idx) == 5
    assert set(y_true[alert_idx]) <= {0, 1}


def test_alerts_for_recall_finds_minimal_set_for_partial_recall():
    y_true = np.array([1, 0, 0, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6])  # positives ranked 1st and 4th

    # 50% recall is satisfied by just the top-1 alert (catches the first positive).
    alert_idx = alerts_for_recall(y_true, y_scores, target_recall=0.5)
    assert len(alert_idx) == 1
    assert alert_idx[0] == 0


def test_alerts_for_recall_rejects_no_positives():
    with pytest.raises(ValueError, match="no positives"):
        alerts_for_recall(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), target_recall=0.5)


def test_false_positive_reduction_matches_manual_calculation():
    # Model raises half as many alerts as the baseline at matched recall -> 50% reduction.
    reduction_half = false_positive_reduction(model_n_alerts=500, baseline_n_alerts=1000)
    reduction_none = false_positive_reduction(model_n_alerts=1000, baseline_n_alerts=1000)
    assert reduction_half == pytest.approx(0.5)
    assert reduction_none == pytest.approx(0.0)


def test_recall_per_typology_breaks_down_by_pattern():
    # 4 laundering transactions: 2 FAN-OUT (1 caught), 2 CYCLE (2 caught). One
    # legitimate transaction (is_laundering=0) with a pattern_type must never count.
    y_true = np.array([1, 1, 1, 1, 0])
    pattern_types = pd.Series(["FAN-OUT", "FAN-OUT", "CYCLE", "CYCLE", "CYCLE"])
    alert_idx = np.array([0, 2, 3])  # catches: FAN-OUT #1, both CYCLEs

    table = recall_per_typology(y_true, alert_idx, pattern_types)

    fan_out = table[table["typology"] == "FAN-OUT"].iloc[0]
    cycle = table[table["typology"] == "CYCLE"].iloc[0]
    assert fan_out["n_transactions"] == 2
    assert fan_out["n_caught"] == 1
    assert fan_out["recall"] == pytest.approx(0.5)
    assert cycle["n_transactions"] == 2
    assert cycle["n_caught"] == 2
    assert cycle["recall"] == pytest.approx(1.0)


def test_recall_per_typology_ignores_nan_pattern_types():
    y_true = np.array([1, 1])
    pattern_types = pd.Series(["FAN-OUT", np.nan])
    alert_idx = np.array([0, 1])

    table = recall_per_typology(y_true, alert_idx, pattern_types)
    assert list(table["typology"]) == ["FAN-OUT"]


def test_pr_roc_auc_perfect_separation_scores_near_one():
    y_true = np.array([0, 0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    pr_auc, roc_auc = pr_roc_auc(y_true, y_scores)
    assert pr_auc == pytest.approx(1.0)
    assert roc_auc == pytest.approx(1.0)
