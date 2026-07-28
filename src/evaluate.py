"""Analyst-style evaluation: precision@k, recall-per-typology, false-positive
reduction vs. the rules baseline at equal recall (the headline metric), PR-AUC/ROC-AUC,
and calibration.

Implemented in PLAN.md Phase 5. Deliberately does not report accuracy as a headline —
meaningless at ~0.10% prevalence (see CLAUDE.md's Dataset section) — and every number
here is computed on the model's held-out TEST split only, never train/val.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """Precision within the top-k highest-scored transactions — the metric an analyst
    actually experiences, since they can only review a fixed number of alerts per day.
    """
    order = np.argsort(-y_scores)
    top_k_labels = y_true[order[:k]]
    return float(top_k_labels.sum() / k)


def alerts_for_recall(y_true: np.ndarray, y_scores: np.ndarray, target_recall: float) -> np.ndarray:
    """The smallest top-ranked alert set (as row positions into `y_true`/`y_scores`)
    that achieves at least `target_recall`.

    Works with explicit index sets rather than a score threshold plus `scores >=
    threshold` re-filtering, deliberately: with a threshold, ties at the boundary score
    could pull in more or fewer alerts than intended. Taking `order[:idx + 1]` directly
    pins down an exact alert count with no tie ambiguity.
    """
    n_positives = int(y_true.sum())
    if n_positives == 0:
        raise ValueError("alerts_for_recall: y_true has no positives")

    order = np.argsort(-y_scores)
    y_sorted = y_true[order]
    cum_recall = np.cumsum(y_sorted) / n_positives

    idx = np.searchsorted(cum_recall, target_recall, side="left")
    idx = min(idx, len(cum_recall) - 1)
    return order[: idx + 1]


def false_positive_reduction(model_n_alerts: int, baseline_n_alerts: int) -> float:
    """Fraction fewer alerts the model raises than the baseline, at matched recall —
    the project's headline sentence: "at equal recall, X% fewer alerts than the rules
    baseline."
    """
    return 1 - (model_n_alerts / baseline_n_alerts)


def recall_per_typology(
    y_true: np.ndarray, alert_idx: np.ndarray, pattern_types: pd.Series
) -> pd.DataFrame:
    """For each labelled typology (fan-in, cycle, ...), what fraction of its known
    laundering transactions fall inside `alert_idx` (the alert set from
    `alerts_for_recall`, evaluated at the equal-recall-to-baseline operating point —
    the same threshold behind the headline false-positive-reduction number, so this
    table shows *which* typologies that headline recall is actually coming from).

    `pattern_types` must be a Series aligned by position with `y_true` (same index
    convention as `alerts_for_recall`'s inputs); NaN entries (laundering transactions
    that don't match a named typology block, or legitimate transactions) are excluded.
    """
    flagged = np.zeros(len(y_true), dtype=bool)
    flagged[alert_idx] = True

    pattern_values = pattern_types.to_numpy()
    rows = []
    for typology in sorted(pd.unique(pattern_types.dropna())):
        mask = (pattern_values == typology) & (y_true == 1)
        n = int(mask.sum())
        caught = int((mask & flagged).sum())
        rows.append(
            {
                "typology": typology,
                "n_transactions": n,
                "n_caught": caught,
                "recall": caught / n if n else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("typology").reset_index(drop=True)


def calibration_table(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability check: within each bin of predicted score, what fraction of
    transactions are actually laundering — analysts triage by score, so scores should
    mean roughly what they claim to.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_scores, n_bins=n_bins, strategy="quantile"
    )
    return pd.DataFrame(
        {"mean_predicted_score": mean_predicted, "fraction_positive": fraction_positive}
    )


def pr_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    return average_precision_score(y_true, y_scores), roc_auc_score(y_true, y_scores)


if __name__ == "__main__":
    from src.data_loader import join_pattern_types, load_config, load_patterns, load_transactions
    from src.model import load_xgboost_model, prepare_feature_matrix, time_ordered_split
    from src.rules_baseline import apply_rules_baseline

    config = load_config()
    split_cfg = config["split"]
    k_values = config["evaluation"]["precision_at_k"]

    print("Loading modelling table and trained model...")
    features_df = pd.read_parquet(config["paths"]["modelling_table"])
    _, _, test_df = time_ordered_split(features_df, split_cfg["train_frac"], split_cfg["val_frac"])

    x_test, y_test_series = prepare_feature_matrix(test_df)
    y_test = y_test_series.to_numpy()
    model = load_xgboost_model(config["paths"]["models_dir"])
    test_scores = model.predict_proba(x_test)[:, 1]

    print(f"Test split: {len(test_df):,} rows, {int(y_test.sum())} positives")

    # --- Rules-baseline comparison, restricted to the SAME test-period rows -----------
    # Computed on the FULL dataset first (so structuring/pass-through rules keep their
    # rolling-window historical context right up to the train/val/test boundary), then
    # restricted to the test split's rows by index — not recomputed on an isolated
    # test-only slice, which would wrongly cold-start every rule at the split boundary.
    raw_txns = load_transactions(f"{config['paths']['raw_dir']}/HI-Small_Trans.csv")
    baseline_full = apply_rules_baseline(raw_txns, config)
    baseline_test = baseline_full.loc[test_df.index]

    baseline_test_alerts = int(baseline_test["alert"].sum())
    baseline_test_recall = float(
        (baseline_test["alert"] & baseline_test["is_laundering"].astype(bool)).sum()
        / baseline_test["is_laundering"].sum()
    )
    print(
        f"Rules baseline on test split: {baseline_test_alerts:,} alerts, "
        f"{baseline_test_recall:.4%} recall (this project's full-dataset Phase 2 number "
        f"was 60.6% recall — the test-split-only number is the correct one to match here, "
        f"since it's the same population the model is scored on)"
    )

    # --- Headline: false-positive reduction at equal recall ---------------------------
    model_alert_idx = alerts_for_recall(y_test, test_scores, baseline_test_recall)
    model_n_alerts = len(model_alert_idx)
    fp_reduction = false_positive_reduction(model_n_alerts, baseline_test_alerts)
    print(
        f"\nHEADLINE: at {baseline_test_recall:.1%} recall, the model raises "
        f"{model_n_alerts:,} alerts vs. the rules baseline's {baseline_test_alerts:,} "
        f"— a {fp_reduction:.1%} reduction."
    )

    # --- Recall per typology, at the same operating point ---------------------------
    patterns = load_patterns(f"{config['paths']['raw_dir']}/HI-Small_Patterns.txt")
    txns_with_patterns = join_pattern_types(raw_txns, patterns)
    assert (
        txns_with_patterns["is_laundering"].to_numpy() == raw_txns["is_laundering"].to_numpy()
    ).all()
    test_pattern_types = txns_with_patterns.loc[test_df.index, "pattern_type"]

    typology_table = recall_per_typology(y_test, model_alert_idx, test_pattern_types)
    print("\nRecall per typology (at the equal-recall operating point above):")
    print(typology_table.to_string(index=False))

    # --- PR-AUC / ROC-AUC / precision@k / calibration ---------------------------------
    test_pr_auc, test_roc_auc = pr_roc_auc(y_test, test_scores)
    print(f"\nTest PR-AUC: {test_pr_auc:.4f}, ROC-AUC: {test_roc_auc:.4f}")

    print("\nPrecision@k:")
    for k in k_values:
        print(f"  precision@{k}: {precision_at_k(y_test, test_scores, k):.2%}")

    calib = calibration_table(y_test, test_scores)
    print("\nCalibration (10 quantile bins):")
    print(calib.to_string(index=False))
