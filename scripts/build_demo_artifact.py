"""Build app/data/demo_alerts.parquet + demo_metrics.json from the trained pipeline.

Implemented in PLAN.md Phase 8. This is the seam between the offline pipeline and the
deployed Streamlit app: everything expensive (feature engineering, scoring, the rules
baseline, SHAP) happens here, once, locally, against the full HI-Small dataset. The app
reads the two artifacts this writes and computes nothing (see CLAUDE.md's deployment
model — Streamlit Community Cloud gives ~1 CPU / ~1GB RAM, which is not enough to load
the 728MB modelling table, let alone score it).

Four decisions in here are load-bearing and deliberately not "simplified":

- **The demo table is a sample, but every rank in it is a true rank.** Scores, ranks and
  the equal-recall alert queue are all computed over the *complete* 1,015,669-row test
  split first; only then are rows sampled for shipping. So "rank 1" in the app is
  genuinely the highest-scored transaction in the test split, not the highest-scored
  transaction that happened to survive sampling. `n_test_rows` in the metrics JSON is
  what the ranks are out of.

- **The whole equal-recall alert queue ships, unsampled.** That queue *is* the headline
  result (10,011 alerts vs. the baseline's 297,564), so sampling it would leave the app
  unable to show the thing the project claims. Negatives outside it are sampled.

- **The 2022-09-11+ tail is flagged, not filtered and not balanced away.** Phase 7
  established it is a different population — 0.109% of test rows but 36.4% of its
  positives (see reports/results.md). Because the sample takes all positives, that tail
  is inevitably over-represented relative to its true share. The response is to measure
  it, expose it as an `is_tail_population` column, and record both the true and sampled
  shares in the metrics JSON, so the app can filter it and the README can state it. The
  headline is also re-derived without the tail here, so neither number is taken on trust
  from Phase 7's writeup.

- **Both reason-code variants ship.** Phase 6 found the faithful variant leads with
  `payment_format_ACH` on 99.86% of the queue, making it near-useless for triage; the
  behavioural variant excludes the payment-format one-hots from the *sentence only*.
  Dropping either would hide the finding (CLAUDE.md's Phase 6 note).

Run from the repo root:  python -m scripts.build_demo_artifact
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Columns carried through to the app for display/filtering, in the order an analyst
# reads them. Kept explicit rather than "everything not a feature" so adding a column
# upstream can't silently change what the deployed app shows.
DISPLAY_COLUMNS = [
    "txn_id",
    "rank",
    "model_score",
    "in_alert_queue",
    "timestamp",
    "from_bank",
    "from_account_key",
    "to_bank",
    "to_account_key",
    "amount_paid",
    "payment_currency",
    "amount_paid_usd",
    "payment_format",
    "is_laundering",
    "pattern_type",
    "is_tail_population",
    "reason_code_faithful",
    "reason_code_behavioural",
]


def select_demo_rows(
    y_true: np.ndarray,
    scores: np.ndarray,
    alert_idx: np.ndarray,
    target_rows: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Choose which test-split row positions ship in the demo artifact.

    Three strata, unioned:

    1. **Every positive.** At 0.18% test prevalence a uniform sample of 30,000 rows would
       contain ~53 laundering transactions, which is too few to demonstrate recall per
       typology (8 typologies) or to let a reviewer click through real examples.
    2. **Every row in the equal-recall alert queue.** This is the artifact the headline
       number describes; it ships whole (see module docstring).
    3. **Negatives sampled uniformly across score deciles** to fill the remainder. Decile
       stratification rather than a flat random draw so the shipped table spans the whole
       score range — a flat draw from a distribution this skewed returns almost nothing
       above the alert threshold, and the app's "browse below the cut-off" view would
       have no rows to show. Deciles are cut on the *remaining pool* after strata 1-2.

    Returns the selected positions plus the per-stratum positions, so the caller can
    record the sample's actual composition in the metrics JSON instead of asserting it.
    Selected positions are sorted, keeping the artifact in time order on disk.
    """
    positives = np.flatnonzero(y_true == 1)
    queue = np.asarray(alert_idx, dtype=int)
    core = np.union1d(positives, queue)

    if len(core) >= target_rows:
        # Never drop stratum 1 or 2 to hit a row target — the target is a size budget for
        # the *filler*, not a cap on the result. Returning more rows than asked and saying
        # so beats shipping a queue with holes in it.
        return {
            "selected": core,
            "positives": positives,
            "queue": queue,
            "filler": np.array([], dtype=int),
        }

    pool = np.setdiff1d(np.arange(len(y_true)), core, assume_unique=False)
    n_draw = min(target_rows - len(core), len(pool))

    pool_scores = scores[pool]
    # Rank-based deciles, not value-based: the score distribution is dense enough near
    # zero that np.quantile edges collide there and produce empty bins.
    decile = np.minimum((np.argsort(np.argsort(pool_scores)) * 10) // len(pool), 9)

    rng = np.random.default_rng(seed)
    per_decile = n_draw // 10
    picks = []
    for d in range(10):
        members = pool[decile == d]
        take = min(per_decile, len(members))
        if take:
            picks.append(rng.choice(members, size=take, replace=False))
    filler = np.concatenate(picks) if picks else np.array([], dtype=int)

    # Integer division above leaves up to 9 rows unallocated; top up from whatever's left
    # so the artifact lands on the configured size rather than a few rows under it.
    shortfall = n_draw - len(filler)
    if shortfall > 0:
        leftover = np.setdiff1d(pool, filler, assume_unique=False)
        if len(leftover):
            filler = np.union1d(
                filler, rng.choice(leftover, size=min(shortfall, len(leftover)), replace=False)
            )

    return {
        "selected": np.sort(np.union1d(core, filler)),
        "positives": positives,
        "queue": queue,
        "filler": filler,
    }


def downsample_pr_curve(
    precision: np.ndarray, recall: np.ndarray, n_points: int = 300
) -> list[dict[str, float]]:
    """Thin `sklearn.precision_recall_curve`'s ~1M points down to something a browser can
    plot, sampling evenly along the *recall* axis rather than by array position.

    Position-sampling would spend most of its budget in the flat, uninteresting
    high-precision/near-zero-recall corner where consecutive points are nearly identical,
    and would render the operating region the project actually reports as a handful of
    points. The curve's endpoints are always kept.
    """
    if len(recall) <= n_points:
        keep = np.arange(len(recall))
    else:
        targets = np.linspace(recall.min(), recall.max(), n_points)
        keep = np.unique(np.searchsorted(recall[::-1], targets))
        keep = np.clip(len(recall) - 1 - keep, 0, len(recall) - 1)
        keep = np.unique(np.concatenate([[0], keep, [len(recall) - 1]]))
    return [
        {"recall": float(recall[i]), "precision": float(precision[i])} for i in sorted(keep)
    ]


def records(df: pd.DataFrame) -> list[dict]:
    """A DataFrame as JSON-ready records, routed through pandas' own JSON writer.

    `DataFrame.to_dict` leaves numpy scalars in place: `np.float64` survives `json.dump`
    (it subclasses `float`), but `np.int64` does not, and with `default=str` it serialises
    as a *quoted string* — so a typology count of 50 silently ships to the app as "50" and
    breaks arithmetic on the other side without raising anywhere. Going via `to_json` also
    turns NaN into `null` rather than the bare `NaN` token, which is invalid JSON and which
    `json.dump` would otherwise emit for the thin PSI slices.
    """
    return json.loads(df.to_json(orient="records"))


def tail_mask(timestamps: pd.Series, tail_start: str) -> np.ndarray:
    """Boolean mask for the anomalous late-dataset tail Phase 7 characterised (daily
    volume collapses from 654,467 rows to 11, laundering rate goes to 59.12%, payment
    format to 100% ACH). Its start date lives in config.yaml, not here.
    """
    return (pd.to_datetime(timestamps) >= pd.Timestamp(tail_start)).to_numpy()


def main() -> None:  # noqa: C901 - linear build script, split only by section comments
    from src.data_loader import (
        join_pattern_types,
        load_config,
        load_patterns,
        load_transactions,
    )
    from src.evaluate import (
        alerts_for_recall,
        calibration_table,
        false_positive_reduction,
        pr_roc_auc,
        precision_at_k,
        recall_per_typology,
    )
    from src.explain import (
        PAYMENT_FORMAT_FEATURES,
        align_features,
        build_reason_codes,
        compute_shap_values,
        global_importance,
        shap_base_value,
    )
    from src.model import (
        load_metadata,
        load_xgboost_model,
        prepare_feature_matrix,
        time_ordered_split,
    )
    from src.monitoring import psi_over_time
    from src.rules_baseline import apply_rules_baseline

    config = load_config()
    paths = config["paths"]
    split_cfg = config["split"]
    mon_cfg = config["monitoring"]
    demo_cfg = config["demo_artifact"]
    seed = config["seed"]

    # --- Score the full test split ----------------------------------------------------
    print(f"Loading modelling table from {paths['modelling_table']} ...")
    features_df = pd.read_parquet(paths["modelling_table"])
    train_df, val_df, test_df = time_ordered_split(
        features_df, split_cfg["train_frac"], split_cfg["val_frac"]
    )
    del features_df

    model = load_xgboost_model(paths["models_dir"])
    metadata = load_metadata(paths["models_dir"])
    feature_names = metadata["feature_names"]

    # Train/val scores are needed only as the PSI reference and series below; matrices are
    # released as soon as they're scored to keep peak memory near the test split's alone.
    x_train, _ = prepare_feature_matrix(train_df)
    train_scores = model.predict_proba(align_features(x_train, feature_names))[:, 1]
    del x_train, train_df

    x_val, _ = prepare_feature_matrix(val_df)
    val_scores = model.predict_proba(align_features(x_val, feature_names))[:, 1]
    val_timestamps = val_df["timestamp"].reset_index(drop=True)
    del x_val, val_df

    x_test, y_test_series = prepare_feature_matrix(test_df)
    x_test = align_features(x_test, feature_names)
    y_test = y_test_series.to_numpy()
    test_scores = model.predict_proba(x_test)[:, 1]
    print(f"Test split: {len(x_test):,} rows, {int(y_test.sum()):,} positives")

    # Rank over the COMPLETE test split, before any sampling (module docstring).
    rank = np.empty(len(test_scores), dtype=np.int32)
    rank[np.argsort(-test_scores, kind="stable")] = np.arange(1, len(test_scores) + 1)

    # --- Rules baseline on the same population, for the headline ----------------------
    # Same approach as src/evaluate.py: run the rules over the FULL dataset so the
    # structuring/pass-through rolling windows keep their history across the split
    # boundary, then restrict to the test rows by index. Recomputing on a test-only slice
    # would cold-start every rule and understate the baseline.
    print("Loading raw transactions for the rules baseline and typology labels ...")
    raw_txns = load_transactions(f"{paths['raw_dir']}/HI-Small_Trans.csv")
    baseline_test = apply_rules_baseline(raw_txns, config).loc[test_df.index]
    baseline_alerts = int(baseline_test["alert"].sum())
    baseline_recall = float(
        (baseline_test["alert"] & baseline_test["is_laundering"].astype(bool)).sum()
        / baseline_test["is_laundering"].sum()
    )
    print(f"Rules baseline on test split: {baseline_alerts:,} alerts, {baseline_recall:.4%} recall")

    alert_idx = alerts_for_recall(y_test, test_scores, baseline_recall)
    fp_reduction = false_positive_reduction(len(alert_idx), baseline_alerts)
    print(
        f"HEADLINE: at {baseline_recall:.1%} recall the model raises {len(alert_idx):,} alerts "
        f"vs. {baseline_alerts:,} — a {fp_reduction:.1%} reduction."
    )

    # --- Same headline with the anomalous tail removed --------------------------------
    # Phase 7 reported 96.2% here. Recomputed rather than quoted, per CLAUDE.md's rule
    # that every reported number must be regenerable from code.
    is_tail = tail_mask(test_df["timestamp"], demo_cfg["tail_start_date"])
    keep = ~is_tail
    baseline_alerts_ex = int((baseline_test["alert"].to_numpy() & keep).sum())
    baseline_recall_ex = float(
        (baseline_test["alert"].to_numpy() & baseline_test["is_laundering"].astype(bool).to_numpy()
         & keep).sum()
        / (baseline_test["is_laundering"].to_numpy() & keep).sum()
    )
    alert_idx_ex = alerts_for_recall(y_test[keep], test_scores[keep], baseline_recall_ex)
    fp_reduction_ex = false_positive_reduction(len(alert_idx_ex), baseline_alerts_ex)
    print(
        f"Excluding the {demo_cfg['tail_start_date']}+ tail: {fp_reduction_ex:.1%} reduction "
        f"({len(alert_idx_ex):,} vs. {baseline_alerts_ex:,} alerts at {baseline_recall_ex:.1%} "
        f"recall)"
    )

    # --- Typology labels ---------------------------------------------------------------
    patterns = load_patterns(f"{paths['raw_dir']}/HI-Small_Patterns.txt")
    txns_with_patterns = join_pattern_types(raw_txns, patterns)
    test_pattern_types = txns_with_patterns.loc[test_df.index, "pattern_type"]
    typology_table = recall_per_typology(y_test, alert_idx, test_pattern_types)
    print(f"\nRecall per typology at the equal-recall operating point:\n{typology_table}")

    # --- Choose the demo rows ----------------------------------------------------------
    sample = select_demo_rows(y_test, test_scores, alert_idx, demo_cfg["target_rows"], seed)
    sel = sample["selected"]
    print(
        f"\nDemo sample: {len(sel):,} rows "
        f"({len(sample['positives']):,} positives, {len(sample['queue']):,} queue, "
        f"{len(sample['filler']):,} stratified negatives)"
    )

    # --- SHAP over the sampled rows only ------------------------------------------------
    x_sample = x_test.iloc[sel]
    del x_test
    print(f"Computing SHAP over {len(x_sample):,} rows x {len(feature_names)} features ...")
    shap_values = compute_shap_values(model, x_sample)
    base_value = shap_base_value(model, x_sample, shap_values)
    print(f"SHAP base value (derived, not TreeExplainer.expected_value): {base_value:.6f}")

    importance = global_importance(shap_values, feature_names)
    faithful = build_reason_codes(shap_values, x_sample)
    behavioural = build_reason_codes(
        shap_values, x_sample, exclude_features=PAYMENT_FORMAT_FEATURES
    )

    # --- Assemble the artifact table ----------------------------------------------------
    in_queue = np.zeros(len(y_test), dtype=bool)
    in_queue[alert_idx] = True
    raw_test = raw_txns.loc[test_df.index]

    demo = pd.DataFrame(
        {
            "txn_id": test_df.index.to_numpy()[sel],
            "rank": rank[sel],
            "model_score": test_scores[sel].astype("float32"),
            "in_alert_queue": in_queue[sel],
            "timestamp": test_df["timestamp"].to_numpy()[sel],
            "from_bank": raw_test["from_bank"].to_numpy()[sel],
            "from_account_key": test_df["from_account_key"].to_numpy()[sel],
            "to_bank": raw_test["to_bank"].to_numpy()[sel],
            "to_account_key": test_df["to_account_key"].to_numpy()[sel],
            "amount_paid": raw_test["amount_paid"].to_numpy()[sel].astype("float32"),
            "payment_currency": raw_test["payment_currency"].to_numpy()[sel],
            "amount_paid_usd": test_df["amount_paid_usd"].to_numpy()[sel].astype("float32"),
            "payment_format": test_df["payment_format"].to_numpy()[sel],
            "is_laundering": y_test[sel].astype("int8"),
            "pattern_type": test_pattern_types.to_numpy()[sel],
            "is_tail_population": is_tail[sel],
            "reason_code_faithful": faithful.to_numpy(),
            "reason_code_behavioural": behavioural.to_numpy(),
        }
    )[DISPLAY_COLUMNS]

    # Raw feature values and their SHAP contributions, so the app can draw a real
    # waterfall (all 54 features, not just the top 3 the sentence cites) without shap
    # installed. float32 throughout: these feed a chart, not a reconciliation.
    #
    # Both blocks are prefixed, giving the app one unambiguous contract — feature `f` has
    # value `feat_f` and contribution `shap_f`. Without the prefix the raw block collides
    # with the display columns (`amount_paid_usd` is both a model feature and something an
    # analyst reads directly), and a bare `.astype("category")` or column lookup downstream
    # would then silently hit whichever duplicate came first.
    features_block = x_sample.reset_index(drop=True).astype("float32")
    features_block.columns = [f"feat_{name}" for name in feature_names]
    shap_block = pd.DataFrame(
        shap_values.astype("float32"),
        columns=[f"shap_{name}" for name in feature_names],
    )
    demo = pd.concat([demo, features_block, shap_block], axis=1)

    # Low-cardinality strings are dictionary-encoded rather than left as object columns —
    # the two reason-code variants are highly repetitive (Phase 6: only 4 distinct leading
    # features across the queue), and this is most of why the artifact fits the budget.
    for col in [
        "payment_currency",
        "payment_format",
        "pattern_type",
        "reason_code_faithful",
        "reason_code_behavioural",
    ]:
        demo[col] = demo[col].astype("category")

    artifact_path = Path(paths["demo_artifact"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    demo.to_parquet(artifact_path, compression="snappy", index=False)
    size_mb = artifact_path.stat().st_size / 1e6
    print(f"\nWrote {artifact_path} — {len(demo):,} rows x {demo.shape[1]} cols, {size_mb:.1f} MB")
    if size_mb > 20:
        print(
            f"  WARNING: {size_mb:.1f} MB exceeds PLAN.md Phase 8's 20MB target. Reduce "
            f"demo_artifact.target_rows in config.yaml, or ship only the top-N SHAP columns."
        )

    # --- Metrics for the Model Performance tab -------------------------------------------
    pr_auc, roc_auc = pr_roc_auc(y_test, test_scores)
    precision, recall_curve, _ = precision_recall_curve(y_test, test_scores)
    psi_series = psi_over_time(
        train_scores,
        np.concatenate([val_scores, test_scores]),
        pd.concat([val_timestamps, test_df["timestamp"].reset_index(drop=True)], ignore_index=True),
        freq=mon_cfg["time_slice_freq"],
        n_bins=mon_cfg["psi_bins"],
        epsilon=float(mon_cfg["psi_epsilon"]),
        min_slice_size=mon_cfg["min_slice_size"],
        thresholds=mon_cfg["psi_thresholds"],
    )

    n_tail_test = int(is_tail.sum())
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "IBM Transactions for AML (HI-Small)",
            "total_rows": int(
                metadata["train_rows"] + metadata["val_rows"] + metadata["test_rows"]
            ),
            "note": "Synthetic dataset. See README caveats.",
        },
        "test_split": {
            "n_rows": len(y_test),
            "n_positives": int(y_test.sum()),
            "prevalence": float(y_test.mean()),
            "start": str(pd.Timestamp(test_df["timestamp"].min())),
            "end": str(pd.Timestamp(test_df["timestamp"].max())),
        },
        "headline": {
            "matched_recall": baseline_recall,
            "model_alerts": int(len(alert_idx)),
            "baseline_alerts": baseline_alerts,
            "fp_reduction": float(fp_reduction),
            "sentence": (
                f"At equal recall ({baseline_recall:.1%}), the model raises "
                f"{fp_reduction:.1%} fewer alerts than the rules baseline "
                f"({len(alert_idx):,} vs. {baseline_alerts:,}) on the held-out test split."
            ),
        },
        "headline_excluding_tail": {
            "tail_start_date": demo_cfg["tail_start_date"],
            "matched_recall": baseline_recall_ex,
            "model_alerts": int(len(alert_idx_ex)),
            "baseline_alerts": baseline_alerts_ex,
            "fp_reduction": float(fp_reduction_ex),
            "note": (
                "The 2022-09-11+ tail is a different population (Phase 7): "
                f"{n_tail_test:,} of {len(y_test):,} test rows "
                f"({n_tail_test / len(y_test):.3%}) holding "
                f"{int(y_test[is_tail].sum()) / int(y_test.sum()):.1%} of test positives. "
                "The headline does not depend on it."
            ),
        },
        "auc": {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)},
        "precision_at_k": {
            str(k): float(precision_at_k(y_test, test_scores, k))
            for k in config["evaluation"]["precision_at_k"]
        },
        "recall_per_typology": records(typology_table),
        "pr_curve": downsample_pr_curve(precision, recall_curve),
        "calibration": records(calibration_table(y_test, test_scores)),
        "psi_over_time": records(psi_series),
        "psi_thresholds": mon_cfg["psi_thresholds"],
        "shap": {
            "base_value": base_value,
            "global_importance": records(importance.head(20)),
        },
        "demo_sample": {
            "n_rows": len(demo),
            "n_positives": int(demo["is_laundering"].sum()),
            "n_in_alert_queue": int(demo["in_alert_queue"].sum()),
            "n_stratified_negatives": int(len(sample["filler"])),
            "ranks_are_out_of": len(y_test),
            "tail_share_in_sample": float(demo["is_tail_population"].mean()),
            "tail_share_in_test_split": float(is_tail.mean()),
            "note": (
                "A stratified sample of the test split, not the whole thing. Ranks and "
                "scores are computed over all "
                f"{len(y_test):,} test rows before sampling, so they are true ranks. "
                "All positives and the entire equal-recall alert queue are included, "
                "which over-represents the anomalous tail relative to its true share "
                "(see tail_share_* above) — flagged per row via is_tail_population."
            ),
        },
        "model": {
            "type": "XGBoost (hist), early-stopped on validation PR-AUC",
            "scale_pos_weight": metadata["scale_pos_weight"],
            "best_iteration": metadata["best_iteration"],
            "n_features": len(feature_names),
        },
    }

    metrics_path = Path(paths["demo_metrics"])
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Wrote {metrics_path} — {metrics_path.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
