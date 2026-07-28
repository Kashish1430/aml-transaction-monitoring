"""SHAP global/local explanations and plain-English per-alert reason codes.

Implemented in PLAN.md Phase 6. This module exists because AML is a regulated domain:
an analyst has to justify why an alert was escalated, and an auditor has to be able to
follow that justification after the fact. A ranked score with no attached reason is not
reviewable, so the deliverable here is not "SHAP plots" but a sentence per alert that a
compliance analyst could paste into a case note.

Three deliberate design choices:

- **SHAP is computed on a bounded sample, not all 1,015,669 test rows.** TreeExplainer
  is exact and fast per row, but the resulting matrix is (n_rows x 53 features) of
  float64 — the full test split would be ~430MB for no analytical gain, since global
  importance converges long before that and per-alert codes are only ever needed for
  rows an analyst actually sees. `select_explanation_sample` takes all positives (so
  recall-side explanations aren't sampled away at 0.1% prevalence) plus the alert set
  plus a seeded negative sample.

- **Reason codes only cite features that pushed the score UP.** A negative SHAP value
  means the feature argued *against* the alert; listing it as a "reason" would be
  actively misleading in a case note. Features are ranked by SHAP contribution and
  filtered to positive contributions only.

- **Two reason-code variants are produced, not one.** Phase 4 established that
  `payment_format_ACH` dominates this model's feature importance because 86.6% of
  labelled laundering in HI-Small uses ACH (see reports/challenges.md). A reason code
  faithful to raw SHAP will therefore often lead with "the payment was made via ACH",
  which is true but near-useless to an analyst — every alert says it. So this module
  exposes both the faithful variant and a behavioural variant that excludes the
  payment-format one-hots (`PAYMENT_FORMAT_FEATURES`) and cites the account/window and
  graph features instead. Neither is hidden; the gap between them is itself a finding
  (see reports/results.md's Phase 6 section).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import shap

from src.model import PAYMENT_FORMATS

# The payment-format one-hot columns, as named by `model.prepare_feature_matrix`. Passed
# as `exclude_features` to get the behavioural reason-code variant described in this
# module's docstring — not deleted from the model, only omitted from that one sentence.
PAYMENT_FORMAT_FEATURES = tuple(f"payment_format_{fmt}" for fmt in PAYMENT_FORMATS)

# Rendered in reason codes wherever a window appears. HI-Small spans only ~17-18 days
# (see CLAUDE.md's Dataset section), so the 30-day window is effectively "all history to
# date" for most transactions — phrased as "30 days" anyway, since that's what the
# feature literally computes and overclaiming precision in an analyst-facing string is
# worse than a slightly loose window label.
_WINDOW_LABELS = {"1": "24h", "7": "7 days", "30": "30 days"}

# Matches the systematic feature-name families produced by src/features.py and
# src/graph_features.py. Ordered most-specific-first: `_distinct_counterparties` and
# `_amount_usd` must be tried before the bare `_count` pattern would otherwise partially
# match a longer name.
_DIRECTION_WORDS = {
    ("out", "verb"): "sent",
    ("in", "verb"): "received",
    ("out", "preposition"): "to",
    ("in", "preposition"): "from",
}


def _window(days: str) -> str:
    return _WINDOW_LABELS.get(days, f"{days} days")


def _money(value: float) -> str:
    return f"${value:,.0f}"


def describe_feature(name: str, value: float) -> str:
    """Render one feature and its raw value as an analyst-readable clause.

    Kept separate from `build_reason_code` so a clause can be unit-tested against a
    known raw value directly — the PLAN.md Phase 6 check is "spot-check flagged alerts'
    reason codes against their raw feature values", which requires the value-to-words
    mapping be independently verifiable rather than only observable inside a finished
    sentence.

    Binary features (`*_graph_in_cycle`, the `payment_format_*` one-hots) are rendered
    according to whether they are set: a zero-valued one-hot that still carries positive
    SHAP means "*not* being this payment format raised the score", and saying so plainly
    beats emitting a clause that implies the opposite. Unrecognised feature names fall
    back to "<name> = <value>" rather than raising — a reason code degrading to a raw
    name is recoverable in production, a crash in the explanation layer is not.
    """
    if name == "amount_paid_usd":
        return f"a {_money(value)} payment"

    if match := re.fullmatch(r"structuring_score_(\d+)d", name):
        return (
            f"{value:,.0f} of the sender's payments in the past {_window(match.group(1))} "
            "sat just under the $10,000 reporting threshold"
        )

    if match := re.fullmatch(r"sender_pass_through_ratio_(\d+)d", name):
        return (
            f"the sender's inflow-to-outflow ratio over the past {_window(match.group(1))} "
            f"is {value:.2f} (near 1.0 means funds pass straight through rather than accumulate)"
        )

    if match := re.fullmatch(r"(sender|receiver)_(in|out)_(\d+)d_distinct_counterparties", name):
        role, direction, days = match.groups()
        verb = _DIRECTION_WORDS[(direction, "verb")]
        prep = _DIRECTION_WORDS[(direction, "preposition")]
        return (
            f"the {role} {verb} {prep} {value:,.0f} distinct counterparties "
            f"in the past {_window(days)}"
        )

    if match := re.fullmatch(r"(sender|receiver)_(in|out)_(\d+)d_amount_usd", name):
        role, direction, days = match.groups()
        verb = _DIRECTION_WORDS[(direction, "verb")]
        return f"the {role} {verb} {_money(value)} in the past {_window(days)}"

    if match := re.fullmatch(r"(sender|receiver)_(in|out)_(\d+)d_count", name):
        role, direction, days = match.groups()
        verb = _DIRECTION_WORDS[(direction, "verb")]
        return f"the {role} {verb} {value:,.0f} transactions in the past {_window(days)}"

    if match := re.fullmatch(r"(sender|receiver)_graph_(in|out)_degree", name):
        role, direction = match.groups()
        verb = _DIRECTION_WORDS[(direction, "verb")]
        prep = _DIRECTION_WORDS[(direction, "preposition")]
        return f"the {role} {verb} {prep} {value:,.0f} distinct accounts in the graph window"

    if match := re.fullmatch(r"(sender|receiver)_graph_fan_(in|out)_score", name):
        role, direction = match.groups()
        shape = (
            "collector (mostly receiving)"
            if direction == "in"
            else "distributor (mostly sending)"
        )
        return f"the {role} is structurally a {shape}, fan-{direction} score {value:.2f}"

    if match := re.fullmatch(r"(sender|receiver)_graph_in_cycle", name):
        role = match.group(1)
        if value:
            return f"the {role} sits on a short directed cycle (funds loop back within a few hops)"
        return f"the {role} does not sit on a short directed cycle"

    if match := re.fullmatch(r"payment_format_(.+)", name):
        fmt = match.group(1)
        return f"the payment was made via {fmt}" if value else f"the payment was not made via {fmt}"

    return f"{name} = {value:,.4g}"


def compute_shap_values(model, x: pd.DataFrame) -> np.ndarray:
    """Exact per-feature SHAP contributions for every row of `x`, shape (len(x), n_features).

    Uses `shap.TreeExplainer`, which computes exact Shapley values for tree ensembles in
    polynomial time — not the sampling-based KernelExplainer, which would be both
    approximate and orders of magnitude slower here. Values are in the model's **margin**
    (log-odds) space, XGBoost's native output space, so they satisfy the additivity
    identity `base_value + shap_values.sum(axis=1) == model.predict(..., output_margin=True)`
    that makes an explanation faithful rather than merely plausible. Use
    `shap_base_value` to obtain that base value — **not** `TreeExplainer.expected_value`,
    which is unreliable here (see that function's docstring and reports/challenges.md's
    Phase 6 entry).

    Column order must match the order the model was trained on; pass a frame built by
    `model.prepare_feature_matrix` (or run it through `align_features` first).
    """
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    # Older SHAP versions return a list of per-class arrays for classifiers; newer ones
    # return a single (n, f) array for binary. Normalize to the positive class either way.
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


def shap_base_value(model, x: pd.DataFrame, shap_values: np.ndarray, tol: float = 1e-3) -> float:
    """The additive intercept for `shap_values`, derived from the model itself, plus a
    runtime check that additivity actually holds.

    Deliberately does NOT read `shap.TreeExplainer.expected_value`, because for XGBoost
    that attribute is **lazily corrected** and is wrong if read too early. On shap 0.45.1
    + xgboost 2.0.3, a freshly constructed `TreeExplainer` reports
    `expected_value == array([logit(base_score)])` (e.g. `[0.3]`), and only on the first
    `shap_values()` call is it silently replaced with the true scalar bias XGBoost's own
    `pred_contribs` reports (e.g. `0.3354011`). The two differ by a constant — small
    enough to look like a rounding artifact, large enough to make every reconstructed
    margin wrong — and nothing warns you. Deriving the bias as
    `margin - shap_values.sum(axis=1)` is immune to that ordering entirely, and asserts
    the property we actually care about instead of trusting a library attribute to mean
    what it says. See investigations/phase6_explainability/ for the reproduction.

    Note that no reason code depends on this value: `build_reason_code` uses only
    per-feature contributions and their ranking, and those are exact (verified
    bit-identical to XGBoost's native `pred_contribs`). This function exists so that
    anything which *does* need the intercept — a per-alert score decomposition in the
    Streamlit app, a SHAP waterfall plot — cannot silently inherit the wrong one.

    Raises if the derived bias is not constant across rows to within `tol`, which would
    mean the contributions and the model's output have genuinely diverged.
    """
    margin = model.predict(x, output_margin=True)
    per_row_bias = margin - shap_values.sum(axis=1)
    spread = float(per_row_bias.max() - per_row_bias.min())
    if spread > tol:
        raise ValueError(
            f"shap_base_value: SHAP additivity violated — implied base value varies by "
            f"{spread:.6g} across rows (tol={tol}); contributions do not reconstruct the "
            f"model's margin output."
        )
    return float(per_row_bias.mean())


def align_features(x: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Reorder/subset `x`'s columns to exactly `feature_names` (the trained model's
    order, as stored in `models/model_metadata.json`). Raises on any missing column
    rather than silently filling zeros — a quietly reordered or zero-filled feature
    matrix produces wrong SHAP values that still look reasonable, which is the worst
    possible failure mode for an explanation layer.
    """
    missing = [name for name in feature_names if name not in x.columns]
    if missing:
        raise ValueError(f"align_features: feature matrix is missing columns {missing}")
    return x.loc[:, list(feature_names)]


def global_importance(shap_values: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:
    """Global feature importance as mean |SHAP| per feature, descending.

    Preferred over XGBoost's built-in `feature_importances_` (gain/split-count) because
    it is in the same units and on the same scale as the per-alert explanations below —
    so the global story and the local story cannot disagree with each other. Also
    reports mean signed SHAP, which distinguishes "this feature matters and pushes
    toward laundering" from "this feature matters and pushes away from it".
    """
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"global_importance: {shap_values.shape[1]} SHAP columns vs. "
            f"{len(feature_names)} feature names"
        )
    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def top_contributors(
    shap_row: np.ndarray,
    feature_names: Sequence[str],
    top_n: int = 3,
    exclude_features: Iterable[str] = (),
) -> list[tuple[str, float]]:
    """The `top_n` features that pushed this row's score UP the most, as
    (feature_name, shap_value) pairs in descending contribution order.

    Positive-only by design (see the module docstring): a feature with negative SHAP
    argued against the alert and is not a reason for it. Returns fewer than `top_n`
    pairs — possibly none — when fewer positive contributors exist, rather than padding
    the list with features that don't support the alert.
    """
    excluded = set(exclude_features)
    candidates = [
        (name, float(value))
        for name, value in zip(feature_names, shap_row, strict=True)
        if value > 0 and name not in excluded
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return candidates[:top_n]


def build_reason_code(
    shap_row: np.ndarray,
    raw_row: pd.Series,
    feature_names: Sequence[str],
    top_n: int = 3,
    exclude_features: Iterable[str] = (),
) -> str:
    """One analyst-readable sentence explaining why this transaction was alerted.

    Composed from the top positive SHAP contributors (`top_contributors`) rendered
    against their own raw feature values (`describe_feature`), so every clause states a
    fact about this specific transaction that can be checked against the modelling
    table — the sentence is generated from the model's actual reasoning, not written to
    sound plausible. Pass `exclude_features=PAYMENT_FORMAT_FEATURES` for the behavioural
    variant.
    """
    contributors = top_contributors(shap_row, feature_names, top_n, exclude_features)
    if not contributors:
        return (
            "Flagged: no individual feature pushed this transaction's score upward "
            "(ranking driven by the model's baseline rate)."
        )
    clauses = [describe_feature(name, float(raw_row[name])) for name, _ in contributors]
    return "Flagged: " + "; ".join(clauses) + "."


def build_reason_codes(
    shap_values: np.ndarray,
    x: pd.DataFrame,
    top_n: int = 3,
    exclude_features: Iterable[str] = (),
) -> pd.Series:
    """Vectorised-caller convenience: a reason code per row of `x`, indexed like `x`.

    Used by scripts/build_demo_artifact.py (PLAN.md Phase 8) to precompute the reason
    codes the deployed Streamlit app displays, since the app never runs SHAP live (see
    CLAUDE.md's deployment model).
    """
    feature_names = list(x.columns)
    codes = [
        build_reason_code(shap_values[i], x.iloc[i], feature_names, top_n, exclude_features)
        for i in range(len(x))
    ]
    return pd.Series(codes, index=x.index, name="reason_code")


def select_explanation_sample(
    y_true: np.ndarray,
    alert_idx: np.ndarray,
    n_background_negatives: int,
    seed: int,
) -> np.ndarray:
    """Row positions to compute SHAP over: every positive, every alerted row, plus a
    seeded random sample of the remaining rows.

    All positives are kept unconditionally because at ~0.10% prevalence a uniform sample
    of any tractable size would contain almost none of them, leaving the global
    importance table describing only what the model thinks about legitimate traffic. The
    negative sample supplies the background distribution that makes mean |SHAP| a
    population statistic rather than an alerts-only one.

    Returns sorted unique positions, so downstream `.iloc[...]` slices stay in the
    original time order of the split (the alert queue is re-sorted by score later; SHAP
    itself is order-independent, but a deterministic, time-ordered sample is easier to
    reason about and to diff between runs).
    """
    positives = np.flatnonzero(y_true == 1)
    selected = np.union1d(positives, np.asarray(alert_idx, dtype=int))

    remaining = np.setdiff1d(np.arange(len(y_true)), selected, assume_unique=False)
    n_draw = min(n_background_negatives, len(remaining))
    if n_draw:
        rng = np.random.default_rng(seed)
        selected = np.union1d(selected, rng.choice(remaining, size=n_draw, replace=False))
    return np.sort(selected)


if __name__ == "__main__":
    from src.data_loader import load_config
    from src.model import (
        load_metadata,
        load_xgboost_model,
        prepare_feature_matrix,
        time_ordered_split,
    )

    config = load_config()
    split_cfg = config["split"]
    seed = config["seed"]

    print("Loading modelling table and trained model...")
    features_df = pd.read_parquet(config["paths"]["modelling_table"])
    _, _, test_df = time_ordered_split(features_df, split_cfg["train_frac"], split_cfg["val_frac"])
    del features_df

    x_test, y_test_series = prepare_feature_matrix(test_df)
    y_test = y_test_series.to_numpy()
    model = load_xgboost_model(config["paths"]["models_dir"])
    x_test = align_features(x_test, load_metadata(config["paths"]["models_dir"])["feature_names"])
    test_scores = model.predict_proba(x_test)[:, 1]
    print(f"Test split: {len(x_test):,} rows, {int(y_test.sum())} positives")

    # Explanation sample: all positives + the top-5,000 ranked alerts + 20,000 background
    # negatives. Deliberately NOT the Phase 5 equal-recall alert set (10,011 rows), which
    # would require recomputing the rules baseline over the full dataset just to fix an
    # operating point that global importance is insensitive to anyway.
    top_alert_idx = np.argsort(-test_scores)[:5000]
    sample_idx = select_explanation_sample(y_test, top_alert_idx, 20_000, seed)
    x_sample = x_test.iloc[sample_idx]
    print(f"Explaining {len(x_sample):,} rows ({int(y_test[sample_idx].sum())} positives)...")

    shap_values = compute_shap_values(model, x_sample)

    importance = global_importance(shap_values, list(x_sample.columns))
    print("\nGlobal importance (top 15 by mean |SHAP|):")
    print(importance.head(15).to_string(index=False))

    # --- PLAN.md Phase 6 check: spot-check flagged alerts against raw feature values ---
    sample_positions = {position: i for i, position in enumerate(sample_idx)}
    print("\nReason codes for the 5 highest-scored test transactions:")
    for rank, position in enumerate(np.argsort(-test_scores)[:5], start=1):
        i = sample_positions[position]
        row = x_sample.iloc[i]
        faithful = build_reason_code(shap_values[i], row, list(x_sample.columns))
        behavioural = build_reason_code(
            shap_values[i], row, list(x_sample.columns), exclude_features=PAYMENT_FORMAT_FEATURES
        )
        print(f"\n  #{rank}  score={test_scores[position]:.4f}  is_laundering={y_test[position]}")
        print(f"      faithful:    {faithful}")
        print(f"      behavioural: {behavioural}")

    # --- How often does the ACH one-hot lead the faithful reason code? -----------------
    # Phase 4 found payment_format_ACH dominates importance; this quantifies what that
    # does to analyst-facing explanations specifically (see reports/results.md Phase 6).
    alert_positions = [sample_positions[p] for p in top_alert_idx]
    leads = [
        top_contributors(shap_values[i], list(x_sample.columns), top_n=1) for i in alert_positions
    ]
    ach_led = sum(1 for pair in leads if pair and pair[0][0] == "payment_format_ACH")
    print(
        f"\npayment_format_ACH is the single largest positive contributor for "
        f"{ach_led:,}/{len(leads):,} ({ach_led / len(leads):.1%}) of the top-5,000 alerts."
    )
