"""Tests for the demo-artifact sampling/serialisation helpers (PLAN.md Phase 8).

`main()` is an I/O-bound orchestration over the full 5M-row dataset and the trained
model, so it isn't unit-tested here; what *is* tested is the logic that decides what the
deployed app gets to see, because a silent bug there produces an artifact that looks fine
and misrepresents the result — the one failure mode this project can least afford.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_demo_artifact import (
    downsample_pr_curve,
    records,
    select_demo_rows,
    tail_mask,
)


@pytest.fixture
def imbalanced_split():
    """20,000 rows at ~0.2% prevalence with scores correlated to the label — a miniature
    of the real test split's shape, so sampling behaviour is exercised under the skew it
    actually has to handle rather than on uniform toy data.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, size=40, replace=False)] = 1
    scores = rng.beta(1.5, 40, size=n) + y * 0.5
    return y, np.clip(scores, 0, 1)


def test_all_positives_and_the_whole_queue_are_kept(imbalanced_split):
    y, scores = imbalanced_split
    alert_idx = np.argsort(-scores)[:500]

    sample = select_demo_rows(y, scores, alert_idx, target_rows=5_000, seed=42)
    selected = set(sample["selected"].tolist())

    # Stratum 1 and 2 are unconditional: at 0.2% prevalence a sampled-away positive is
    # invisible in aggregate but breaks recall-per-typology in the app.
    assert set(np.flatnonzero(y == 1).tolist()) <= selected
    assert set(alert_idx.tolist()) <= selected


def test_hits_the_target_row_count(imbalanced_split):
    y, scores = imbalanced_split
    alert_idx = np.argsort(-scores)[:500]

    sample = select_demo_rows(y, scores, alert_idx, target_rows=5_000, seed=42)

    # Exactly on target, not merely close: integer division across 10 deciles leaves a
    # remainder that the top-up branch exists to absorb.
    assert len(sample["selected"]) == 5_000
    assert len(np.unique(sample["selected"])) == 5_000


def test_target_smaller_than_the_queue_returns_the_queue_not_a_truncation(imbalanced_split):
    y, scores = imbalanced_split
    alert_idx = np.argsort(-scores)[:500]

    sample = select_demo_rows(y, scores, alert_idx, target_rows=100, seed=42)

    # The size target is a budget for filler, never a cap that silently drops alerts.
    assert len(sample["selected"]) > 100
    assert set(alert_idx.tolist()) <= set(sample["selected"].tolist())
    assert len(sample["filler"]) == 0


def test_filler_spans_the_whole_score_range(imbalanced_split):
    y, scores = imbalanced_split
    alert_idx = np.argsort(-scores)[:500]

    sample = select_demo_rows(y, scores, alert_idx, target_rows=5_000, seed=42)
    filler_scores = scores[sample["filler"]]

    # The point of decile stratification: a flat random draw from this skew returns
    # almost nothing in the upper score range, leaving the app's below-the-cutoff view
    # empty. Compare against the pool the filler was actually drawn from.
    pool_max = scores[np.setdiff1d(np.arange(len(y)), sample["selected"])].max(initial=0)
    assert filler_scores.max() >= 0.5 * max(pool_max, filler_scores.max())
    assert filler_scores.min() <= np.quantile(scores, 0.2)


def test_selection_is_deterministic_and_seed_sensitive(imbalanced_split):
    y, scores = imbalanced_split
    alert_idx = np.argsort(-scores)[:500]

    a = select_demo_rows(y, scores, alert_idx, target_rows=5_000, seed=42)["selected"]
    b = select_demo_rows(y, scores, alert_idx, target_rows=5_000, seed=42)["selected"]
    c = select_demo_rows(y, scores, alert_idx, target_rows=5_000, seed=7)["selected"]

    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_selected_positions_are_sorted(imbalanced_split):
    y, scores = imbalanced_split
    sample = select_demo_rows(y, scores, np.argsort(-scores)[:500], target_rows=5_000, seed=42)

    # Keeps the artifact in time order on disk, so the parquet reads sensibly without
    # the app having to re-sort a 30k table on every page load.
    assert np.all(np.diff(sample["selected"]) > 0)


def test_tail_mask_splits_on_the_configured_date():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2022-09-10 23:59",
                "2022-09-11 00:00",
                "2022-09-11 12:00",
                "2022-09-09 08:00",
            ]
        )
    )
    assert tail_mask(timestamps, "2022-09-11").tolist() == [False, True, True, False]


def test_pr_curve_downsample_keeps_endpoints_and_recall_coverage():
    # sklearn returns recall descending; the helper must cope with that ordering.
    recall = np.linspace(1.0, 0.0, 50_000)
    precision = np.linspace(0.001, 1.0, 50_000)

    points = downsample_pr_curve(precision, recall, n_points=300)
    recalls = [p["recall"] for p in points]

    assert len(points) <= 320
    assert recalls[0] == pytest.approx(1.0)
    assert recalls[-1] == pytest.approx(0.0)
    # Even coverage along recall, not clustered in the flat high-precision corner: every
    # decile of the recall axis must be represented.
    assert len(set(np.floor(np.array(recalls) * 10).astype(int))) >= 10


def test_records_keeps_integer_counts_as_numbers():
    # The failure this guards against is silent: json.dump(default=str) turns np.int64
    # into a quoted string, so recall_per_typology's counts reach the app as "50" and
    # nothing raises until someone does arithmetic on them in the browser.
    df = pd.DataFrame({"typology": ["FAN-IN"], "n_transactions": [137], "recall": [0.8978]})
    assert df["n_transactions"].dtype == np.int64

    row = records(df)[0]

    assert row["n_transactions"] == 137
    assert isinstance(row["n_transactions"], int)
    assert not isinstance(row["n_transactions"], str)


def test_records_emits_null_not_nan_for_thin_psi_slices():
    # psi_over_time reports under-sized slices as NaN, which json.dump would write as the
    # bare token NaN — accepted by Python, rejected by a browser's JSON.parse.
    df = pd.DataFrame({"period": ["2022-09-14"], "psi": [np.nan], "band": ["insufficient_data"]})

    row = records(df)[0]

    assert row["psi"] is None
    assert json.dumps(row)  # would raise/emit invalid JSON if NaN had survived


def test_pr_curve_downsample_passes_short_curves_through():
    recall = np.linspace(1.0, 0.0, 12)
    precision = np.linspace(0.1, 1.0, 12)

    points = downsample_pr_curve(precision, recall, n_points=300)

    assert len(points) == 12


# --- Integrity of the committed artifact ------------------------------------------------
# Unlike everything above, these read the real app/data/* files. They can do that because
# those two artifacts are the one data product this repo commits (CLAUDE.md), so CI can
# check them — and they need checking: the artifact is built by a ~15-minute offline run
# against data that is NOT in the repo, so nothing else would notice if it went stale,
# got truncated, or was rebuilt from a different model than the metrics claim.

ARTIFACT = Path("app/data/demo_alerts.parquet")
METRICS = Path("app/data/demo_metrics.json")

needs_artifact = pytest.mark.skipif(
    not (ARTIFACT.exists() and METRICS.exists()),
    reason="demo artifact not built yet — run python -m scripts.build_demo_artifact",
)


@pytest.fixture(scope="module")
def artifact():
    return pd.read_parquet(ARTIFACT)


@pytest.fixture(scope="module")
def metrics():
    with open(METRICS, encoding="utf-8") as f:
        return json.load(f)


@needs_artifact
def test_artifact_fits_the_deployment_budget():
    # PLAN.md Phase 8's check. The binding constraint is Streamlit Community Cloud's ~1GB
    # RAM, not GitHub's 100MB file limit.
    assert ARTIFACT.stat().st_size / 1e6 < 20


@needs_artifact
def test_shap_columns_reconstruct_the_shipped_score(artifact, metrics):
    """The app draws its waterfall from the shipped `shap_*` columns and its ranking from
    the shipped `model_score`. If those two ever disagree, the explanation shown to an
    analyst is for a different prediction than the one being explained — silently. This
    asserts the additivity identity end-to-end on the actual bytes that deploy.
    """
    shap_cols = [c for c in artifact.columns if c.startswith("shap_")]
    feat_cols = [c for c in artifact.columns if c.startswith("feat_")]
    assert [c[5:] for c in shap_cols] == [c[5:] for c in feat_cols]

    margin = metrics["shap"]["base_value"] + artifact[shap_cols].to_numpy("float64").sum(axis=1)
    reconstructed = 1 / (1 + np.exp(-margin))

    # float32 storage, so exact equality is not on offer; 1e-5 is far tighter than any
    # difference that could reorder the queue or change a waterfall's shape.
    assert np.abs(reconstructed - artifact["model_score"].to_numpy("float64")).max() < 1e-5


@needs_artifact
def test_ranks_are_true_ranks_over_the_full_test_split(artifact, metrics):
    """The artifact is a sample; its ranks are not. Rank 1 must mean "highest-scored
    transaction in the test split", not "highest-scored transaction that survived
    sampling" — otherwise the app's queue silently misstates the model's actual output.
    """
    ordered = artifact.sort_values("rank")
    assert ordered["model_score"].diff().dropna().max() <= 1e-9  # monotone non-increasing
    assert artifact["rank"].max() <= metrics["demo_sample"]["ranks_are_out_of"]
    assert artifact["rank"].is_unique


@needs_artifact
def test_entire_alert_queue_ships_unsampled(artifact, metrics):
    """The equal-recall queue is the headline result; a hole in it would understate the
    model. It must be exactly ranks 1..N with nothing missing.
    """
    queue = artifact[artifact["in_alert_queue"]]
    assert len(queue) == metrics["headline"]["model_alerts"]
    assert sorted(queue["rank"].tolist()) == list(range(1, len(queue) + 1))


@needs_artifact
def test_both_reason_code_variants_are_populated_and_differ(artifact):
    # Phase 6's finding only survives into the app if both variants ship (CLAUDE.md).
    for col in ("reason_code_faithful", "reason_code_behavioural"):
        assert artifact[col].notna().all()
        assert artifact[col].astype(str).str.startswith("Flagged").all()

    assert not artifact["reason_code_behavioural"].astype(str).str.contains("via ACH").any()

    # Compared as str, not as the shipped `category` dtype: two categoricals with
    # different category sets raise on `!=` rather than comparing their values.
    differ = artifact["reason_code_faithful"].astype(str) != artifact[
        "reason_code_behavioural"
    ].astype(str)

    # Asserted on the alert queue specifically, which is what Phase 6 measured (99.86% of
    # the queue led with ACH) and what the app displays. Across the whole artifact the
    # divergence is only ~56%, and that is correct rather than a dilution bug: off-queue
    # rows are mostly Cheque/Credit Card, and for a non-ACH transaction the
    # `payment_format_ACH` one-hot is 0 with a NEGATIVE contribution, so it never enters
    # the top *positive* contributors and excluding it changes nothing. The variants are
    # meant to diverge exactly where the model is alerting, not everywhere.
    assert differ[artifact["in_alert_queue"]].mean() > 0.99
    assert differ.mean() > 0.4


@needs_artifact
def test_tail_is_flagged_and_its_over_representation_is_disclosed(artifact, metrics):
    """Taking all positives necessarily over-represents the Phase 7 tail. That's allowed;
    silently over-representing it is not. The flag must exist and the metrics must state
    both shares.
    """
    sample_share = metrics["demo_sample"]["tail_share_in_sample"]
    true_share = metrics["demo_sample"]["tail_share_in_test_split"]

    assert artifact["is_tail_population"].mean() == pytest.approx(sample_share)
    assert sample_share > true_share  # the honest direction; asserted so it can't silently flip


@needs_artifact
def test_metrics_are_json_clean_and_typed(metrics):
    # NaN and numpy scalars both survive json.dump in ways that break the browser or
    # arithmetic downstream — see `records`.
    assert "NaN" not in METRICS.read_text(encoding="utf-8")
    for row in metrics["recall_per_typology"]:
        assert isinstance(row["n_transactions"], int)
        assert isinstance(row["recall"], float)
    assert all(r["psi"] is None or isinstance(r["psi"], float) for r in metrics["psi_over_time"])


@needs_artifact
def test_headline_in_metrics_matches_the_project_result(metrics):
    # Guards against shipping an artifact built from a stale or retrained model whose
    # numbers no longer match reports/results.md.
    headline = metrics["headline"]
    assert headline["model_alerts"] == 10_011
    assert headline["baseline_alerts"] == 297_564
    assert headline["fp_reduction"] == pytest.approx(0.966, abs=0.001)
    assert metrics["headline_excluding_tail"]["fp_reduction"] == pytest.approx(0.962, abs=0.001)
