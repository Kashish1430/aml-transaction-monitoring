"""Tests for the deployed app's data layer (PLAN.md Phase 9).

`app/streamlit_app.py` is UI only and executes top-to-bottom on import, so it can't be
imported here; everything it does to the data lives in `app/artifact.py` and is tested
below against the real committed artifact. The properties that matter are the ones a
reviewer of the *live app* can't check by clicking: that the app never re-ranks, that a
filtered view stays rank-ordered, and that the waterfall it draws adds up to the score it
displays beside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import artifact

needs_artifact = pytest.mark.skipif(
    not (artifact.ARTIFACT_PATH.exists() and artifact.METRICS_PATH.exists()),
    reason="demo artifact not built — run python -m scripts.build_demo_artifact",
)

pytestmark = needs_artifact


@pytest.fixture(scope="module")
def alerts():
    return artifact.load_alerts()


@pytest.fixture(scope="module")
def metrics():
    return artifact.load_metrics()


def test_loads_rank_ordered_without_re_ranking(alerts):
    """The app sorts by the shipped rank and must never recompute one. A rank recomputed
    over the 30,000 shipped rows would redefine "rank 1" from "highest-scored transaction
    in the held-out split" to "highest-scored row that survived sampling".
    """
    assert alerts["rank"].is_monotonic_increasing
    assert alerts["model_score"].diff().dropna().max() <= 1e-9
    # Ranks are sparse above the queue — proof they index the full split, not this table.
    assert alerts["rank"].max() > len(alerts)


def test_feature_names_recovered_from_the_artifact(alerts, metrics):
    names = artifact.feature_names(alerts)

    assert len(names) == metrics["model"]["n_features"]
    assert all(f"feat_{n}" in alerts.columns and f"shap_{n}" in alerts.columns for n in names)
    assert "payment_format_ACH" in names


def test_display_columns_exclude_the_feature_blocks(alerts):
    cols = artifact.display_columns(alerts)

    assert not any(c.startswith(("feat_", "shap_")) for c in cols)
    assert {"rank", "model_score", "reason_code_behavioural", "is_tail_population"} <= set(cols)


def test_queue_only_filter_returns_exactly_the_queue(alerts, metrics):
    queue = artifact.apply_filters(alerts, queue_only=True)

    assert len(queue) == metrics["headline"]["model_alerts"]
    assert queue["in_alert_queue"].all()
    assert queue["rank"].tolist() == list(range(1, len(queue) + 1))


def test_filters_preserve_rank_order(alerts):
    # Every filter is a mask over an already-sorted frame; if one ever reorders rows the
    # queue would silently stop being a queue.
    filtered = artifact.apply_filters(
        alerts, queue_only=False, include_tail=False,
        typologies=["FAN-IN", "CYCLE"], min_score=0.5,
    )

    assert filtered["rank"].is_monotonic_increasing


def test_none_and_empty_filter_lists_mean_different_things(alerts):
    """An untouched filter (None) shows everything; a filter the user emptied shows
    nothing. Collapsing the two would make clearing a filter look like a no-op.
    """
    unconstrained = artifact.apply_filters(alerts, queue_only=False, typologies=None)
    emptied = artifact.apply_filters(alerts, queue_only=False, typologies=[])

    assert len(unconstrained) == len(alerts)
    assert len(emptied) == 0


def test_tail_filter_removes_exactly_the_flagged_rows(alerts, metrics):
    with_tail = artifact.apply_filters(alerts, queue_only=False, include_tail=True)
    without = artifact.apply_filters(alerts, queue_only=False, include_tail=False)

    assert not without["is_tail_population"].any()
    assert len(with_tail) - len(without) == int(alerts["is_tail_population"].sum())
    assert with_tail["is_tail_population"].mean() == pytest.approx(
        metrics["demo_sample"]["tail_share_in_sample"]
    )


def test_date_filter_is_inclusive_of_both_endpoints(alerts):
    day = alerts["timestamp"].dt.normalize().iloc[0]
    window = (day, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))

    filtered = artifact.apply_filters(alerts, queue_only=False, date_range=window)

    assert (filtered["timestamp"].dt.normalize() == day).all()
    assert len(filtered) == int((alerts["timestamp"].dt.normalize() == day).sum())


def test_search_matches_account_substring_and_exact_bank(alerts):
    row = alerts.iloc[0]

    by_account = artifact.apply_filters(alerts, queue_only=False, search=row["from_account_key"])
    by_bank = artifact.apply_filters(alerts, queue_only=False, search=str(row["from_bank"]))

    assert len(by_account) >= 1
    assert row["rank"] in by_account["rank"].values
    assert len(by_bank) >= 1
    assert (
        by_bank["from_bank"].eq(row["from_bank"]) | by_bank["to_bank"].eq(row["from_bank"])
        | by_bank["from_account_key"].str.contains(str(row["from_bank"]))
        | by_bank["to_account_key"].str.contains(str(row["from_bank"]))
    ).all()


def test_queue_stats_flags_when_counts_are_sample_counts(alerts, metrics):
    """The app labels a view differently depending on whether its counts are population
    counts. Only the alert queue ships complete, so any view reaching below it is sampled.
    """
    queue_view = artifact.queue_stats(artifact.apply_filters(alerts, queue_only=True), metrics)
    wide_view = artifact.queue_stats(artifact.apply_filters(alerts, queue_only=False), metrics)

    assert queue_view["population_exact"] is True
    assert wide_view["population_exact"] is False
    assert queue_view["n_shown"] == metrics["headline"]["model_alerts"]


def test_queue_stats_on_an_empty_view_does_not_divide_by_zero(alerts, metrics):
    empty = artifact.queue_stats(artifact.apply_filters(alerts, typologies=[]), metrics)

    assert empty["n_shown"] == 0
    assert np.isnan(empty["precision"])
    assert empty["population_exact"] is False


def test_waterfall_preserves_the_additivity_identity(alerts, metrics):
    """The collapsed "N other features" bar must be a real sum, not a residual — the bars
    are shown adding up to the score, so they have to actually add up to it.
    """
    names = artifact.feature_names(alerts)
    base = metrics["shap"]["base_value"]
    row = alerts.iloc[0]

    wf = artifact.waterfall_data(row, names, base, top_n=10)
    total = base + wf["contribution"].sum()
    full_total = base + sum(float(row[f"shap_{n}"]) for n in names)

    assert wf["contribution"].sum() == pytest.approx(full_total - base, abs=1e-6)
    assert artifact.score_from_margin(total) == pytest.approx(row["model_score"], abs=1e-5)


def test_waterfall_shows_top_n_plus_one_aggregate_row(alerts, metrics):
    names = artifact.feature_names(alerts)
    wf = artifact.waterfall_data(alerts.iloc[0], names, metrics["shap"]["base_value"], top_n=10)

    assert len(wf) == 11
    assert wf["feature"].str.contains("other features").sum() == 1
    assert wf["contribution"].is_monotonic_decreasing  # most-positive first, for the chart


def test_reconstructed_score_matches_the_shipped_score_across_the_queue(alerts, metrics):
    """What the Alert Detail tab displays as the waterfall's endpoint must equal what the
    Alert Queue tab ranked by — otherwise the app explains a different prediction than the
    one it showed, silently.
    """
    names = artifact.feature_names(alerts)
    base = metrics["shap"]["base_value"]
    sample = alerts.head(200)

    implied = np.array([artifact.reconstructed_score(r, names, base) for _, r in sample.iterrows()])

    assert np.abs(implied - sample["model_score"].to_numpy("float64")).max() < 1e-5


def test_metric_frames_are_well_formed(metrics):
    typ = artifact.typology_frame(metrics)
    pr = artifact.pr_curve_frame(metrics)
    psi = artifact.psi_frame(metrics)
    imp = artifact.importance_frame(metrics)

    assert len(typ) == 8 and typ["recall"].between(0, 1).all()
    assert pr["recall"].is_monotonic_decreasing and pr["precision"].between(0, 1).all()
    assert psi["psi"].isna().any()  # thin slices are kept as rows, not dropped
    assert imp["mean_abs_shap"].is_monotonic_decreasing


def test_prettify_is_cosmetic_only():
    assert artifact.prettify_feature("payment_format_ACH") == "payment format: ACH"
    assert (
        artifact.prettify_feature("sender_out_7d_distinct_counterparties")
        == "sender out 7d distinct counterparties"
    )
