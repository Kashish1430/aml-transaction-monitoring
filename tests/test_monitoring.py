"""Drift-metric checks on hand-built synthetic distributions (PLAN.md Phase 7).

PLAN.md's stated Phase 7 check is "PSI values compute without NaN/inf on real data; a
deliberately shuffled 'future' slice shows elevated PSI vs. a stable slice." Taken
literally the second half of that check passes trivially and proves nothing — shuffling
rows preserves the marginal distribution exactly, so its PSI is ~0 by construction. That
makes a shuffle a good *negative* control and a useless positive one, so it is used as
exactly that here (`test_shuffled_slice_is_a_negative_control_not_a_positive_one`), and
the positive control is a genuinely perturbed slice instead.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    DEFAULT_EPSILON,
    category_share_psi,
    characteristic_stability,
    classify_psi,
    population_stability_index,
    psi_over_time,
    psi_table,
    quantile_bin_edges,
)

SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


# --- Bin edges ---------------------------------------------------------------------


def test_bin_edges_come_from_reference_and_are_open_ended(rng):
    reference = rng.normal(size=10_000)
    edges = quantile_bin_edges(reference, n_bins=10)

    assert len(edges) == 11
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert np.all(np.diff(edges) > 0)  # strictly increasing, no zero-width bins


def test_bin_edges_collapse_ties_instead_of_emitting_zero_width_bins():
    # 90% zeros: deciles 0.0 through 0.9 all land on 0.0. Naive quantile binning would
    # emit eight zero-width [0, 0] bins; the real feature families in src/features.py
    # (structuring_score_1d, *_graph_in_cycle) look exactly like this.
    reference = np.concatenate([np.zeros(9_000), np.arange(1, 1_001, dtype=float)])
    edges = quantile_bin_edges(reference, n_bins=10)

    assert np.all(np.diff(edges) > 0)
    assert len(edges) - 1 < 10  # fewer effective bins than requested, by design


def test_constant_reference_degenerates_to_one_bin_and_zero_psi():
    constant = np.full(1_000, 7.0)
    edges = quantile_bin_edges(constant, n_bins=10)

    assert list(edges) == [-np.inf, np.inf]
    # A constant feature cannot shift in distribution — even against different values.
    assert population_stability_index(constant, np.full(1_000, 9.0)) == pytest.approx(0.0)


def test_bin_edges_rejects_nonsense_bin_count():
    with pytest.raises(ValueError, match="n_bins must be >= 1"):
        quantile_bin_edges(np.arange(10.0), n_bins=0)


# --- PSI core ----------------------------------------------------------------------


def test_psi_of_a_distribution_against_itself_is_zero(rng):
    values = rng.normal(size=5_000)
    assert population_stability_index(values, values) == pytest.approx(0.0, abs=1e-12)


def test_shuffled_slice_is_a_negative_control_not_a_positive_one(rng):
    """A row shuffle leaves the marginal distribution identical, so PSI must stay ~0.

    This is the check PLAN.md's Phase 7 line asks for, and it is included precisely to
    document that it is *not* evidence the metric detects drift — see the module
    docstring. `test_perturbed_slice_shows_elevated_psi` is the real positive control.
    """
    values = rng.normal(size=20_000)
    shuffled = rng.permutation(values)

    assert population_stability_index(values, shuffled) == pytest.approx(0.0, abs=1e-12)
    assert classify_psi(population_stability_index(values, shuffled)) == "stable"


def test_perturbed_slice_shows_elevated_psi(rng):
    reference = rng.normal(loc=0.0, scale=1.0, size=20_000)
    stable = rng.normal(loc=0.0, scale=1.0, size=20_000)  # same generator, fresh draw
    shifted = rng.normal(loc=1.5, scale=1.0, size=20_000)  # 1.5-sigma mean shift

    psi_stable = population_stability_index(reference, stable)
    psi_shifted = population_stability_index(reference, shifted)

    assert classify_psi(psi_stable) == "stable"
    assert classify_psi(psi_shifted) == "significant"
    assert psi_shifted > 10 * psi_stable


def test_psi_grows_monotonically_with_the_size_of_the_shift(rng):
    reference = rng.normal(size=20_000)
    psis = [
        population_stability_index(reference, rng.normal(loc=shift, size=20_000))
        for shift in (0.1, 0.3, 0.6, 1.0)
    ]
    assert psis == sorted(psis)


def test_psi_is_finite_when_a_current_bin_empties_out():
    """The NaN/inf failure mode PLAN.md's Phase 7 check names, forced deliberately.

    Without the epsilon floor this is `(0 - 0.1) * ln(0 / 0.1)` = `-0.1 * -inf` = `+inf`.
    """
    reference = np.arange(10_000, dtype=float)
    # Current slice occupies only the reference's bottom decile -> nine empty bins.
    current = np.arange(1_000, dtype=float)

    psi = population_stability_index(reference, current)
    assert np.isfinite(psi)
    assert psi > 1.0  # a genuinely enormous shift, and reported as one
    assert classify_psi(psi) == "significant"


def test_psi_handles_values_beyond_the_reference_range():
    # Outer edges are +/-inf, so a new maximum bins into the top bucket rather than
    # being dropped. Dropping it would understate drift at exactly the moment it matters.
    reference = np.arange(1_000, dtype=float)
    current = np.arange(10_000, 11_000, dtype=float)

    psi = population_stability_index(reference, current)
    assert np.isfinite(psi)
    assert psi > 1.0


def test_psi_rejects_non_finite_input():
    reference = np.arange(100, dtype=float)
    with pytest.raises(ValueError, match="non-finite"):
        population_stability_index(reference, np.array([1.0, np.nan, 3.0]))
    with pytest.raises(ValueError, match="non-finite"):
        population_stability_index(np.array([1.0, np.inf]), reference)


def test_psi_rejects_empty_input():
    with pytest.raises(ValueError, match="empty array"):
        population_stability_index(np.arange(10.0), np.array([]))


# --- psi_table ---------------------------------------------------------------------


def test_psi_table_contributions_sum_to_the_headline_psi(rng):
    reference = rng.normal(size=10_000)
    current = rng.normal(loc=0.5, size=10_000)

    table = psi_table(reference, current)
    assert table["psi_contribution"].sum() == pytest.approx(
        population_stability_index(reference, current)
    )
    # Proportions are proportions, floored but never renormalised away.
    assert table["reference_pct"].sum() == pytest.approx(1.0, abs=1e-9)
    assert table["current_pct"].sum() == pytest.approx(1.0, abs=1e-9)


def test_psi_table_localises_the_shift_to_the_right_bins():
    reference = np.arange(10_000, dtype=float)
    current = np.arange(9_000, 10_000, dtype=float)  # only the reference's TOP decile

    table = psi_table(reference, current)
    worst_bin = table["psi_contribution"].idxmax()
    # The emptied bottom bins dominate; the top bin is the one that gained mass.
    assert table.loc[table.index[-1], "current_pct"] == pytest.approx(1.0, abs=1e-6)
    assert table.loc[worst_bin, "psi_contribution"] > 0


# --- classify_psi ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("psi", "expected"),
    [(0.0, "stable"), (0.099, "stable"), (0.10, "moderate"), (0.24, "moderate"),
     (0.25, "significant"), (3.0, "significant")],
)
def test_classify_psi_bands_are_inclusive_at_the_lower_edge(psi, expected):
    assert classify_psi(psi) == expected


def test_classify_psi_accepts_custom_thresholds():
    assert classify_psi(0.05, {"moderate": 0.01, "significant": 0.02}) == "significant"


# --- Categorical / binary features -------------------------------------------------


def test_quantile_psi_is_structurally_blind_to_a_vanishing_binary_feature():
    """Documents the failure `category_share_psi` exists to fix, so it stays fixed.

    This is not a hypothetical: on the real train->test comparison,
    `payment_format_Reinvestment` went from 15.8% of transactions to zero and quantile
    binning scored it 0.0000. Every quantile of a 0/1 column is the same value, so the
    edges collapse to one bin and the metric can no longer express anything.
    """
    reference = np.concatenate([np.ones(1_580), np.zeros(8_420)])  # 15.8% ones
    current = np.zeros(10_000)  # the format disappears entirely

    assert population_stability_index(reference, current) == pytest.approx(0.0)
    assert category_share_psi(reference, current) > 1.0


def test_category_share_psi_detects_a_shift_in_a_binary_rate():
    reference = np.concatenate([np.ones(1_000), np.zeros(9_000)])  # 10%
    current = np.concatenate([np.ones(3_000), np.zeros(7_000)])  # 30%

    psi = category_share_psi(reference, current)
    assert np.isfinite(psi)
    assert classify_psi(psi) == "significant"


def test_category_share_psi_is_zero_for_an_unchanged_mix():
    values = np.array([0, 0, 0, 1, 1, 2] * 1_000, dtype=float)
    assert category_share_psi(values, values) == pytest.approx(0.0, abs=1e-12)


def test_category_share_psi_counts_a_brand_new_category():
    # A category absent from the reference is drift and must contribute, not be dropped.
    reference = np.array([0.0, 1.0] * 5_000)
    current = np.array([0.0, 1.0, 2.0] * 3_333)

    assert category_share_psi(reference, current) > 0.1


def test_characteristic_stability_routes_low_cardinality_features_to_category_shares():
    n = 10_000
    reference = pd.DataFrame(
        {
            "binary_flag": np.concatenate([np.ones(1_580), np.zeros(n - 1_580)]),
            "continuous": np.linspace(0, 100, n),
        }
    )
    current = pd.DataFrame(
        {
            "binary_flag": np.zeros(n),  # vanishes
            "continuous": np.linspace(0, 100, n),
        }
    )

    table = characteristic_stability(reference, current).set_index("feature")

    assert table.loc["binary_flag", "method"] == "category_share"
    assert table.loc["continuous", "method"] == "quantile"
    # The vanished flag must now be the top-ranked drift, not an invisible 0.0.
    assert table.loc["binary_flag", "band"] == "significant"
    assert table.loc["continuous", "band"] == "stable"


def test_characteristic_stability_cardinality_cutoff_is_configurable():
    n = 5_000
    rng_local = np.random.default_rng(SEED)
    # 6 distinct values -> categorical by default, quantile if the cutoff is lowered.
    values = rng_local.integers(0, 6, size=n).astype(float)
    frame = pd.DataFrame({"few_valued": values})

    default_method = characteristic_stability(frame, frame).iloc[0]["method"]
    lowered_method = characteristic_stability(
        frame, frame, max_categorical_cardinality=3
    ).iloc[0]["method"]

    assert default_method == "category_share"
    assert lowered_method == "quantile"


# --- CSI ---------------------------------------------------------------------------


def test_characteristic_stability_ranks_the_drifted_feature_first(rng):
    n = 10_000
    reference = pd.DataFrame(
        {
            "stable_feature": rng.normal(size=n),
            "drifted_feature": rng.normal(size=n),
            "constant_feature": np.ones(n),
        }
    )
    current = pd.DataFrame(
        {
            "stable_feature": rng.normal(size=n),
            "drifted_feature": rng.normal(loc=2.0, size=n),
            "constant_feature": np.ones(n),
        }
    )

    table = characteristic_stability(reference, current)

    assert table.iloc[0]["feature"] == "drifted_feature"
    assert table.iloc[0]["band"] == "significant"
    assert table[table["feature"] == "stable_feature"].iloc[0]["band"] == "stable"


def test_characteristic_stability_flags_a_constant_feature_that_changed_value():
    """A feature constant at 1.0 that becomes constant at 5.0 has completely changed.

    Quantile binning cannot express this (one bin, CSI 0.0 — see
    `test_constant_reference_degenerates_to_one_bin_and_zero_psi`), which is exactly why
    a single-valued reference routes to the category-share path instead.
    """
    reference = pd.DataFrame({"constant_feature": np.ones(1_000)})
    current = pd.DataFrame({"constant_feature": np.full(1_000, 5.0)})

    row = characteristic_stability(reference, current).iloc[0]
    assert row["method"] == "category_share"
    assert row["band"] == "significant"
    assert row["reference_mean"] != row["current_mean"]


def test_characteristic_stability_is_silent_on_a_genuinely_unchanged_constant():
    constant = pd.DataFrame({"constant_feature": np.ones(1_000)})

    row = characteristic_stability(constant, constant).iloc[0]
    assert row["csi"] == pytest.approx(0.0)
    assert row["band"] == "stable"


def test_characteristic_stability_raises_on_a_vanished_column():
    reference = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0)})
    current = pd.DataFrame({"a": np.arange(100.0)})

    with pytest.raises(ValueError, match="missing from current_df"):
        characteristic_stability(reference, current)


def test_characteristic_stability_respects_an_explicit_column_subset(rng):
    reference = pd.DataFrame({"a": rng.normal(size=1_000), "b": rng.normal(size=1_000)})
    current = pd.DataFrame({"a": rng.normal(size=1_000), "b": rng.normal(size=1_000)})

    table = characteristic_stability(reference, current, feature_columns=["a"])
    assert list(table["feature"]) == ["a"]


# --- psi_over_time -----------------------------------------------------------------


def test_psi_over_time_flags_only_the_drifted_days(rng):
    reference = rng.normal(size=20_000)

    # Three days drawn from the reference distribution, then two shifted days.
    per_day = 2_000
    days, values = [], []
    for day, loc in enumerate([0.0, 0.0, 0.0, 2.0, 2.0]):
        days.extend([pd.Timestamp("2022-09-01") + pd.Timedelta(days=day)] * per_day)
        values.append(rng.normal(loc=loc, size=per_day))

    table = psi_over_time(reference, np.concatenate(values), pd.Series(days), min_slice_size=500)

    assert len(table) == 5
    assert list(table["band"][:3]) == ["stable"] * 3
    assert list(table["band"][3:]) == ["significant"] * 2
    assert (table["n_rows"] == per_day).all()


def test_psi_over_time_keeps_thin_slices_as_rows_rather_than_dropping_them(rng):
    reference = rng.normal(size=5_000)
    days = ["2022-09-01"] * 2_000 + ["2022-09-02"] * 10
    values = rng.normal(size=2_010)

    table = psi_over_time(reference, values, pd.Series(pd.to_datetime(days)), min_slice_size=1_000)

    assert list(table["period"]) == ["2022-09-01", "2022-09-02"]
    thin = table.iloc[1]
    assert thin["n_rows"] == 10
    assert np.isnan(thin["psi"])
    assert thin["band"] == "insufficient_data"


def test_psi_over_time_rejects_mismatched_timestamps(rng):
    with pytest.raises(ValueError, match="same length"):
        psi_over_time(
            rng.normal(size=100),
            rng.normal(size=100),
            pd.Series(pd.to_datetime(["2022-09-01"] * 99)),
        )


def test_psi_over_time_is_anchored_to_the_reference_not_to_neighbouring_slices(rng):
    """Gradual drift must accumulate, which chained slice-to-slice comparison hides."""
    reference = rng.normal(size=20_000)
    per_day = 2_000
    days, values = [], []
    for day, loc in enumerate([0.4, 0.8, 1.2, 1.6]):  # each day only mildly past the last
        days.extend([pd.Timestamp("2022-09-01") + pd.Timedelta(days=day)] * per_day)
        values.append(rng.normal(loc=loc, size=per_day))

    table = psi_over_time(reference, np.concatenate(values), pd.Series(days), min_slice_size=500)

    assert list(table["psi"]) == sorted(table["psi"])  # monotonically worsening
    assert table.iloc[-1]["band"] == "significant"


def test_default_epsilon_is_small_enough_not_to_distort_a_populated_bin(rng):
    values = rng.normal(size=10_000)
    assert population_stability_index(values, values, epsilon=DEFAULT_EPSILON) == pytest.approx(
        0.0, abs=1e-12
    )
