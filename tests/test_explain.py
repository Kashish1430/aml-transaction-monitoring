"""Explainability checks: SHAP faithfulness and reason-code correctness (PLAN.md Phase 6).

The reason-code tests matter more than they look. A reason code is generated text shown
to an analyst as a justification for escalating a case, so the failure mode to guard
against is not a crash but a *plausible-sounding sentence that misstates the transaction*
— citing a feature that argued against the alert, or printing a number that isn't the
row's actual value. Every test below pins one of those down against hand-built values.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.explain import (
    PAYMENT_FORMAT_FEATURES,
    align_features,
    build_reason_code,
    build_reason_codes,
    compute_shap_values,
    describe_feature,
    global_importance,
    select_explanation_sample,
    shap_base_value,
    top_contributors,
)

FEATURE_NAMES = [
    "amount_paid_usd",
    "structuring_score_1d",
    "sender_pass_through_ratio_1d",
    "sender_out_1d_distinct_counterparties",
    "sender_graph_in_cycle",
    "payment_format_ACH",
]


@pytest.fixture
def tiny_model():
    """A real, tiny XGBClassifier — SHAP additivity is a property of the fitted model, so
    it has to be checked against an actual tree ensemble, not a mock.
    """
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.random((200, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    # A learnable signal so the trees actually split rather than emitting a constant.
    y = (x["structuring_score_1d"] + x["amount_paid_usd"] > 1.0).astype(int)
    model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(x, y)
    return model, x


def test_shap_values_satisfy_additivity_against_model_margin(tiny_model):
    """The faithfulness guarantee: base value + summed SHAP contributions must reconstruct
    the model's raw margin output exactly. If this fails, every reason code in the system
    is citing contributions that don't correspond to the model's actual decision.
    """
    model, x = tiny_model
    shap_values = compute_shap_values(model, x)
    base_value = shap_base_value(model, x, shap_values)

    reconstructed = base_value + shap_values.sum(axis=1)
    actual_margin = model.predict(x, output_margin=True)

    np.testing.assert_allclose(reconstructed, actual_margin, rtol=1e-5, atol=1e-5)


def test_shap_base_value_does_not_inherit_lazily_corrected_expected_value(tiny_model):
    """Regression test for the trap documented in `shap_base_value` and challenges.md.

    On shap 0.45.1 + xgboost 2.0.3, a freshly built `TreeExplainer` reports
    `expected_value == array([logit(base_score)])` and only silently corrects it to the
    true bias during the first `shap_values()` call. `shap_base_value` derives the bias
    from the model instead, so it must agree with the *corrected* value, not the stale
    pre-call one. Asserted rather than assumed so that a future shap/xgboost upgrade
    which changes this behaviour surfaces here instead of quietly shifting every
    reconstructed margin.
    """
    import shap

    model, x = tiny_model
    explainer = shap.TreeExplainer(model)
    stale_expected_value = np.asarray(explainer.expected_value).copy()

    shap_values = explainer.shap_values(x)
    corrected_expected_value = float(np.asarray(explainer.expected_value).item())
    derived = shap_base_value(model, x, shap_values)

    assert derived == pytest.approx(corrected_expected_value, abs=1e-5)
    # The whole reason this function exists: the pre-call attribute is NOT the right value.
    assert stale_expected_value.item() != pytest.approx(derived, abs=1e-5)


def test_shap_base_value_rejects_contributions_that_break_additivity(tiny_model):
    model, x = tiny_model
    tampered = compute_shap_values(model, x)
    tampered[0, 0] += 5.0  # one row's contributions no longer reconstruct its margin

    with pytest.raises(ValueError, match="additivity violated"):
        shap_base_value(model, x, tampered)


def test_shap_values_shape_matches_feature_matrix(tiny_model):
    model, x = tiny_model
    assert compute_shap_values(model, x).shape == x.shape


def test_top_contributors_excludes_features_arguing_against_the_alert():
    # Feature 1 has the largest |SHAP| of all, but it is NEGATIVE -- it pushed the score
    # DOWN. It must never appear as a "reason" the transaction was flagged.
    shap_row = np.array([0.4, -0.9, 0.2, 0.0, 0.1, 0.3])

    contributors = top_contributors(shap_row, FEATURE_NAMES, top_n=3)
    names = [name for name, _ in contributors]

    assert "structuring_score_1d" not in names
    assert names == ["amount_paid_usd", "payment_format_ACH", "sender_pass_through_ratio_1d"]


def test_top_contributors_returns_fewer_than_requested_rather_than_padding():
    # Only two features pushed the score up; asking for 4 must yield 2, not 2 real plus
    # 2 features that argued the other way.
    shap_row = np.array([0.5, 0.1, -0.2, -0.3, 0.0, -0.1])
    assert len(top_contributors(shap_row, FEATURE_NAMES, top_n=4)) == 2


def test_top_contributors_honours_exclusions():
    shap_row = np.array([0.1, 0.2, 0.05, 0.0, 0.0, 0.9])  # ACH dominates

    faithful = top_contributors(shap_row, FEATURE_NAMES, top_n=1)
    behavioural = top_contributors(
        shap_row, FEATURE_NAMES, top_n=1, exclude_features=PAYMENT_FORMAT_FEATURES
    )

    assert faithful[0][0] == "payment_format_ACH"
    assert behavioural[0][0] == "structuring_score_1d"


def test_describe_feature_renders_raw_values_accurately():
    """Each clause must state this row's real number. Checked per family because the
    families have genuinely different phrasings and units (counts vs. USD vs. ratios).
    """
    assert describe_feature("amount_paid_usd", 9_450.0) == "a $9,450 payment"

    structuring = describe_feature("structuring_score_1d", 14)
    assert "14 of the sender's payments" in structuring
    assert "past 24h" in structuring
    assert "$10,000 reporting threshold" in structuring

    pass_through = describe_feature("sender_pass_through_ratio_7d", 0.98)
    assert "0.98" in pass_through
    assert "past 7 days" in pass_through

    counterparties = describe_feature("sender_out_1d_distinct_counterparties", 9)
    assert counterparties == "the sender sent to 9 distinct counterparties in the past 24h"

    assert describe_feature("receiver_in_30d_count", 42) == (
        "the receiver received 42 transactions in the past 30 days"
    )
    assert describe_feature("sender_out_7d_amount_usd", 1_234_567.0) == (
        "the sender sent $1,234,567 in the past 7 days"
    )
    assert describe_feature("receiver_graph_in_degree", 31) == (
        "the receiver received from 31 distinct accounts in the graph window"
    )
    assert "collector" in describe_feature("receiver_graph_fan_in_score", 0.95)
    assert "distributor" in describe_feature("sender_graph_fan_out_score", 0.9)


def test_describe_feature_respects_binary_feature_direction():
    """A zero-valued binary feature must not be described as if it were set — the whole
    point of a reason code is that the analyst can trust its factual claims.
    """
    assert "sits on a short directed cycle" in describe_feature("sender_graph_in_cycle", 1.0)
    assert "does not sit on a short directed cycle" == describe_feature(
        "sender_graph_in_cycle", 0.0
    ).replace("the sender ", "")

    assert describe_feature("payment_format_ACH", 1.0) == "the payment was made via ACH"
    assert describe_feature("payment_format_ACH", 0.0) == "the payment was not made via ACH"


def test_describe_feature_falls_back_on_unknown_name_instead_of_raising():
    # Degrading to a raw name/value is recoverable in production; an exception inside the
    # explanation layer would take down an alert queue that is otherwise fine.
    assert describe_feature("some_future_feature", 3.5) == "some_future_feature = 3.5"


def test_build_reason_code_composes_top_contributors_with_their_own_values():
    shap_row = np.array([0.3, 0.9, 0.0, 0.0, 0.0, 0.1])
    raw_row = pd.Series(
        {
            "amount_paid_usd": 9_800.0,
            "structuring_score_1d": 12,
            "sender_pass_through_ratio_1d": 0.5,
            "sender_out_1d_distinct_counterparties": 2,
            "sender_graph_in_cycle": 0.0,
            "payment_format_ACH": 1.0,
        }
    )

    code = build_reason_code(shap_row, raw_row, FEATURE_NAMES, top_n=2)

    assert code.startswith("Flagged: ")
    assert code.endswith(".")
    # Ordered by SHAP contribution: structuring (0.9) before amount (0.3).
    assert code.index("12 of the sender's payments") < code.index("$9,800 payment")
    # top_n=2 means the third-largest contributor (ACH, 0.1) is not cited.
    assert "ACH" not in code


def test_build_reason_code_handles_no_positive_contributors():
    shap_row = np.array([-0.1, -0.2, 0.0, 0.0, 0.0, -0.3])
    raw_row = pd.Series(dict.fromkeys(FEATURE_NAMES, 0.0))

    code = build_reason_code(shap_row, raw_row, FEATURE_NAMES)
    assert "no individual feature pushed" in code


def test_build_reason_codes_is_indexed_like_the_input_frame():
    shap_values = np.array([[0.5, 0.1, 0, 0, 0, 0.2], [0.1, 0.7, 0, 0, 0, 0.2]])
    x = pd.DataFrame(
        [[100.0, 3, 0.4, 1, 0.0, 1.0], [200.0, 7, 0.9, 4, 1.0, 0.0]],
        columns=FEATURE_NAMES,
        index=[41, 99],  # non-default index, as a real split slice would have
    )

    codes = build_reason_codes(shap_values, x, top_n=1)

    assert list(codes.index) == [41, 99]
    assert "$100 payment" in codes.loc[41]
    assert "7 of the sender's payments" in codes.loc[99]


def test_global_importance_ranks_by_mean_absolute_shap():
    # Feature 2's contributions are large but cancel in sign; mean |SHAP| must still rank
    # it top, and mean_shap must expose the cancellation.
    shap_values = np.array([[0.1, 1.0, 0.0, 0, 0, 0], [0.1, -1.0, 0.0, 0, 0, 0]])

    importance = global_importance(shap_values, FEATURE_NAMES)

    assert importance.iloc[0]["feature"] == "structuring_score_1d"
    assert importance.iloc[0]["mean_abs_shap"] == pytest.approx(1.0)
    assert importance.iloc[0]["mean_shap"] == pytest.approx(0.0)


def test_global_importance_rejects_mismatched_feature_names():
    with pytest.raises(ValueError, match="SHAP columns"):
        global_importance(np.zeros((5, 3)), FEATURE_NAMES)


def test_align_features_reorders_to_training_order():
    x = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["c", "a", "b"])
    assert list(align_features(x, ["a", "b", "c"]).columns) == ["a", "b", "c"]
    assert align_features(x, ["a", "b", "c"]).iloc[0].tolist() == [2.0, 3.0, 1.0]


def test_align_features_raises_on_missing_column_instead_of_filling_zeros():
    x = pd.DataFrame([[1.0]], columns=["a"])
    with pytest.raises(ValueError, match="missing columns"):
        align_features(x, ["a", "b"])


def test_select_explanation_sample_keeps_every_positive_and_every_alert():
    y_true = np.zeros(1000, dtype=int)
    y_true[[5, 700, 999]] = 1
    alert_idx = np.array([0, 1, 2, 700])

    sample = select_explanation_sample(y_true, alert_idx, n_background_negatives=50, seed=42)

    assert set([5, 700, 999]).issubset(sample)
    assert set(alert_idx).issubset(sample)
    assert len(sample) == len(np.unique(sample))
    assert list(sample) == sorted(sample)
    # 3 positives + 4 alerts (700 overlaps) = 6 unique, plus 50 sampled negatives.
    assert len(sample) == 56


def test_select_explanation_sample_is_deterministic_for_a_fixed_seed():
    y_true = np.zeros(500, dtype=int)
    y_true[3] = 1
    first = select_explanation_sample(y_true, np.array([1]), 20, seed=7)
    second = select_explanation_sample(y_true, np.array([1]), 20, seed=7)
    np.testing.assert_array_equal(first, second)


def test_select_explanation_sample_caps_at_available_rows():
    y_true = np.zeros(10, dtype=int)
    y_true[0] = 1
    sample = select_explanation_sample(y_true, np.array([1]), n_background_negatives=1000, seed=1)
    assert len(sample) == 10
