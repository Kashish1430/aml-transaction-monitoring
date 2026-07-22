"""Model pipeline checks on hand-built synthetic cases (PLAN.md Phase 4): the
time-ordered split's leakage guarantee, feature-matrix construction, imbalance
weighting, and an end-to-end train/save/load round trip on a tiny synthetic dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.model import (
    NON_FEATURE_COLUMNS,
    PAYMENT_FORMATS,
    compute_scale_pos_weight,
    load_metadata,
    load_xgboost_model,
    prepare_feature_matrix,
    save_model,
    time_ordered_split,
    train_isolation_forest,
    train_xgboost,
)


def _modelling_table(n=100, seed=0):
    rng = np.random.default_rng(seed)
    timestamps = pd.to_datetime("2022-09-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 60 * 60 * 24 * 10, size=n)), unit="s"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "from_account_key": [f"A{i % 10}" for i in range(n)],
            "to_account_key": [f"B{i % 10}" for i in range(n)],
            "amount_paid_usd": rng.uniform(10, 20000, size=n),
            "payment_format": rng.choice(["ACH", "Cheque", "Wire"], size=n),
            "sender_out_1d_count": rng.integers(0, 20, size=n).astype(float),
            "sender_graph_in_cycle": rng.choice([True, False], size=n),
            "is_laundering": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
        }
    )


def test_time_ordered_split_boundaries_are_strictly_time_ordered():
    df = _modelling_table(n=1000)
    train, val, test = time_ordered_split(df, train_frac=0.6, val_frac=0.2)

    assert len(train) + len(val) + len(test) == len(df)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_time_ordered_split_respects_fractions():
    df = _modelling_table(n=1000)
    train, val, test = time_ordered_split(df, train_frac=0.6, val_frac=0.2)

    assert len(train) == 600
    assert len(val) == 200
    assert len(test) == 200


def test_prepare_feature_matrix_produces_fixed_payment_format_columns():
    # Only "ACH" appears in this split -- the one-hot columns must still cover every
    # entry in PAYMENT_FORMATS, not just the categories actually present.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2022-09-01"] * 3),
            "from_account_key": ["A", "B", "C"],
            "to_account_key": ["X", "Y", "Z"],
            "payment_format": ["ACH", "ACH", "ACH"],
            "amount_paid_usd": [100.0, 200.0, 300.0],
            "sender_out_1d_count": [1.0, 2.0, 3.0],
            "is_laundering": [0, 0, 1],
        }
    )
    x, y = prepare_feature_matrix(df)

    expected_dummy_cols = {f"payment_format_{fmt}" for fmt in PAYMENT_FORMATS}
    assert expected_dummy_cols.issubset(set(x.columns))
    assert x["payment_format_ACH"].tolist() == [1.0, 1.0, 1.0]
    assert x["payment_format_Wire"].tolist() == [0.0, 0.0, 0.0]
    assert not (set(NON_FEATURE_COLUMNS) - {"is_laundering"}) & set(x.columns)
    assert list(y) == [0, 0, 1]
    assert x.dtypes.unique().tolist() == [np.dtype("float32")]


def test_compute_scale_pos_weight_matches_manual_ratio():
    y = pd.Series([0, 0, 0, 0, 1, 1])  # 4 negatives, 2 positives
    assert compute_scale_pos_weight(y) == pytest.approx(2.0)


def test_compute_scale_pos_weight_rejects_zero_positives():
    y = pd.Series([0, 0, 0])
    with pytest.raises(ValueError, match="zero positives"):
        compute_scale_pos_weight(y)


def test_train_and_save_and_load_round_trip(tmp_path):
    df = _modelling_table(n=400, seed=1)
    train, val, test = time_ordered_split(df, train_frac=0.6, val_frac=0.2)
    x_train, y_train = prepare_feature_matrix(train)
    x_val, y_val = prepare_feature_matrix(val)

    spw = compute_scale_pos_weight(y_train)
    model = train_xgboost(x_train, y_train, x_val, y_val, spw, seed=42)
    iso_forest = train_isolation_forest(x_train, seed=42)

    preds = model.predict_proba(x_train)[:, 1]
    assert len(preds) == len(x_train)
    assert ((preds >= 0) & (preds <= 1)).all()

    anomaly_scores = iso_forest.decision_function(x_train)
    assert len(anomaly_scores) == len(x_train)

    metadata = {"scale_pos_weight": spw, "seed": 42}
    save_model(model, iso_forest, list(x_train.columns), metadata, tmp_path)

    assert (tmp_path / "xgboost_model.json").exists()
    assert (tmp_path / "isolation_forest.joblib").exists()
    assert (tmp_path / "model_metadata.json").exists()

    loaded_model = load_xgboost_model(tmp_path)
    loaded_preds = loaded_model.predict_proba(x_train)[:, 1]
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5)

    loaded_metadata = load_metadata(tmp_path)
    assert loaded_metadata["scale_pos_weight"] == pytest.approx(spw)
    assert loaded_metadata["feature_names"] == list(x_train.columns)
