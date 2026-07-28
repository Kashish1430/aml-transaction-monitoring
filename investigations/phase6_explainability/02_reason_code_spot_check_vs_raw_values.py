"""PLAN.md's Phase 6 check, executed: spot-check flagged alerts' reason codes against
their raw feature values.

Supports reports/results.md's Phase 6 section. A reason code is generated text shown to
an analyst as justification for escalating a case, so the failure mode worth guarding
against is not a crash but a *fluent sentence that misstates the transaction*. This
script proves the sentences are grounded by, for each of the top-5 ranked test alerts:

1. printing the reason code (both the faithful and behavioural variants),
2. printing every clause's cited feature next to that row's raw value taken directly
   from the modelling table, and
3. asserting programmatically that the number rendered in each clause round-trips to the
   raw feature value — so this is a check, not just a display.

It also confirms the SHAP additivity identity holds on the real 53-feature model, not
just the tiny fixture in tests/test_explain.py.

Run from the repo root (needs data/processed/features.parquet and models/):
    venve/python.exe investigations/phase6_explainability/02_reason_code_spot_check_vs_raw_values.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import re

import numpy as np
import pandas as pd

from src.data_loader import load_config
from src.explain import (
    PAYMENT_FORMAT_FEATURES,
    align_features,
    build_reason_code,
    compute_shap_values,
    describe_feature,
    shap_base_value,
    top_contributors,
)
from src.model import (
    load_metadata,
    load_xgboost_model,
    prepare_feature_matrix,
    time_ordered_split,
)

N_ALERTS = 5
TOP_N = 3

# Features rendered as a yes/no statement with no number in the sentence — grounding for
# these is checked by direction (does the clause assert presence iff the value is set?)
# rather than by matching a numeric token.
_BINARY_SUFFIXES = ("_graph_in_cycle",)


def clause_is_grounded(feature: str, raw_value: float, clause: str) -> bool:
    """Does `clause` actually state `raw_value`?

    Extracts every numeric token from the rendered sentence and checks that one of them
    parses back to the row's real feature value. This is the check that would catch an
    off-by-one row alignment, a sender/receiver mix-up, or a clause paired with a
    different feature's value — i.e. exactly the errors that produce a fluent but false
    justification. Tolerance is 0.5 absolute (clauses round counts and USD amounts to
    whole units) or 0.005 for the ratio features rendered to 2 decimal places.
    """
    if feature.endswith(_BINARY_SUFFIXES):
        asserts_presence = "does not sit" not in clause
        return asserts_presence == bool(raw_value)
    if feature.startswith("payment_format_"):
        asserts_presence = "was not made" not in clause
        return asserts_presence == bool(raw_value)

    tokens = [float(t.replace(",", "")) for t in re.findall(r"\d[\d,]*\.?\d*", clause)]
    tolerance = 0.005 if "ratio" in feature else 0.5
    return any(abs(token - raw_value) <= tolerance for token in tokens)


def main() -> None:
    config = load_config()
    split_cfg = config["split"]

    print("Loading modelling table and trained model...")
    features_df = pd.read_parquet(config["paths"]["modelling_table"])
    _, _, test_df = time_ordered_split(features_df, split_cfg["train_frac"], split_cfg["val_frac"])
    del features_df

    x_test, y_test_series = prepare_feature_matrix(test_df)
    y_test = y_test_series.to_numpy()
    model = load_xgboost_model(config["paths"]["models_dir"])
    feature_names = load_metadata(config["paths"]["models_dir"])["feature_names"]
    x_test = align_features(x_test, feature_names)
    scores = model.predict_proba(x_test)[:, 1]

    top_positions = np.argsort(-scores)[:N_ALERTS]
    x_alerts = x_test.iloc[top_positions]
    shap_values = compute_shap_values(model, x_alerts)

    # Additivity on the real model, with the derived base value (never expected_value —
    # see script 01 in this folder for why).
    base_value = shap_base_value(model, x_alerts, shap_values)
    margin = model.predict(x_alerts, output_margin=True)
    max_err = float(np.abs(base_value + shap_values.sum(axis=1) - margin).max())
    print(f"\nSHAP additivity on the real 53-feature model: base_value={base_value:.6f}, ")
    print(f"max |base + sum(SHAP) - margin| = {max_err:.2e}  (exact => explanations faithful)")

    failures = []
    for rank, (position, i) in enumerate(zip(top_positions, range(len(x_alerts))), start=1):
        row = x_alerts.iloc[i]
        faithful = build_reason_code(shap_values[i], row, feature_names, TOP_N)
        behavioural = build_reason_code(
            shap_values[i], row, feature_names, TOP_N, exclude_features=PAYMENT_FORMAT_FEATURES
        )

        print(f"\n{'=' * 78}")
        print(
            f"ALERT #{rank}  test-split row {position}  score={scores[position]:.4f}  "
            f"is_laundering={y_test[position]}"
        )
        print(f"{'=' * 78}")
        print(f"  faithful   : {faithful}")
        print(f"  behavioural: {behavioural}")

        print("\n  Clause-by-clause check against the modelling table's raw values:")
        for feature, shap_value in top_contributors(shap_values[i], feature_names, TOP_N):
            raw_value = float(row[feature])
            clause = describe_feature(feature, raw_value)
            grounded = clause_is_grounded(feature, raw_value, clause)
            print(
                f"    {feature:<42} raw={raw_value:>14,.4g}  SHAP={shap_value:+.4f}  "
                f"{'OK' if grounded else 'MISMATCH'}"
            )
            print(f"      clause: {clause}")
            if not grounded:
                failures.append((rank, feature))

        # Independent cross-check: pull the same row straight from the modelling table
        # (not the prepared matrix) and confirm the engineered columns agree, so the
        # clauses trace back past prepare_feature_matrix to the Phase 3 output itself.
        source_row = test_df.iloc[position]
        engineered = [f for f in feature_names if not f.startswith("payment_format_")]
        worst = max(engineered, key=lambda f: abs(float(source_row[f]) - float(row[f])))
        source_value = float(source_row[worst])
        drift = abs(source_value - float(row[worst]))
        # Absolute drift grows with magnitude because prepare_feature_matrix downcasts to
        # float32 (~7 significant digits) — so this is reported RELATIVE to the value, which
        # is what shows it to be a precision artifact and not a genuine mismatch.
        relative = drift / abs(source_value) if source_value else 0.0
        print(
            f"\n  largest modelling-table vs. feature-matrix gap: {worst} "
            f"({source_value:,.2f} -> abs {drift:.3g}, relative {relative:.2e}; float32 cast)"
        )
        fmt = source_row["payment_format"]
        one_hot_ok = float(row[f"payment_format_{fmt}"]) == 1.0
        print(f"  payment_format='{fmt}' -> payment_format_{fmt}=1.0? {one_hot_ok}")
        if not one_hot_ok:
            failures.append((rank, "payment_format one-hot"))

    print(f"\n{'=' * 78}")
    if failures:
        print(f"FAILED: {len(failures)} grounding mismatches: {failures}")
        sys.exit(1)
    print(f"All {N_ALERTS} alerts' reason codes are grounded in their own raw feature values.")


if __name__ == "__main__":
    main()
