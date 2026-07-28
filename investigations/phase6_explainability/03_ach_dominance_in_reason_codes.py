"""Quantifies how badly `payment_format_ACH` degrades analyst-facing reason codes, and
what the behavioural variant recovers.

Supports the "a faithful reason code that says the same thing on every alert carries no
information" entry in reports/challenges.md's Phase 6 section, and the reason-code numbers
in reports/results.md's Phase 6 section.

Phase 4 already established WHY ACH dominates (86.6% of labelled laundering in HI-Small
uses ACH — see investigations/phase4_modelling/01_payment_format_ach_correlation.py) and
HOW MUCH model performance rests on it (the ablation in that folder's script 02). Neither
of those answers the Phase 6 question, which is about the explanation layer specifically:
if the top SHAP contributor is the same categorical on nearly every alert, then a reason
code faithful to raw SHAP is constant across the queue, and a constant explanation cannot
help an analyst decide which of 10,011 alerts to open first.

This script measures:
- how often payment_format_ACH leads the faithful reason code,
- how much distinct information the faithful vs. behavioural variants carry across the
  alert queue (count of distinct leading features, and distinct whole sentences),
- which features lead the behavioural variant instead.

Run from the repo root (needs data/processed/features.parquet and models/):
    venve/python.exe investigations/phase6_explainability/03_ach_dominance_in_reason_codes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter

import numpy as np
import pandas as pd

from src.data_loader import load_config
from src.explain import (
    PAYMENT_FORMAT_FEATURES,
    align_features,
    build_reason_codes,
    compute_shap_values,
    top_contributors,
)
from src.model import (
    load_metadata,
    load_xgboost_model,
    prepare_feature_matrix,
    time_ordered_split,
)

# The Phase 5 headline operating point: the model raises 10,011 alerts at the rules
# baseline's own test-split recall of 68.8% (see reports/results.md). Hardcoded here
# rather than recomputed, because deriving it requires re-running the rules baseline over
# the full 5.08M-row dataset — this script only needs "the size of the queue an analyst
# would actually be handed", and the reason-code statistics below are insensitive to a
# few hundred rows either way (spot-checked at 5,000 and 10,011).
ALERT_QUEUE_SIZE = 10_011
TOP_N = 3


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

    queue_positions = np.argsort(-scores)[:ALERT_QUEUE_SIZE]
    x_queue = x_test.iloc[queue_positions]
    print(
        f"Alert queue: top {ALERT_QUEUE_SIZE:,} of {len(x_test):,} test transactions "
        f"({int(y_test[queue_positions].sum())} true positives). Computing SHAP..."
    )
    shap_values = compute_shap_values(model, x_queue)

    # --- How often does ACH lead the faithful explanation? ----------------------------
    faithful_leads = Counter()
    behavioural_leads = Counter()
    for i in range(len(x_queue)):
        faithful = top_contributors(shap_values[i], feature_names, top_n=1)
        behavioural = top_contributors(
            shap_values[i], feature_names, top_n=1, exclude_features=PAYMENT_FORMAT_FEATURES
        )
        faithful_leads[faithful[0][0] if faithful else "<none>"] += 1
        behavioural_leads[behavioural[0][0] if behavioural else "<none>"] += 1

    n = len(x_queue)
    ach = faithful_leads["payment_format_ACH"]
    print(f"\n{'=' * 78}\nFAITHFUL reason code — leading feature across the queue\n{'=' * 78}")
    for feature, count in faithful_leads.most_common(8):
        print(f"  {feature:<44} {count:>7,}  ({count / n:6.2%})")
    print(f"\n  payment_format_ACH leads {ach:,}/{n:,} = {ach / n:.2%} of alerts.")

    print(f"\n{'=' * 78}\nBEHAVIOURAL reason code — leading feature across the queue\n{'=' * 78}")
    for feature, count in behavioural_leads.most_common(8):
        print(f"  {feature:<44} {count:>7,}  ({count / n:6.2%})")

    # --- Information content: how many genuinely different sentences does each produce? -
    faithful_codes = build_reason_codes(shap_values, x_queue, TOP_N)
    behavioural_codes = build_reason_codes(
        shap_values, x_queue, TOP_N, exclude_features=PAYMENT_FORMAT_FEATURES
    )

    print(f"\n{'=' * 78}\nInformation content across the {n:,}-alert queue\n{'=' * 78}")
    print(f"{'':<24}{'faithful':>14}{'behavioural':>14}")
    print(f"{'distinct leading feature':<24}{len(faithful_leads):>14,}{len(behavioural_leads):>14,}")
    print(
        f"{'distinct full sentences':<24}"
        f"{faithful_codes.nunique():>14,}{behavioural_codes.nunique():>14,}"
    )
    most_common_faithful = faithful_codes.value_counts().iloc[0]
    most_common_behavioural = behavioural_codes.value_counts().iloc[0]
    print(
        f"{'most repeated sentence':<24}"
        f"{most_common_faithful:>14,}{most_common_behavioural:>14,}"
    )

    print(
        "\nReading this: the faithful variant is honest about what the model uses, but it\n"
        "opens with the same clause on nearly every alert, so it cannot help an analyst\n"
        "triage within the queue. The behavioural variant drops the payment-format\n"
        "one-hots from the sentence only (never from the model) and surfaces the Phase 3\n"
        "account/window and graph features instead. Both are reported; see\n"
        "reports/results.md's Phase 6 section."
    )


if __name__ == "__main__":
    main()
