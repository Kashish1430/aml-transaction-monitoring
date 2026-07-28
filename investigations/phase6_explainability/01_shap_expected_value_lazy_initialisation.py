"""Reproduces the SHAP base-value trap behind `src/explain.shap_base_value`.

Supports the "SHAP's `expected_value` is lazily corrected — deriving the base value is
the only safe option" entry in reports/challenges.md's Phase 6 section.

The finding: on shap 0.45.1 + xgboost 2.0.3, `shap.TreeExplainer(model).expected_value`
returns `array([logit(base_score)])` immediately after construction, and is silently
replaced with a *different*, correct scalar during the first `shap_values()` call. Code
that reads the attribute before explaining anything — the natural reading order, and
what a "reconstruct the margin to prove faithfulness" check does — gets a value wrong by
a constant offset, with no warning and no exception.

Why it matters here: the per-feature contributions are exact either way (this script
verifies them bit-identical against XGBoost's native `pred_contribs`), so reason codes,
which use only contributions and their ranking, were never affected. But a SHAP waterfall
or score decomposition anchored on the wrong base value would be quietly wrong, and the
constant offset is small enough to pass an eyeball check.

Run from the repo root:
    venve/python.exe investigations/phase6_explainability/01_shap_expected_value_lazy_initialisation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

SEED = 42


def main() -> None:
    print(f"shap {shap.__version__}, xgboost {xgb.__version__}\n")

    # A small, self-contained model — the behaviour is a property of the shap/xgboost
    # pairing, not of this project's data, so the repro deliberately needs no dataset.
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(rng.random((500, 6)), columns=[f"f{i}" for i in range(6)])
    y = (x["f0"] + x["f1"] > 1.0).astype(int)
    model = xgb.XGBClassifier(n_estimators=25, max_depth=3, random_state=SEED)
    model.fit(x, y)

    config = json.loads(model.get_booster().save_config())
    base_score = float(config["learner"]["learner_model_param"]["base_score"])
    logit_base_score = float(np.log(base_score / (1 - base_score)))
    print(f"model base_score           : {base_score:.7f}")
    print(f"logit(base_score)          : {logit_base_score:.7f}")

    # --- The trap: read expected_value before vs. after the first shap_values() call ---
    explainer = shap.TreeExplainer(model)
    raw_before = explainer.expected_value
    before = float(np.asarray(raw_before).reshape(-1)[0])
    print(f"\nexpected_value BEFORE call : {raw_before!r}")

    shap_values = explainer.shap_values(x)
    raw_after = explainer.expected_value
    after = float(np.asarray(raw_after).reshape(-1)[0])
    print(f"expected_value AFTER  call : {raw_after!r}")
    print(f"  -> silently changed by     {after - before:+.7f}")
    print(f"  -> BEFORE == logit(base_score)? {np.isclose(before, logit_base_score)}")

    # --- Which one actually satisfies additivity? -------------------------------------
    margin = model.predict(x, output_margin=True)
    derived_bias = margin - shap_values.sum(axis=1)
    print(
        f"\nderived bias (margin - sum SHAP): mean {derived_bias.mean():.7f}, "
        f"spread {derived_bias.max() - derived_bias.min():.2e} (constant => additive)"
    )

    for label, candidate in (("BEFORE-call value", before), ("AFTER-call value", after)):
        err = np.abs(candidate + shap_values.sum(axis=1) - margin).max()
        verdict = "additivity HOLDS" if err < 1e-4 else "additivity FAILS"
        print(f"  {label:>18}: max reconstruction error {err:.7f}  -> {verdict}")

    # --- The contributions themselves are exact, regardless of the base value ---------
    native = model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)
    print(
        f"\nshap contributions vs. xgboost native pred_contribs: "
        f"max abs difference {np.abs(native[:, :-1] - shap_values).max():.2e}"
    )
    print(f"native bias column (constant)                      : {native[0, -1]:.7f}")
    print(
        "\nConclusion: contributions are exact and safe to rank (so reason codes are\n"
        "unaffected), but the base value must be DERIVED, not read from expected_value.\n"
        "src/explain.shap_base_value does that and asserts additivity while it's at it."
    )


if __name__ == "__main__":
    main()
