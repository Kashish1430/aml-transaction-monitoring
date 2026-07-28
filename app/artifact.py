"""Loading, filtering and waterfall-assembly for the demo artifacts.

Implemented in PLAN.md Phase 9. Split out of `streamlit_app.py` deliberately: a Streamlit
script executes top-to-bottom on import, so any logic living there cannot be imported by a
test without spinning up the whole UI. Everything here is a pure function over a DataFrame
or a dict, which is what `tests/test_app_artifact.py` exercises — the UI module is then
thin enough that reading it is sufficient review.

This module must not import `src/`. The deployed app installs only
`app/requirements.txt` (streamlit, pandas, pyarrow, plotly); `src/` pulls in xgboost,
shap and scikit-learn, none of which are present on Streamlit Community Cloud and none of
which the app needs, because every model output it displays was precomputed offline by
`scripts/build_demo_artifact.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
ARTIFACT_PATH = DATA_DIR / "demo_alerts.parquet"
METRICS_PATH = DATA_DIR / "demo_metrics.json"

# Paths are resolved from this file, not the working directory: Streamlit Community Cloud
# runs the entrypoint from the repo root, while `streamlit run app/streamlit_app.py` from
# a subdirectory would otherwise resolve them somewhere else entirely.


def load_alerts(path: Path | str = ARTIFACT_PATH) -> pd.DataFrame:
    """The demo alert table, sorted by the shipped rank.

    Sorted, never re-ranked. `rank` and `model_score` were computed over the complete
    1,015,669-row test split before sampling (PLAN.md Phase 8), so they are true values;
    recomputing a rank over the 30,000 shipped rows would silently redefine "rank 1" from
    "highest-scored transaction in the test split" to "highest-scored row that survived
    sampling" — a weaker claim displayed as if it were the stronger one.
    """
    return pd.read_parquet(path).sort_values("rank").reset_index(drop=True)


def load_metrics(path: Path | str = METRICS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_names(alerts: pd.DataFrame) -> list[str]:
    """Model feature names, recovered from the `feat_*` columns.

    Read off the artifact rather than hardcoded, so a rebuilt artifact with different
    features can't leave this module describing the previous one.
    """
    return [c[len("feat_") :] for c in alerts.columns if c.startswith("feat_")]


def display_columns(alerts: pd.DataFrame) -> list[str]:
    """Everything except the `feat_*`/`shap_*` blocks — i.e. what a human reads."""
    return [c for c in alerts.columns if not c.startswith(("feat_", "shap_"))]


def apply_filters(
    alerts: pd.DataFrame,
    queue_only: bool = True,
    include_tail: bool = True,
    typologies: list[str] | None = None,
    payment_formats: list[str] | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    min_score: float = 0.0,
    search: str = "",
) -> pd.DataFrame:
    """Filter the queue. Order is preserved (already rank-sorted), never recomputed.

    `typologies`/`payment_formats` of None mean "no constraint", which is not the same as
    an empty list — an empty multiselect in the UI means the user has deselected
    everything and should see nothing, whereas an untouched filter should see everything.
    Collapsing the two would make clearing a filter look like a no-op.
    """
    mask = pd.Series(True, index=alerts.index)

    if queue_only:
        mask &= alerts["in_alert_queue"]
    if not include_tail:
        mask &= ~alerts["is_tail_population"]
    if typologies is not None:
        mask &= alerts["pattern_type"].isin(typologies)
    if payment_formats is not None:
        mask &= alerts["payment_format"].isin(payment_formats)
    if date_range is not None:
        start, end = date_range
        mask &= alerts["timestamp"].between(pd.Timestamp(start), pd.Timestamp(end))
    if min_score > 0:
        mask &= alerts["model_score"] >= min_score
    if search:
        needle = search.strip()
        if needle:
            mask &= (
                alerts["from_account_key"].str.contains(needle, case=False, na=False)
                | alerts["to_account_key"].str.contains(needle, case=False, na=False)
                | alerts["from_bank"].astype(str).eq(needle)
                | alerts["to_bank"].astype(str).eq(needle)
            )

    return alerts[mask]


def queue_stats(filtered: pd.DataFrame, metrics: dict) -> dict:
    """Headline counters for the current filter, labelled by what they actually count.

    `n_shown` is a count of *sampled* rows wherever the filter reaches below the alert
    queue, because only the queue itself and the positives ship complete (PLAN.md Phase
    8). `population_exact` says whether the visible count is a population count or a
    sample count, so the UI can label it rather than implying precision it doesn't have.
    """
    n_shown = len(filtered)
    n_positives = int(filtered["is_laundering"].sum())
    return {
        "n_shown": n_shown,
        "n_positives": n_positives,
        "precision": n_positives / n_shown if n_shown else float("nan"),
        "population_exact": bool(filtered["in_alert_queue"].all()) and n_shown > 0,
        "queue_size": metrics["headline"]["model_alerts"],
    }


def prettify_feature(name: str) -> str:
    """`sender_out_7d_distinct_counterparties` -> `sender out 7d distinct counterparties`.

    Deliberately a cosmetic transform only. The analyst-facing *interpretation* of a
    feature lives in the precomputed reason codes (built by `src.explain.describe_feature`
    against each row's raw value); restating it here would duplicate that logic in a module
    that can't import it, and a second copy of an explanation is a second copy that can
    drift out of agreement with the first.
    """
    return name.replace("payment_format_", "payment format: ").replace("_", " ")


def waterfall_data(
    row: pd.Series, names: list[str], base_value: float, top_n: int = 10
) -> pd.DataFrame:
    """Rows for a SHAP waterfall of one alert: the `top_n` largest contributions by
    magnitude, the rest collapsed into one bar, ordered most-positive first.

    Contributions are in the model's margin (log-odds) space, which is the space they are
    additive in — `base_value + sum(contributions) == margin`. The aggregated remainder is
    a real sum, not a residual: collapsing it keeps the identity exact, so the bars still
    add up to the score shown beside them.
    """
    contributions = np.array([row[f"shap_{name}"] for name in names], dtype="float64")
    order = np.argsort(-np.abs(contributions))
    head, tail = order[:top_n], order[top_n:]

    rows = [
        {
            "feature": prettify_feature(names[i]),
            "value": float(row[f"feat_{names[i]}"]),
            "contribution": float(contributions[i]),
        }
        for i in head
    ]
    if len(tail):
        rows.append(
            {
                "feature": f"{len(tail)} other features",
                "value": float("nan"),
                "contribution": float(contributions[tail].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("contribution", ascending=False).reset_index(drop=True)


def score_from_margin(margin: float) -> float:
    return float(1.0 / (1.0 + np.exp(-margin)))


def reconstructed_score(row: pd.Series, names: list[str], base_value: float) -> float:
    """The score implied by this row's shipped contributions.

    Used by the UI to display the waterfall's endpoint. It must agree with the shipped
    `model_score` (CI asserts max abs error 5.2e-07 across the artifact); computing it
    here rather than reusing `model_score` means the displayed waterfall is shown to
    terminate where the ranking says it does, instead of being drawn next to a number it
    was never checked against.
    """
    margin = base_value + sum(float(row[f"shap_{name}"]) for name in names)
    return score_from_margin(margin)


def typology_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["recall_per_typology"])


def pr_curve_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["pr_curve"])


def psi_frame(metrics: dict) -> pd.DataFrame:
    """PSI series with thin slices kept as rows.

    `psi_over_time` reports slices under `min_slice_size` as null rather than dropping
    them (Phase 7): a thin day is a fact about the data, and silently omitting it leaves an
    unexplained gap in the chart. The UI plots them on a separate marker track.
    """
    return pd.DataFrame(metrics["psi_over_time"])


def importance_frame(metrics: dict) -> pd.DataFrame:
    df = pd.DataFrame(metrics["shap"]["global_importance"])
    df["feature"] = df["feature"].map(prettify_feature)
    return df
