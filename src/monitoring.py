"""PSI/CSI drift checks on feature and score distributions across time slices.

Implemented in PLAN.md Phase 7. This module answers the question a model-risk function
asks before it will let a model stay in production: *is the data still the data the model
was trained on?* A model that was excellent on September 1st is not automatically
excellent on September 18th, and at ~0.10% prevalence you cannot wait for labels to tell
you it degraded — SAR outcomes arrive weeks-to-months later (see reports/challenges.md's
architecture entry). Distribution monitoring is the only signal available in the meantime.

Two metrics, one formula:

- **PSI (Population Stability Index)** — applied to the *model's output score*. Answers
  "is the alert queue still made of the same kind of thing?"
- **CSI (Characteristic Stability Index)** — the identical calculation applied to an
  *input feature*. Answers "which input moved?" PSI tells you something broke; CSI tells
  you where. They are the same function (`population_stability_index`) with different
  inputs, and are named separately here only because that is the vocabulary a bank's
  model-risk documentation uses.

Both are `sum over bins of (current_pct - reference_pct) * ln(current_pct / reference_pct)`,
read against the industry-conventional bands in `config.yaml` (<0.10 stable, 0.10-0.25
moderate shift, >0.25 significant shift). Those bands are convention, not theory — they
come from credit-scorecard practice and are used here because they are what a reviewer
will expect to see, not because 0.25 is derived from anything.

Three implementation choices that are load-bearing, not incidental:

- **Bin edges come from the reference distribution only.** Re-deriving quantile edges on
  each current slice would make every slice look stable by construction — deciles of any
  distribution hold 10% each, so the PSI would be ~0 no matter how far the data moved.
  `quantile_bin_edges` is called once on the reference and reused for every comparison.

- **Empty bins are floored, not dropped.** A current slice can put zero mass in a bin the
  reference filled, which sends `ln(0)` to `-inf` and the PSI term to `+inf`. Dropping
  such bins would hide the largest real shifts (a bin emptying out is drift, not a
  nuisance), so proportions are floored at `epsilon` instead. This bounds the per-bin term
  at roughly `reference_pct * ln(reference_pct / epsilon)`, which is large-but-finite —
  the intended behaviour. PLAN.md's Phase 7 check ("PSI computes without NaN/inf on real
  data") is exactly this failure mode, and it does fire on this dataset.

- **Duplicate quantile edges are collapsed, and the effective bin count is reported.**
  Many features here are counts that are zero for the large majority of rows
  (`structuring_score_1d`, `*_graph_in_cycle`, most distinct-counterparty windows), so
  several deciles of the reference share the same edge value. `quantile_bin_edges`
  de-duplicates them rather than emitting zero-width bins, which means a feature can
  legitimately be scored on 3 bins instead of 10. `psi_table` carries `n_bins_effective`
  so that a suspiciously low CSI can be read as "this feature is too tied to bin finely"
  rather than mistaken for stability.

- **Low-cardinality features are binned by category, not by quantile.** Quantile binning
  a binary feature is not merely imprecise, it is silently blind: every quantile of a
  0/1 column lands on the same value, the edges collapse to a single bin, and the CSI is
  0.0 no matter what happened. This was not hypothetical here — the first run of this
  module reported CSI 0.0000 for `payment_format_Reinvestment` on a train->test
  comparison where that format went from 15.8% of transactions to *zero*, and 0.0000 for
  `payment_format_ACH`, the single most important feature in the model (Phase 4). A drift
  monitor that cannot see the model's top feature vanish is worse than no monitor, since
  it reports reassurance. `characteristic_stability` therefore routes any feature with at
  most `max_categorical_cardinality` distinct reference values through
  `category_share_psi`, which compares per-category shares directly, and reports which
  method it used per feature in the `method` column.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# Conventional credit-risk reading of a PSI/CSI value. Overridable from config.yaml's
# `monitoring.psi_thresholds`; these are the defaults so the functions are usable
# standalone (in tests, notebooks) without threading config through every call.
DEFAULT_PSI_THRESHOLDS = {"moderate": 0.10, "significant": 0.25}

# Proportion floor for empty bins — see this module's docstring. Small enough not to
# distort a populated bin, large enough to keep an emptied bin's contribution finite.
# The choice affects the magnitude of an extreme PSI but not which band it lands in;
# verified empirically in investigations/phase7_monitoring/.
DEFAULT_EPSILON = 1e-6


def _as_float_array(values: Sequence[float] | np.ndarray | pd.Series, label: str) -> np.ndarray:
    """Coerce to a 1-D float array, refusing NaN/inf loudly.

    `np.histogram` silently discards NaN — it belongs to no bin — which would quietly
    shrink the denominator and produce a plausible-looking PSI computed on a different
    population than the caller thinks. In a monitoring module whose entire job is to
    notice when data changed, silently dropping the rows that changed is the worst
    available failure. Raising is the correct behaviour: the caller must decide whether
    a NaN is a missing feature or a broken upstream join.
    """
    array = np.asarray(values, dtype=float).ravel()
    if array.size == 0:
        raise ValueError(f"{label}: empty array, cannot compute a distribution")
    if not np.isfinite(array).all():
        n_bad = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"{label}: contains {n_bad:,} non-finite value(s) (NaN or inf). "
            "Handle them explicitly before computing PSI — histogram binning would "
            "otherwise drop them silently and change the denominator."
        )
    return array


def quantile_bin_edges(
    reference: Sequence[float] | np.ndarray | pd.Series, n_bins: int
) -> np.ndarray:
    """Quantile bin edges derived from the REFERENCE distribution only.

    Outer edges are set to -inf/+inf so that a current slice containing values beyond the
    reference's observed range still bins (rather than falling outside every bin and
    being dropped) — a new maximum is drift worth measuring, not an error.

    Duplicate edges from tied values are collapsed via `np.unique`, so the returned array
    may describe fewer than `n_bins` bins; see this module's docstring. A constant
    reference degenerates to the single bin `[-inf, inf]`, against which every PSI is 0 —
    correct, if uninformative: a feature with one value cannot shift in distribution,
    only in whether it is still constant.
    """
    if n_bins < 1:
        raise ValueError(f"quantile_bin_edges: n_bins must be >= 1, got {n_bins}")
    array = _as_float_array(reference, "quantile_bin_edges(reference)")

    edges = np.unique(np.quantile(array, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return np.array([-np.inf, np.inf])

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_proportions(array: np.ndarray, edges: np.ndarray, epsilon: float) -> np.ndarray:
    counts, _ = np.histogram(array, bins=edges)
    proportions = counts / counts.sum()
    return np.maximum(proportions, epsilon)


def psi_table(
    reference: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
    epsilon: float = DEFAULT_EPSILON,
) -> pd.DataFrame:
    """Per-bin PSI breakdown: where the shift is, not just how big it is.

    Returned columns: `bin_lower`, `bin_upper`, `reference_pct`, `current_pct`,
    `psi_contribution`. Summing `psi_contribution` reproduces
    `population_stability_index` exactly (same code path), which is what makes a headline
    PSI explainable — "0.31, and 0.27 of it is the top decile emptying out" is
    actionable in a way that "0.31" is not.
    """
    ref_array = _as_float_array(reference, "psi_table(reference)")
    cur_array = _as_float_array(current, "psi_table(current)")

    edges = quantile_bin_edges(ref_array, n_bins)
    ref_pct = _bin_proportions(ref_array, edges, epsilon)
    cur_pct = _bin_proportions(cur_array, edges, epsilon)

    return pd.DataFrame(
        {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "reference_pct": ref_pct,
            "current_pct": cur_pct,
            "psi_contribution": (cur_pct - ref_pct) * np.log(cur_pct / ref_pct),
        }
    )


def population_stability_index(
    reference: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI of `current` against `reference`, binned on the reference's quantiles.

    Note the metric is symmetric — `(a - e) * ln(a / e)` is unchanged by swapping a and e
    — but this function is *not*, because the bins are the reference's. Which argument is
    the reference is a real modelling decision (here: the training split), not an
    argument-order detail.
    """
    return float(psi_table(reference, current, n_bins, epsilon)["psi_contribution"].sum())


def category_share_psi(
    reference: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """PSI over per-category shares, for features quantile binning cannot see.

    Each distinct value is its own bin, so a binary flag is compared as
    "15.8% ones -> 0.0% ones" rather than collapsed into a single degenerate bin. The
    bin set is the *union* of reference and current categories: a category that appears
    only in the current slice is new-category drift and must contribute, and one that
    appears only in the reference has disappeared and must contribute too. Both are
    floored at `epsilon` for the same reason as `_bin_proportions`.
    """
    ref_array = _as_float_array(reference, "category_share_psi(reference)")
    cur_array = _as_float_array(current, "category_share_psi(current)")

    categories = np.union1d(np.unique(ref_array), np.unique(cur_array))
    ref_pct = np.maximum(
        np.array([(ref_array == c).sum() for c in categories]) / ref_array.size, epsilon
    )
    cur_pct = np.maximum(
        np.array([(cur_array == c).sum() for c in categories]) / cur_array.size, epsilon
    )
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def classify_psi(psi: float, thresholds: dict[str, float] | None = None) -> str:
    """Map a PSI/CSI value onto the conventional `stable` / `moderate` / `significant`
    bands. Returning a label rather than a bare bool keeps the middle band visible: a
    0.15 is a "look at this next week", not a "stop the model", and collapsing it into a
    pass/fail flag would lose that.
    """
    bands = thresholds or DEFAULT_PSI_THRESHOLDS
    if psi >= bands["significant"]:
        return "significant"
    if psi >= bands["moderate"]:
        return "moderate"
    return "stable"


def characteristic_stability(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    n_bins: int = 10,
    epsilon: float = DEFAULT_EPSILON,
    thresholds: dict[str, float] | None = None,
    max_categorical_cardinality: int = 10,
) -> pd.DataFrame:
    """CSI for every feature, ranked most-drifted first — the "which input moved?" table.

    Features with at most `max_categorical_cardinality` distinct values in the reference
    are scored by `category_share_psi` rather than by quantile bins; see this module's
    docstring for why that is a correctness fix and not a refinement. The `method` column
    records which path each feature took, so the table is auditable rather than magic.

    Skips nothing silently: a feature that is constant in *both* frames still appears,
    with `csi = 0.0`, so a reader can tell "did not move" apart from "was not checked".
    Columns present in `reference_df` but missing from `current_df` raise, because a
    vanished feature is a pipeline break and the loudest possible failure is the useful one.
    """
    columns = list(feature_columns) if feature_columns is not None else list(reference_df.columns)

    missing = [c for c in columns if c not in current_df.columns]
    if missing:
        raise ValueError(
            f"characteristic_stability: {len(missing)} column(s) missing from current_df: "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}"
        )

    rows = []
    for column in columns:
        ref_values = reference_df[column].to_numpy(dtype=float)
        cur_values = current_df[column].to_numpy(dtype=float)

        n_distinct = len(np.unique(ref_values))
        if n_distinct <= max_categorical_cardinality:
            csi = category_share_psi(ref_values, cur_values, epsilon)
            method, n_bins_effective = "category_share", n_distinct
        else:
            csi = population_stability_index(ref_values, cur_values, n_bins, epsilon)
            method = "quantile"
            n_bins_effective = len(quantile_bin_edges(ref_values, n_bins)) - 1

        rows.append(
            {
                "feature": column,
                "csi": csi,
                "band": classify_psi(csi, thresholds),
                "method": method,
                "n_bins_effective": n_bins_effective,
                "reference_mean": float(ref_values.mean()),
                "current_mean": float(cur_values.mean()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("csi", ascending=False)
        .reset_index(drop=True)
    )


def psi_over_time(
    reference: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    timestamps: pd.Series,
    freq: str = "D",
    n_bins: int = 10,
    epsilon: float = DEFAULT_EPSILON,
    min_slice_size: int = 1000,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """PSI of each time slice of `current` against one fixed `reference` — the
    "score PSI over time" series PLAN.md Phase 7 asks for in `results.md`.

    The reference is held fixed rather than compared slice-to-consecutive-slice on
    purpose. Chained adjacent comparisons make gradual drift invisible: each day looks
    like the day before it right up until the model is scoring a population it has never
    seen. Anchoring every slice to the training distribution is what a production monitor
    actually does.

    Slices smaller than `min_slice_size` are reported with `psi = NaN` and kept as rows
    rather than dropped — a thin day is a fact about the data, and silently omitting it
    would leave an unexplained gap in the plot. The floor exists because PSI on a few
    hundred rows is dominated by sampling noise, not drift: with 10 bins, a slice needs
    enough mass per bin for the proportions to mean anything.
    """
    cur_array = _as_float_array(current, "psi_over_time(current)")
    if len(timestamps) != len(cur_array):
        raise ValueError(
            f"psi_over_time: timestamps ({len(timestamps):,}) and current "
            f"({len(cur_array):,}) must be the same length"
        )

    ref_array = _as_float_array(reference, "psi_over_time(reference)")
    periods = pd.Series(pd.to_datetime(timestamps).to_numpy()).dt.to_period(freq)

    rows = []
    for period, positions in periods.groupby(periods).groups.items():
        slice_values = cur_array[np.asarray(positions, dtype=int)]
        if len(slice_values) < min_slice_size:
            psi = float("nan")
            band = "insufficient_data"
        else:
            psi = population_stability_index(ref_array, slice_values, n_bins, epsilon)
            band = classify_psi(psi, thresholds)
        rows.append(
            {
                "period": str(period),
                "n_rows": len(slice_values),
                "psi": psi,
                "band": band,
            }
        )

    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import load_config
    from src.model import load_xgboost_model, prepare_feature_matrix, time_ordered_split

    config = load_config()
    split_cfg = config["split"]
    mon_cfg = config["monitoring"]
    n_bins = mon_cfg["psi_bins"]
    epsilon = float(mon_cfg["psi_epsilon"])
    thresholds = mon_cfg["psi_thresholds"]

    print(f"Loading modelling table from {config['paths']['modelling_table']} ...")
    features_df = pd.read_parquet(config["paths"]["modelling_table"])
    train_df, val_df, test_df = time_ordered_split(
        features_df, split_cfg["train_frac"], split_cfg["val_frac"]
    )
    del features_df

    model = load_xgboost_model(config["paths"]["models_dir"])

    # Reference = the TRAINING split. Everything below is measured against the
    # distribution the model actually learned from, never against its own neighbours.
    x_train, _ = prepare_feature_matrix(train_df)
    train_scores = model.predict_proba(x_train)[:, 1]
    print(f"Reference (train): {len(train_scores):,} rows")

    x_val, _ = prepare_feature_matrix(val_df)
    val_scores = model.predict_proba(x_val)[:, 1]
    del x_val

    x_test, _ = prepare_feature_matrix(test_df)
    test_scores = model.predict_proba(x_test)[:, 1]

    # --- Score PSI: train -> val, train -> test --------------------------------------
    print("\n=== Score PSI vs. training distribution ===")
    for name, scores in (("val", val_scores), ("test", test_scores)):
        psi = population_stability_index(train_scores, scores, n_bins, epsilon)
        print(f"  {name:5s}: PSI = {psi:.4f}  ({classify_psi(psi, thresholds)})")

    print("\nPer-bin breakdown, train -> test:")
    breakdown = psi_table(train_scores, test_scores, n_bins, epsilon)
    print(breakdown.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # --- Score PSI over time ----------------------------------------------------------
    print(f"\n=== Score PSI by {mon_cfg['time_slice_freq']} (val+test, vs. train) ===")
    later_scores = np.concatenate([val_scores, test_scores])
    later_timestamps = pd.concat([val_df["timestamp"], test_df["timestamp"]], ignore_index=True)
    over_time = psi_over_time(
        train_scores,
        later_scores,
        later_timestamps,
        freq=mon_cfg["time_slice_freq"],
        n_bins=n_bins,
        epsilon=epsilon,
        min_slice_size=mon_cfg["min_slice_size"],
        thresholds=thresholds,
    )
    print(over_time.to_string(index=False))

    # --- Feature CSI: train -> test ---------------------------------------------------
    print("\n=== Top 15 features by CSI (train -> test) ===")
    csi = characteristic_stability(x_train, x_test, n_bins=n_bins, epsilon=epsilon,
                                   thresholds=thresholds)
    print(csi.head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBand counts across all {len(csi)} features:")
    print(csi["band"].value_counts().to_string())

    # --- Figure for reports/results.md ------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    scoreable = over_time[over_time["psi"].notna()]
    thin = over_time[over_time["psi"].isna()]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhspan(0, thresholds["moderate"], color="#2e7d32", alpha=0.08)
    ax.axhspan(thresholds["moderate"], thresholds["significant"], color="#f9a825", alpha=0.12)
    ax.axhspan(thresholds["significant"], 0.45, color="#c62828", alpha=0.10)
    ax.axhline(thresholds["moderate"], color="#f9a825", ls="--", lw=1)
    ax.axhline(thresholds["significant"], color="#c62828", ls="--", lw=1)

    ax.plot(scoreable["period"], scoreable["psi"], marker="o", color="#1565c0", lw=2,
            label="score PSI vs. training distribution")
    for _, row in thin.iterrows():
        ax.annotate("", xy=(row["period"], 0), xytext=(row["period"], 0))
    ax.scatter(thin["period"], np.zeros(len(thin)), marker="x", color="#757575", zorder=3,
               label=f"too thin to score (<{mon_cfg['min_slice_size']:,} rows)")

    step_psi = scoreable.loc[scoreable["period"] == "2022-09-09", "psi"]
    if not step_psi.empty:
        ax.annotate(
            "7-day graph lookback evicts 2022-09-01\n(22% of all edges) — see challenges.md",
            xy=("2022-09-09", float(step_psi.iloc[0])), xytext=("2022-09-11", 0.37),
            arrowprops={"arrowstyle": "->", "color": "#424242"}, fontsize=9, color="#424242",
        )
    ax.annotate(
        "real population change here\n(volume collapses, ~59% laundering)\n"
        "but every slice is too thin to score",
        xy=("2022-09-14", 0.0), xytext=("2022-09-12", 0.13),
        arrowprops={"arrowstyle": "->", "color": "#757575"}, fontsize=8.5, color="#616161",
    )
    ax.text(0.995, thresholds["significant"] + 0.008, "significant", ha="right", fontsize=8,
            color="#c62828", transform=ax.get_yaxis_transform())
    ax.text(0.995, thresholds["moderate"] + 0.008, "moderate", ha="right", fontsize=8,
            color="#f9a825", transform=ax.get_yaxis_transform())

    ax.set_title("Score PSI by day, val+test vs. the training distribution (HI-Small)")
    ax.set_ylabel("PSI")
    ax.set_ylim(0, 0.45)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    figure_path = figures_dir / "07_score_psi_over_time.png"
    fig.savefig(figure_path, dpi=150)
    print(f"\nSaved {figure_path}")
