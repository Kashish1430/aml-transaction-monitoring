"""Analyst Alert-Triage Dashboard: Alert Queue / Alert Detail / Model Performance / About.

Implemented in PLAN.md Phase 9. This module is UI only — every transformation it performs
lives in `app/artifact.py`, which is unit-tested (`tests/test_app_artifact.py`); what's
left here is layout and chart construction, which is reviewed by reading rather than by
assertion.

It computes nothing about the model. Scores, ranks, the alert queue, both reason-code
variants and all 54 SHAP contributions per row were precomputed offline by
`scripts/build_demo_artifact.py` against the full pipeline. Streamlit Community Cloud's
free tier (~1 CPU / ~1GB RAM) cannot hold the 728MB modelling table, so the app reads a
15.3MB parquet and a 36KB JSON and does nothing heavier than filtering them.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# `streamlit run app/streamlit_app.py` puts the SCRIPT's directory on sys.path, not the
# repo root — so `from app import artifact` raises ModuleNotFoundError under the real
# server even though it resolves fine under pytest (whose pythonpath is "."). Adding the
# repo root explicitly makes the import work identically in both, rather than leaving the
# deployed app depending on a path Streamlit never sets. Must precede the `app` import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import artifact  # noqa: E402

st.set_page_config(page_title="AML Alert Triage", page_icon="🔍", layout="wide")

QUEUE_COLOUR = "#1565c0"
POSITIVE_COLOUR = "#c62828"
MUTED = "#757575"


@st.cache_data
def _alerts() -> pd.DataFrame:
    return artifact.load_alerts()


@st.cache_data
def _metrics() -> dict:
    return artifact.load_metrics()


alerts = _alerts()
metrics = _metrics()
names = artifact.feature_names(alerts)
base_value = metrics["shap"]["base_value"]

# --- Sidebar filters --------------------------------------------------------------------

st.sidebar.title("Filters")
st.sidebar.caption(
    f"{metrics['headline']['model_alerts']:,} alerts from a "
    f"{metrics['test_split']['n_rows']:,}-transaction held-out period."
)

queue_only = st.sidebar.toggle(
    "Alert queue only",
    value=True,
    help=(
        "The equal-recall alert queue — the model's actual output at the operating point "
        "behind the headline result. Switch off to browse sampled transactions below the "
        "cut-off."
    ),
)
include_tail = st.sidebar.toggle(
    "Include anomalous tail",
    value=True,
    help=(
        "From 2022-09-11 the dataset becomes a different population: daily volume "
        "collapses and the laundering rate jumps to ~59%. It is 0.109% of the held-out "
        "period but 3.1% of this demo sample, because every known positive is kept."
    ),
)

typology_options = list(alerts["pattern_type"].cat.categories)
typologies = st.sidebar.multiselect(
    "Typology", typology_options, default=typology_options,
    help="Only laundering transactions carry a typology label; unlabelled rows are excluded "
         "when this filter is narrowed.",
)
format_options = list(alerts["payment_format"].cat.categories)
payment_formats = st.sidebar.multiselect(
    "Payment format", format_options, default=format_options
)

ts_min, ts_max = alerts["timestamp"].min(), alerts["timestamp"].max()
date_range = st.sidebar.date_input(
    "Date range", value=(ts_min.date(), ts_max.date()),
    min_value=ts_min.date(), max_value=ts_max.date(),
)
min_score = st.sidebar.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
search = st.sidebar.text_input(
    "Account or bank", placeholder="e.g. 8000EBD30",
    help="Matches either account key (substring) or an exact bank ID.",
)

# `date_input` returns a 1-tuple mid-edit, while the user is picking the second date;
# treating that as a range would silently show a single day's rows without saying so.
selected_range = None
if isinstance(date_range, tuple) and len(date_range) == 2:
    selected_range = (
        pd.Timestamp(date_range[0]),
        pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
    )

filtered = artifact.apply_filters(
    alerts,
    queue_only=queue_only,
    include_tail=include_tail,
    typologies=typologies if len(typologies) < len(typology_options) else None,
    payment_formats=payment_formats if len(payment_formats) < len(format_options) else None,
    date_range=selected_range,
    min_score=min_score,
    search=search,
)
stats = artifact.queue_stats(filtered, metrics)

# --- Header -----------------------------------------------------------------------------

st.title("AML Alert-Triage Dashboard")
st.markdown(
    f"**{metrics['headline']['sentence']}** "
    f"&nbsp;·&nbsp; {metrics['headline']['baseline_alerts']:,} alerts → "
    f"{metrics['headline']['model_alerts']:,}, at the same recall of known laundering.",
    unsafe_allow_html=True,
)

queue_tab, detail_tab, performance_tab, about_tab = st.tabs(
    ["Alert Queue", "Alert Detail", "Model Performance", "About"]
)

# --- Alert Queue ------------------------------------------------------------------------

with queue_tab:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alerts shown", f"{stats['n_shown']:,}")
    c2.metric("Known laundering in view", f"{stats['n_positives']:,}")
    c3.metric(
        "Precision in view",
        f"{stats['precision']:.1%}" if stats["n_shown"] else "—",
    )
    c4.metric("Full queue size", f"{stats['queue_size']:,}")

    if not stats["population_exact"] and stats["n_shown"]:
        st.caption(
            "⚠️ This view includes transactions below the alert cut-off, which are a "
            "**stratified sample** — ranks and scores are exact, but counts in that region "
            "are sample counts, not population counts."
        )

    # `st.slider` raises when min_value == max_value, so a filter narrowing the view to
    # <=10 rows would crash this tab outright — which any emptied filter does. Handled with
    # nesting rather than `st.stop()`: stop() halts the whole script run, which would blank
    # the Model Performance and About tabs too, even though neither depends on this filter.
    if stats["n_shown"] == 0:
        st.info("No alerts match the current filters. Widen them in the sidebar.")
    else:
        if stats["n_shown"] <= 10:
            top_k = stats["n_shown"]
            st.caption(f"Showing all {top_k} matching alerts.")
        else:
            top_k = st.slider(
                "Show top N by rank", min_value=10,
                max_value=min(2000, stats["n_shown"]),
                value=min(100, stats["n_shown"]), step=10,
            )

        view = filtered.head(top_k)
        table = view[
            [
                "rank", "model_score", "timestamp", "amount_paid_usd", "payment_format",
                "from_account_key", "to_account_key", "pattern_type", "is_laundering",
                "is_tail_population", "reason_code_behavioural",
            ]
        ]

        selection = st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "model_score": st.column_config.ProgressColumn(
                    "Score", format="%.3f", min_value=0.0, max_value=1.0
                ),
                "timestamp": st.column_config.DatetimeColumn(
                    "Timestamp", format="YYYY-MM-DD HH:mm"
                ),
                "amount_paid_usd": st.column_config.NumberColumn(
                    "Amount (USD)", format="$%.0f"
                ),
                "payment_format": "Format",
                "from_account_key": "From",
                "to_account_key": "To",
                "pattern_type": "Typology",
                "is_laundering": st.column_config.CheckboxColumn("Laundering", width="small"),
                "is_tail_population": st.column_config.CheckboxColumn("Tail", width="small"),
                "reason_code_behavioural": st.column_config.TextColumn(
                    "Why flagged", width="large"
                ),
            },
        )

        if selection["selection"]["rows"]:
            st.session_state["selected_rank"] = int(
                view.iloc[selection["selection"]["rows"][0]]["rank"]
            )
            st.caption("Selected — open the **Alert Detail** tab.")
        else:
            st.caption("Select a row to inspect it in the **Alert Detail** tab.")

        st.download_button(
            "Download this view (CSV)",
            view[artifact.display_columns(alerts)].to_csv(index=False).encode("utf-8"),
            file_name="aml_alert_queue.csv",
            mime="text/csv",
        )

# --- Alert Detail -------------------------------------------------------------------------

with detail_tab:
    if filtered.empty:
        st.info("No alerts match the current filters.")
    else:
        ranks = filtered["rank"].tolist()
        default_rank = st.session_state.get("selected_rank", ranks[0])
        if default_rank not in ranks:
            default_rank = ranks[0]

        chosen = st.selectbox(
            "Alert (by rank)", ranks, index=ranks.index(default_rank),
            format_func=lambda r: f"#{r}",
        )
        row = filtered[filtered["rank"] == chosen].iloc[0]

        left, right = st.columns([2, 3])
        with left:
            st.subheader(f"Alert #{int(row['rank'])}")
            st.metric("Model score", f"{row['model_score']:.4f}")
            st.write(
                pd.DataFrame(
                    {
                        "Field": [
                            "Timestamp", "Amount (USD)", "Amount (original)", "Format",
                            "From", "To", "Typology", "Known laundering", "Anomalous tail",
                        ],
                        "Value": [
                            str(row["timestamp"]),
                            f"${row['amount_paid_usd']:,.2f}",
                            f"{row['amount_paid']:,.2f} {row['payment_currency']}",
                            str(row["payment_format"]),
                            f"{row['from_account_key']} (bank {row['from_bank']})",
                            f"{row['to_account_key']} (bank {row['to_bank']})",
                            str(row["pattern_type"]) if pd.notna(row["pattern_type"]) else "—",
                            "yes" if row["is_laundering"] else "no",
                            "yes" if row["is_tail_population"] else "no",
                        ],
                    }
                ).set_index("Field")
            )

        with right:
            st.subheader("Why this was flagged")
            st.success(row["reason_code_behavioural"])
            with st.expander("Reason code faithful to raw SHAP"):
                st.write(row["reason_code_faithful"])
                st.caption(
                    "The faithful variant leads with the payment-format feature on 99.86% of "
                    "the queue, so it carries almost no triage information despite being "
                    "exactly what the model did. The behavioural variant above excludes "
                    "payment format from the *sentence only* — never from the model."
                )

        st.subheader("Contribution breakdown")
        wf = artifact.waterfall_data(row, names, base_value, top_n=10)
        implied = artifact.reconstructed_score(row, names, base_value)

        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["absolute"] + ["relative"] * len(wf) + ["total"],
                x=["baseline"] + wf["feature"].tolist() + ["final score (log-odds)"],
                y=[base_value] + wf["contribution"].tolist() + [None],
                text=[f"{base_value:+.2f}"]
                + [f"{v:+.2f}" for v in wf["contribution"]]
                + [""],
                textposition="outside",
                connector={"line": {"color": MUTED, "width": 1}},
                increasing={"marker": {"color": POSITIVE_COLOUR}},
                decreasing={"marker": {"color": QUEUE_COLOUR}},
                totals={"marker": {"color": "#37474f"}},
            )
        )
        fig.update_layout(
            height=430, margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis_title="log-odds contribution", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Bars are exact SHAP contributions in log-odds space and sum to the model's "
            f"output: baseline {base_value:.4f} + contributions → score "
            f"**{implied:.4f}** (shipped score {row['model_score']:.4f}). Red pushed this "
            f"transaction toward *laundering*, blue pushed it away."
        )

        with st.expander("All 54 features, with values and contributions"):
            st.dataframe(
                pd.DataFrame(
                    {
                        "feature": [artifact.prettify_feature(n) for n in names],
                        "value": [float(row[f"feat_{n}"]) for n in names],
                        "shap_contribution": [float(row[f"shap_{n}"]) for n in names],
                    }
                ).sort_values("shap_contribution", key=abs, ascending=False),
                hide_index=True, use_container_width=True,
            )

# --- Model Performance ---------------------------------------------------------------------

with performance_tab:
    st.subheader("The headline result")
    h, hx = metrics["headline"], metrics["headline_excluding_tail"]
    a, b, c, d = st.columns(4)
    a.metric("Rules baseline alerts", f"{h['baseline_alerts']:,}")
    b.metric("Model alerts", f"{h['model_alerts']:,}", delta=f"-{h['fp_reduction']:.1%}")
    c.metric("At matched recall", f"{h['matched_recall']:.1%}")
    d.metric("Excluding anomalous tail", f"{hx['fp_reduction']:.1%}")
    st.caption(
        f"Both computed on all {metrics['test_split']['n_rows']:,} held-out transactions, "
        f"not on this demo's sample. {hx['note']}"
    )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Precision–recall curve")
        pr = artifact.pr_curve_frame(metrics)
        fig = go.Figure(
            go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                       line={"color": QUEUE_COLOUR, "width": 2})
        )
        fig.add_hline(
            y=metrics["test_split"]["prevalence"], line_dash="dash", line_color=MUTED,
            annotation_text=f"prevalence {metrics['test_split']['prevalence']:.3%}",
            annotation_position="top right",
        )
        fig.update_layout(
            height=360, xaxis_title="Recall", yaxis_title="Precision",
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"PR-AUC **{metrics['auc']['pr_auc']:.4f}**, ROC-AUC "
            f"{metrics['auc']['roc_auc']:.4f}. The dashed line is the base rate a random "
            f"ranker would achieve — accuracy is meaningless at this prevalence."
        )

    with right:
        st.subheader("Recall by laundering typology")
        typ = artifact.typology_frame(metrics)
        fig = go.Figure(
            go.Bar(
                x=typ["recall"], y=typ["typology"], orientation="h",
                marker_color=QUEUE_COLOUR,
                text=[f"{r:.1%} ({c}/{n})" for r, c, n in
                      zip(typ["recall"], typ["n_caught"], typ["n_transactions"], strict=True)],
                textposition="auto",
            )
        )
        fig.update_layout(
            height=360, xaxis_title="Recall at the equal-recall operating point",
            xaxis_tickformat=".0%", margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "No typology collapses — the headline recall isn't coming from one easy pattern."
        )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Score drift over time (PSI)")
        psi = artifact.psi_frame(metrics)
        scored = psi[psi["psi"].notna()]
        thin = psi[psi["psi"].isna()]
        thresholds = metrics["psi_thresholds"]

        fig = go.Figure()
        fig.add_hrect(y0=0, y1=thresholds["moderate"], fillcolor="#2e7d32", opacity=0.07,
                      line_width=0)
        fig.add_hrect(y0=thresholds["moderate"], y1=thresholds["significant"],
                      fillcolor="#f9a825", opacity=0.10, line_width=0)
        fig.add_hrect(y0=thresholds["significant"], y1=0.45, fillcolor="#c62828",
                      opacity=0.08, line_width=0)
        fig.add_trace(go.Scatter(x=scored["period"], y=scored["psi"], mode="lines+markers",
                                 line={"color": QUEUE_COLOUR, "width": 2}, name="score PSI"))
        fig.add_trace(go.Scatter(x=thin["period"], y=[0] * len(thin), mode="markers",
                                 marker={"symbol": "x", "color": MUTED, "size": 9},
                                 name="too thin to score"))
        fig.update_layout(
            height=340, yaxis_title="PSI vs. training distribution", yaxis_range=[0, 0.45],
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            legend={"orientation": "h", "y": 1.15},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The 2022-09-09 step is **not a data incident**: it is the 7-day graph feature "
            "lookback evicting the dataset's largest day for the first time. A drift monitor "
            "watches the features, not the world. The ✗ days are the anomalous tail — under "
            "the minimum slice size, so correctly reported as unscoreable rather than "
            "guessed at."
        )

    with right:
        st.subheader("Global feature importance")
        imp = artifact.importance_frame(metrics).head(12)
        fig = go.Figure(
            go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h",
                   marker_color=QUEUE_COLOUR)
        )
        fig.update_layout(
            height=340, xaxis_title="mean |SHAP|",
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "`payment format: ACH` dominates because 86.6% of labelled laundering in this "
            "synthetic dataset uses ACH. That is a property of the data, disclosed rather "
            "than smoothed over — an ablation without it is in `reports/challenges.md`."
        )

    st.divider()
    st.subheader("Precision at k")
    pk = metrics["precision_at_k"]
    cols = st.columns(len(pk))
    for col, (k, value) in zip(cols, sorted(pk.items(), key=lambda kv: int(kv[0])), strict=True):
        col.metric(f"precision@{int(k):,}", f"{value:.1%}")
    st.caption(
        "What an analyst actually experiences: they can only review a fixed number of "
        "alerts a day, so precision within the top k is the metric that matters."
    )

# --- About -----------------------------------------------------------------------------------

with about_tab:
    st.subheader("What this is")
    st.markdown(
        """
This is an **alert-triage** system for anti-money-laundering transaction monitoring — not
a classifier that "detects money laundering". A rules-based engine of the kind banks
actually run raises an enormous number of alerts, the overwhelming majority of which are
false positives that a compliance analyst has to clear by hand. This model re-ranks and
filters that queue.

The result it exists to defend:

> **At equal recall of known laundering, the model raises 96.6% fewer alerts than the
> rules baseline** — 10,011 instead of 297,564, on a held-out period of 1,015,669
> transactions.
"""
    )

    st.subheader("How it works")
    st.markdown(
        f"""
- **Rules baseline** — large-amount, structuring, and rapid pass-through rules, with
  amounts normalised to USD across 15 currencies. This is what the model must beat.
- **Features** — account/window velocity and volume over 1/7/30-day windows, structuring
  scores, pass-through ratios, plus directed-graph features (degree, fan-in/fan-out,
  short-cycle membership) computed on a rolling 7-day account graph. All strictly
  past-only per transaction; splits are time-ordered, never random k-fold.
- **Model** — {metrics['model']['type']}, `scale_pos_weight`
  {metrics['model']['scale_pos_weight']:,.0f}, {metrics['model']['n_features']} features,
  stopped at iteration {metrics['model']['best_iteration']}.
- **Explanations** — exact SHAP contributions per alert, rendered into a plain-English
  reason code against each row's own raw values.
- **Monitoring** — PSI/CSI against the training distribution as a fixed reference.
"""
    )

    st.subheader("Honest caveats")
    st.markdown(
        f"""
- **The data is synthetic.** IBM's public AML benchmark ({metrics['dataset']['total_rows']:,}
  transactions), not real bank data. Treat these numbers as a demonstration of approach.
- **This app shows a sample; the numbers come from the full split.** Every metric on the
  Model Performance tab is computed over all
  {metrics['test_split']['n_rows']:,} held-out transactions. The browsable table is
  {len(alerts):,} rows: every known positive, the entire
  {metrics['headline']['model_alerts']:,}-alert queue, and a stratified sample of the rest.
  **Ranks and scores are true values** computed before sampling — but counts below the
  alert cut-off are sample counts.
- **The anomalous tail is over-represented here** —
  {metrics['demo_sample']['tail_share_in_sample']:.1%}
  of this sample versus {metrics['demo_sample']['tail_share_in_test_split']:.3%} of the real
  held-out period, because every positive is kept. It's flagged per row and filterable, and
  the headline holds without it ({hx['fp_reduction']:.1%}).
- **Nothing is computed live.** All scores, reason codes and SHAP values were precomputed
  offline; this app only reads and filters them.
- **Scores are rankings, not calibrated probabilities** — a direct consequence of
  `scale_pos_weight`. Every metric here depends only on ranking.
"""
    )

    st.subheader("Source")
    st.markdown(
        "[github.com/Kashish1430/aml-transaction-monitoring]"
        "(https://github.com/Kashish1430/aml-transaction-monitoring) — full write-ups in "
        "`reports/results.md` (every number, with the command that regenerates it) and "
        "`reports/challenges.md` (what went wrong and why)."
    )
    st.caption(f"Artifact built {metrics['generated_at_utc']}.")
