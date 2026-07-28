"""End-to-end checks that the Streamlit app actually runs (PLAN.md Phase 9).

`streamlit.testing.v1.AppTest` executes `app/streamlit_app.py` headless, in-process, the
same way the server would, and surfaces any exception the script raised. That makes the
PLAN.md Phase 9 check ("click through all four tabs, no exceptions") executable rather
than a thing someone remembers to do by hand — which matters because Phase 10 deploys this
file to a public URL, and a filter combination that throws would be found by a visitor.

Each test re-runs the app from scratch (~1-3s): AppTest carries widget state forward
within a single instance, so sharing one across tests would leak one test's filters into
the next.

Note on `set_value` for widgets with a `format_func`: `AppTest` reports `.options` already
formatted (`"#1"`), but `set_value` expects the *raw* option and applies the format itself.
Passing `sb.options[i]` therefore double-formats and raises inside AppTest, not the app.
Pass the raw value.
"""

from __future__ import annotations

import pytest

from app import artifact

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

pytestmark = pytest.mark.skipif(
    not (artifact.ARTIFACT_PATH.exists() and artifact.METRICS_PATH.exists()),
    reason="demo artifact not built — run python -m scripts.build_demo_artifact",
)

APP = "app/streamlit_app.py"


def run_app():
    return AppTest.from_file(APP, default_timeout=120).run()


def labelled(app, label):
    return next(m for m in app.metric if m.label == label)


def test_app_runs_clean_with_all_four_tabs():
    app = run_app()

    assert not app.exception
    assert len(app.tabs) == 4
    assert app.title[0].value == "AML Alert-Triage Dashboard"


def test_default_view_is_the_alert_queue_and_matches_the_headline():
    app = run_app()

    # Defaults land on the equal-recall queue, so the first thing a visitor sees is the
    # population the headline describes — not an arbitrary slice of the sample.
    metrics = artifact.load_metrics()
    assert labelled(app, "Alerts shown").value == f"{metrics['headline']['model_alerts']:,}"
    assert labelled(app, "Rules baseline alerts").value == (
        f"{metrics['headline']['baseline_alerts']:,}"
    )


def test_emptying_a_filter_shows_nothing_without_crashing_the_other_tabs():
    """The regression this exists for: `st.slider` raises when min_value == max_value, so
    a view narrowed to <=10 rows used to crash the queue tab. Fixed with nesting rather
    than `st.stop()`, which would have blanked the Performance and About tabs too.
    """
    app = run_app()
    app.multiselect[0].set_value([]).run()

    assert not app.exception
    assert labelled(app, "Alerts shown").value == "0"
    assert any("No alerts match" in i.value for i in app.info)
    # The tabs that don't depend on the filter must still render.
    assert labelled(app, "Rules baseline alerts").value


def test_a_single_row_view_renders():
    # Exercises the "<=10 rows, no slider" branch on a real filter rather than a mock.
    app = run_app()
    app.slider[1].set_value(0.995).run()

    assert not app.exception
    assert int(labelled(app, "Alerts shown").value.replace(",", "")) <= 10


def test_leaving_the_queue_labels_counts_as_sample_counts():
    """Below the alert cut-off the artifact is a stratified sample, so the app must say
    the counts there aren't population counts (PLAN.md Phase 8).
    """
    app = run_app()
    app.toggle[0].set_value(False).run()

    assert not app.exception
    assert labelled(app, "Alerts shown").value == "30,000"
    assert any("sample counts" in c.value for c in app.caption)


def test_tail_toggle_removes_the_flagged_rows():
    app = run_app()
    before = int(labelled(app, "Alerts shown").value.replace(",", ""))
    app.toggle[1].set_value(False).run()

    assert not app.exception
    assert int(labelled(app, "Alerts shown").value.replace(",", "")) < before


def test_alert_detail_renders_a_waterfall_for_a_selected_alert():
    app = run_app()
    app.selectbox[0].set_value(7).run()

    assert not app.exception
    assert any(s.value == "Alert #7" for s in app.subheader)
    # The waterfall's endpoint must be stated as agreeing with the ranked score.
    assert any("shipped score" in c.value for c in app.caption)


def test_about_tab_discloses_the_sampling_and_the_synthetic_data():
    app = run_app()
    markdown = " ".join(m.value for m in app.markdown)

    assert "synthetic" in markdown.lower()
    assert "sample" in markdown.lower()
    assert "Ranks and scores are true values" in markdown
