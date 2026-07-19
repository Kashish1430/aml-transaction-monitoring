# AML Transaction Alert-Triage System

> **Status: scaffolding stage.** Full framing, headline results, live demo link, and usage instructions land in Phase 11 of `PLAN.md`. This placeholder will be replaced then.

An anti-money-laundering (AML) transaction-monitoring system that triages the alerts a rules-based engine would generate — cutting false positives while preserving recall of known laundering typologies, with per-alert SHAP explanations suitable for a compliance analyst.

See `AML_Transaction_Monitoring_Project_Brief.md` for the full technical brief and `PLAN.md` for the build/deployment roadmap.

## Repository structure

```
├── app/            # Streamlit demo (Phase 9-10)
├── data/           # raw/processed data, gitignored
├── notebooks/      # EDA, feature engineering, modelling, evaluation walkthroughs
├── src/            # pipeline source of truth
├── scripts/        # one-off/build scripts (e.g. demo artifact builder)
├── tests/          # pytest suite, run in CI
├── reports/        # figures + results.md
└── config.yaml     # paths, seeds, thresholds
```

## Local setup

```bash
pip install -r requirements.txt
pytest -v
```
