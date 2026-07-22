"""Train/predict wrappers: XGBoost with explicit imbalance handling and a strictly
time-ordered train/val/test split, plus an optional Isolation Forest unsupervised layer.

Implemented in PLAN.md Phase 4. Every design choice below is deliberate and documented
inline — see `reports/challenges.md`'s Phase 4 entry for the full reasoning where a
choice involved a real trade-off, not just a default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest

# Columns that exist in the modelling table but are not model inputs: identifiers,
# the timestamp used only for splitting/ordering, the label, and payment_format (which
# gets one-hot encoded separately, not passed through as a raw string).
NON_FEATURE_COLUMNS = {
    "timestamp",
    "from_account_key",
    "to_account_key",
    "is_laundering",
    "payment_format",
}

# Confirmed empirically against the real data (not assumed) — see the Phase 4 entry in
# reports/challenges.md. Hardcoded and sorted, not derived from whatever categories
# happen to appear in a given split, so one-hot columns are identical in count and
# order across train/val/test/production regardless of which formats appear in each —
# a split that happens to contain zero Bitcoin transactions must not silently produce a
# differently-shaped feature matrix than one that does.
PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]


def time_ordered_split(
    df: pd.DataFrame, train_frac: float, val_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test by transaction position after sorting by timestamp —
    train on the earliest `train_frac` of transactions, validate on the next `val_frac`,
    test on the remainder. Never a random shuffle or k-fold (CLAUDE.md's no-leakage
    constraint): a row-position split on time-sorted data is a time-quantile split, and
    is deliberately used here instead of a fixed calendar-date cutoff, because
    HI-Small's daily volume is extremely uneven (day 1 alone is ~22% of all
    transactions — see CLAUDE.md's Dataset section) and a date-based cutoff would make
    the split fractions unpredictable. Raises if the resulting splits aren't strictly
    time-ordered — this should be structurally impossible given the sort, but it's the
    Phase 4 leakage guarantee this function exists to provide, so it's asserted, not
    assumed.

    Skips the actual sort when `df["timestamp"]` is already monotonic (true for
    `data/processed/features.parquet` — built from `load_transactions`, which sorts by
    timestamp, and neither `assemble_feature_table` nor `build_daily_graph_features`
    reorders rows). `DataFrame.sort_values` on 5M+ rows allocates a full copy; skipping
    it when it would be a no-op anyway avoids briefly doubling memory on top of the
    ~2.85GB the raw table already occupies (see this module's docstring and
    reports/challenges.md's Phase 4 entry).
    """
    if df["timestamp"].is_monotonic_increasing:
        df_sorted = df.reset_index(drop=True)
    else:
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df_sorted.iloc[:train_end]
    val = df_sorted.iloc[train_end:val_end]
    test = df_sorted.iloc[val_end:]

    if len(train) and len(val):
        assert train["timestamp"].max() <= val["timestamp"].min()
    if len(val) and len(test):
        assert val["timestamp"].max() <= test["timestamp"].min()

    return train, val, test


def prepare_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the numeric feature matrix X and label vector y from a modelling-table
    slice (a `time_ordered_split` output).

    Two deliberate choices here, both driven by real constraints hit while building
    this phase (see reports/challenges.md):
    - `payment_format` is one-hot encoded against the fixed `PAYMENT_FORMATS` list
      (via `pd.Categorical(..., categories=PAYMENT_FORMATS)`), not `pd.get_dummies`'s
      default of inferring categories from whatever's present in `df` — this guarantees
      identical columns across train/val/test even though the label imbalance means
      some splits could plausibly miss a rare payment format entirely.
    - Engineered features are downcast to float32. The full modelling table is ~2.85GB
      in memory as float64; float32 halves that with no meaningful precision loss for
      count/ratio/degree-style features, and memory headroom on the development machine
      was tight enough (single-digit GB free) that this wasn't optional polish.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    x_numeric = df[feature_cols].astype("float32")

    payment_dummies = pd.get_dummies(
        pd.Categorical(df["payment_format"], categories=PAYMENT_FORMATS)
    ).astype("float32")
    payment_dummies.columns = [f"payment_format_{c}" for c in payment_dummies.columns]
    payment_dummies.index = df.index

    x = pd.concat([x_numeric, payment_dummies], axis=1)
    y = df["is_laundering"].astype("int8")
    return x, y


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """XGBoost's `scale_pos_weight` = negatives/positives, computed from the TRAINING
    split only (never val/test — using their class balance would leak split-specific
    information into training). Preferred over naive oversampling per the project
    brief; if resampling were used instead it would need to be train-fold-only too, for
    the same reason.
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos == 0:
        raise ValueError("compute_scale_pos_weight: training split has zero positives")
    return n_neg / n_pos


def train_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
    seed: int,
) -> xgb.XGBClassifier:
    """Train the primary supervised model.

    Hyperparameters are reasonable, documented defaults, not the product of a grid/
    random search — that's out of scope for this phase (disclosed here, not silently
    skipped). The one piece of real tuning done is early stopping on the validation
    set (never the test set, which stays untouched until Phase 5's evaluation):
    - `eval_metric="aucpr"`: PR-AUC, not accuracy or plain AUC-ROC — the right curve
      under ~0.1% prevalence per the project brief, so early stopping optimizes the
      metric that actually matters here.
    - `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`: mild regularization
      (row/column subsampling, moderate tree depth) against overfitting on a training
      split with only ~2,300 positive examples.
    - `tree_method="hist"`: histogram-based splits — the standard choice for datasets
      at this scale (millions of rows), both faster and more memory-efficient than the
      exact method.
    - `scale_pos_weight`: passed in, computed by `compute_scale_pos_weight` from the
      training split only.
    """
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=seed,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def train_isolation_forest(x_train: pd.DataFrame, seed: int) -> IsolationForest:
    """Unsupervised anomaly layer (brief's Step 4.3, "optional but realistic"): scores
    transactions with no reliance on labels at all, reflecting that launderers invent
    patterns the labelled typologies don't cover. `contamination="auto"` (sklearn's
    default heuristic) is used deliberately instead of setting it to the training
    split's true positive rate — plugging the real label rate into an "unsupervised"
    model's threshold would quietly smuggle label information back in, undermining the
    point of having this layer at all. Compared against the supervised model's
    agreement/disagreement in the accompanying notebook, not blended into one score.
    """
    model = IsolationForest(contamination="auto", random_state=seed, n_jobs=-1)
    model.fit(x_train)
    return model


def save_model(
    xgb_model: xgb.XGBClassifier,
    isolation_forest: IsolationForest | None,
    feature_names: list[str],
    metadata: dict,
    models_dir: str | Path,
) -> None:
    """Serialize both models plus a metadata sidecar to `models_dir` (gitignored — a
    fresh clone regenerates this by rerunning notebooks/03_modelling.ipynb, per
    CLAUDE.md's "every reported number must be regenerable from code" rule extended to
    artifacts).

    XGBoost's native `save_model` (not pickle) is used for the supervised model —
    it's the format-stable, XGBoost-version-portable choice. `IsolationForest` has no
    native serializer, so `joblib` is used for it (imported lazily; only needed here).
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    xgb_model.save_model(models_dir / "xgboost_model.json")

    if isolation_forest is not None:
        import joblib

        joblib.dump(isolation_forest, models_dir / "isolation_forest.joblib")

    full_metadata = {**metadata, "feature_names": feature_names}
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(full_metadata, f, indent=2, default=str)


def load_xgboost_model(models_dir: str | Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(Path(models_dir) / "xgboost_model.json")
    return model


def load_metadata(models_dir: str | Path) -> dict:
    with open(Path(models_dir) / "model_metadata.json", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from src.data_loader import load_config

    config = load_config()
    seed = config["seed"]
    split_cfg = config["split"]

    print(f"Loading modelling table from {config['paths']['modelling_table']} ...")
    df = pd.read_parquet(config["paths"]["modelling_table"])

    train_df, val_df, test_df = time_ordered_split(
        df, split_cfg["train_frac"], split_cfg["val_frac"]
    )
    del df
    print(
        f"Split: train={len(train_df):,} ({train_df['is_laundering'].sum()} positives), "
        f"val={len(val_df):,} ({val_df['is_laundering'].sum()} positives), "
        f"test={len(test_df):,} ({test_df['is_laundering'].sum()} positives)"
    )

    x_train, y_train = prepare_feature_matrix(train_df)
    x_val, y_val = prepare_feature_matrix(val_df)
    x_test, y_test = prepare_feature_matrix(test_df)

    spw = compute_scale_pos_weight(y_train)
    print(f"scale_pos_weight (train-only): {spw:.2f}")

    model = train_xgboost(x_train, y_train, x_val, y_val, spw, seed)
    print(f"Best iteration: {model.best_iteration}, best val PR-AUC: {model.best_score:.4f}")

    iso_forest = train_isolation_forest(x_train, seed)

    test_scores = model.predict_proba(x_test)[:, 1]
    from sklearn.metrics import average_precision_score, roc_auc_score

    print(f"Test PR-AUC: {average_precision_score(y_test, test_scores):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")

    metadata = {
        "seed": seed,
        "scale_pos_weight": spw,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_positives": int(y_train.sum()),
        "val_positives": int(y_val.sum()),
        "test_positives": int(y_test.sum()),
        "best_iteration": int(model.best_iteration),
        "best_val_aucpr": float(model.best_score),
        "test_aucpr": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
    }
    save_model(model, iso_forest, list(x_train.columns), metadata, config["paths"]["models_dir"])
    print(f"Saved model + metadata to {config['paths']['models_dir']}")
