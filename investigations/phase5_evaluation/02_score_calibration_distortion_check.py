"""Reproduces the Phase 5 finding in reports/challenges.md: the model's raw scores are
not calibrated probabilities -- mean predicted score on the test split is ~41x the
actual test prevalence, the expected consequence of `scale_pos_weight`.

Run from the repo root: venve/python.exe investigations/phase5_evaluation/02_score_calibration_distortion_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_loader import load_config
from src.evaluate import calibration_table
from src.model import load_xgboost_model, prepare_feature_matrix, time_ordered_split


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    modelling_table_path = Path(__file__).resolve().parents[2] / config["paths"]["modelling_table"]
    models_dir = Path(__file__).resolve().parents[2] / config["paths"]["models_dir"]

    df = pd.read_parquet(modelling_table_path)
    _, _, test_df = time_ordered_split(df, config["split"]["train_frac"], config["split"]["val_frac"])
    del df

    x_test, y_test = prepare_feature_matrix(test_df)
    model = load_xgboost_model(models_dir)
    scores = model.predict_proba(x_test)[:, 1]

    prevalence = y_test.mean()
    mean_score = scores.mean()
    print(f"Actual test prevalence: {prevalence:.4%}")
    print(f"Mean predicted score:   {mean_score:.4%}")
    print(f"Median predicted score: {pd.Series(scores).median():.6%}")
    print(f"Mean predicted score is {mean_score / prevalence:.1f}x the true prevalence")

    print("\nCalibration table (10 quantile bins) -- mean_predicted_score should equal")
    print("fraction_positive under perfect calibration; it does not, at any bin:")
    print(calibration_table(y_test.to_numpy(), scores).to_string(index=False))


if __name__ == "__main__":
    main()
