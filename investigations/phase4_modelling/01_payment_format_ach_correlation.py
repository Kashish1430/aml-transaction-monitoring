"""Reproduces the Phase 4 finding in reports/challenges.md: `payment_format_ACH`
dominates feature importance because 86.6% of all laundering transactions in
HI-Small use ACH format, and ACH's laundering rate is ~7.3x the dataset's overall
prevalence -- while Wire and Reinvestment have zero laundering transactions anywhere
in the data.

Run from the repo root: venve/python.exe investigations/phase4_modelling/01_payment_format_ach_correlation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data_loader import load_config


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    modelling_table_path = Path(__file__).resolve().parents[2] / config["paths"]["modelling_table"]

    df = pd.read_parquet(modelling_table_path, columns=["payment_format", "is_laundering"])

    print("Laundering rate by payment_format:")
    rate = df.groupby("payment_format")["is_laundering"].agg(["size", "sum", "mean"])
    rate = rate.sort_values("mean", ascending=False)
    print(rate.to_string())

    print("\nShare of ALL laundering transactions that use each format:")
    laundering = df[df["is_laundering"] == 1]
    print((laundering["payment_format"].value_counts(normalize=True) * 100).round(2).to_string())

    overall_prevalence = df["is_laundering"].mean()
    ach_rate = rate.loc["ACH", "mean"]
    print(f"\nACH laundering rate ({ach_rate:.4%}) is {ach_rate / overall_prevalence:.1f}x the overall prevalence ({overall_prevalence:.4%})")


if __name__ == "__main__":
    main()
