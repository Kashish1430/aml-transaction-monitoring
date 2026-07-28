"""Reproduces the Phase 5 finding in reports/challenges.md: `join_pattern_types`
(src/data_loader.py) needed verifying against the previously-documented 62%/
3,209-of-5,177 typology match rate before it could be trusted for recall-per-typology
-- and confirms there are zero duplicate/ambiguous join keys on the real data.

Run from the repo root: venve/python.exe investigations/phase5_evaluation/01_pattern_type_join_verification.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loader import PATTERN_COLUMNS, join_pattern_types, load_config, load_patterns, load_transactions


def main():
    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    raw_dir = Path(__file__).resolve().parents[2] / config["paths"]["raw_dir"]

    txns = load_transactions(raw_dir / "HI-Small_Trans.csv")
    patterns = load_patterns(raw_dir / "HI-Small_Patterns.txt")

    print(f"Transactions: {len(txns):,}, laundering: {txns['is_laundering'].sum():,}")
    print(f"Pattern rows: {len(patterns):,}, unique typologies: {patterns['pattern_type'].nunique()}")

    join_cols = [c for c in PATTERN_COLUMNS if c != "pattern_type"]
    laundering_txns = txns[txns["is_laundering"] == 1]
    dup_in_txns = laundering_txns.duplicated(subset=join_cols, keep=False).sum()
    dup_in_patterns = patterns.duplicated(subset=join_cols, keep=False).sum()
    print(f"\nDuplicate join-keys among laundering transactions: {dup_in_txns}")
    print(f"Duplicate join-keys among pattern rows: {dup_in_patterns}")

    result = join_pattern_types(txns, patterns)
    n_matched = result.loc[result["is_laundering"] == 1, "pattern_type"].notna().sum()
    n_laundering = int(result["is_laundering"].sum())
    n_false_matches = result.loc[result["is_laundering"] == 0, "pattern_type"].notna().sum()

    print(f"\nMatched pattern_type for {n_matched:,} / {n_laundering:,} laundering transactions "
          f"({n_matched / n_laundering:.1%})")
    print(f"Legitimate transactions that accidentally matched a pattern: {n_false_matches} (must be 0)")
    print(f"\nExpected from CLAUDE.md's Dataset section: 3,209 / 5,177 (62%) -- {'MATCHES' if n_matched == 3209 else 'MISMATCH, investigate'}")

    print("\nTypology breakdown:")
    print(result.loc[result["is_laundering"] == 1, "pattern_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
