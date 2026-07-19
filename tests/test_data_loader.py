"""Schema validation and parsing checks for src.data_loader (PLAN.md Phase 1)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_accounts, load_patterns, load_transactions

VALID_TXN_HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
)
VALID_TXN_ROW = (
    "2022/09/01 00:20,10,8000EBD30,10,8000EBD30,100.00,US Dollar,100.00,US Dollar,Cheque,0\n"
)


def test_load_transactions_fails_loudly_on_missing_columns(tmp_path: Path):
    bad_csv = tmp_path / "bad_trans.csv"
    bad_csv.write_text("Timestamp,From Bank,Amount Paid\n2022/09/01 00:20,10,100.00\n")

    with pytest.raises(ValueError, match="missing expected columns"):
        load_transactions(bad_csv)


def test_load_transactions_parses_valid_schema(tmp_path: Path):
    good_csv = tmp_path / "good_trans.csv"
    good_csv.write_text(VALID_TXN_HEADER + VALID_TXN_ROW)

    df = load_transactions(good_csv)

    assert len(df) == 1
    assert df.loc[0, "timestamp"] == pd.Timestamp("2022-09-01 00:20")
    assert df.loc[0, "from_account_key"] == "10_8000EBD30"
    assert df.loc[0, "to_account_key"] == "10_8000EBD30"


def test_load_accounts_fails_loudly_on_missing_columns(tmp_path: Path):
    bad_csv = tmp_path / "bad_accounts.csv"
    bad_csv.write_text("Bank Name,Bank ID\nSome Bank,10\n")

    with pytest.raises(ValueError, match="missing expected columns"):
        load_accounts(bad_csv)


def test_load_accounts_builds_account_key(tmp_path: Path):
    good_csv = tmp_path / "good_accounts.csv"
    good_csv.write_text(
        "Bank Name,Bank ID,Account Number,Entity ID,Entity Name\n"
        "Some Bank,10,8000EBD30,ENT1,Some Entity\n"
    )

    df = load_accounts(good_csv)

    assert df.loc[0, "account_key"] == "10_8000EBD30"


def test_load_patterns_parses_typology_blocks(tmp_path: Path):
    patterns_txt = tmp_path / "patterns.txt"
    patterns_txt.write_text(
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out\n"
        "2022/09/01 00:06,21174,800737690,12,80011F990,2848.96,Euro,2848.96,Euro,ACH,1\n"
        "END LAUNDERING ATTEMPT - FAN-OUT\n"
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  4-node cycle\n"
        "2022/09/02 00:06,1,800AAAAAA,2,800BBBBBB,500.00,US Dollar,500.00,US Dollar,ACH,1\n"
        "END LAUNDERING ATTEMPT - CYCLE\n"
    )

    df = load_patterns(patterns_txt)

    assert len(df) == 2
    assert set(df["pattern_type"]) == {"FAN-OUT", "CYCLE"}
    assert df["is_laundering"].eq(1).all()
