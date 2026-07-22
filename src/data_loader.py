"""Load HI-Small, validate schema, standardise account keys.

Column names and dtypes were confirmed against the actual downloaded files rather than
assumed (see PLAN.md Phase 1) — notably `Trans.csv` has two columns both named "Account"
(sender/receiver), which pandas auto-renames to "Account" and "Account.1" on load.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

TRANSACTION_RENAME = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

ACCOUNTS_RENAME = {
    "Bank Name": "bank_name",
    "Bank ID": "bank_id",
    "Account Number": "account_number",
    "Entity ID": "entity_id",
    "Entity Name": "entity_name",
}

PATTERN_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]


def load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_columns(df: pd.DataFrame, expected: set[str], source: str | Path) -> None:
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"{source} is missing expected columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Load the HI-Small transaction CSV: validate schema, parse types, build account keys.

    Account keys are `f"{bank_id}_{account_number}"`, matching load_accounts' key so the
    two tables join cleanly (see PLAN.md's "standardise account identifiers" requirement).
    """
    df = pd.read_csv(path)
    _validate_columns(df, set(TRANSACTION_RENAME), path)

    df = df.rename(columns=TRANSACTION_RENAME)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    df["from_account_key"] = df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
    df["to_account_key"] = df["to_bank"].astype(str) + "_" + df["to_account"].astype(str)

    return df.sort_values("timestamp").reset_index(drop=True)


def load_accounts(path: str | Path) -> pd.DataFrame:
    """Load HI-Small_accounts.csv (bank/account -> owning entity), validate schema."""
    df = pd.read_csv(path)
    _validate_columns(df, set(ACCOUNTS_RENAME), path)

    df = df.rename(columns=ACCOUNTS_RENAME)
    df["account_key"] = df["bank_id"].astype(str) + "_" + df["account_number"].astype(str)
    return df


def load_patterns(path: str | Path) -> pd.DataFrame:
    """Parse HI-Small_Patterns.txt into one row per labelled transaction, tagged with its
    laundering typology (fan-in, fan-out, cycle, ...).

    The file 's blockis not a CSV: its of
    "BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>: <note>" / transaction rows / "END LAUNDERING
    ATTEMPT - <TYPOLOGY>". There is no transaction ID column anywhere in this dataset, so
    joining these labels back onto load_transactions' output has to happen on the shared
    field values themselves (timestamp + banks/accounts + amounts + format) — see
    `join_pattern_types` below (Phase 5's recall-per-typology is the first thing that
    actually needs this join; this docstring previously pointed at Phase 3, which never
    implemented it — corrected here rather than left stale).
    """
    rows = []
    current_pattern = None

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                current_pattern = line.split("-", 1)[1].split(":", 1)[0].strip()
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                current_pattern = None
                continue
            if current_pattern is None:
                continue
            fields = line.split(",")
            if len(fields) != len(PATTERN_COLUMNS):
                continue
            rows.append([*fields, current_pattern])

    patterns_df = pd.DataFrame(rows, columns=[*PATTERN_COLUMNS, "pattern_type"])
    patterns_df["timestamp"] = pd.to_datetime(patterns_df["timestamp"], format="%Y/%m/%d %H:%M")
    for col in ["amount_received", "amount_paid", "is_laundering"]:
        patterns_df[col] = pd.to_numeric(patterns_df[col])

    return patterns_df


def join_pattern_types(txns: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Attach each laundering transaction's typology label (fan-in, cycle, ...) from
    `load_patterns`'s output onto `load_transactions`'s output, by matching on every
    field the two tables share (there is no transaction ID anywhere in this dataset).

    Returns `txns` with one new `pattern_type` column (NaN for transactions that don't
    match a named typology block — including every legitimate transaction, since
    Patterns.txt only ever contains laundering examples). Recovers `pattern_type` for
    3,209 of 5,177 laundering transactions (62%) on the real HI-Small data — the
    remaining 38% are laundering but don't match a named typology block, which is a
    known, disclosed data limitation (see CLAUDE.md's Dataset section), not a bug in
    this join.

    Row count and duplicate-key safety are asserted, not assumed: `patterns` is
    deduplicated on the join key first (so a txns row matches at most one pattern row),
    and this raises loudly if that dedup would have silently dropped a genuinely
    distinct pattern row, or if the merge somehow changes `txns`' row count.
    """
    join_cols = [c for c in PATTERN_COLUMNS if c != "pattern_type"]

    patterns = patterns.copy()
    patterns["from_bank"] = patterns["from_bank"].astype(txns["from_bank"].dtype)
    patterns["to_bank"] = patterns["to_bank"].astype(txns["to_bank"].dtype)

    dedup_patterns = patterns[[*join_cols, "pattern_type"]].drop_duplicates(subset=join_cols)
    if len(dedup_patterns) != len(patterns):
        raise ValueError(
            "join_pattern_types: duplicate join keys within patterns — matching would "
            "be ambiguous. This has not happened on the real HI-Small data; if it's "
            "happening now, the dedup below would silently hide it."
        )

    merged = txns.merge(dedup_patterns, on=join_cols, how="left")
    if len(merged) != len(txns):
        raise ValueError(
            "join_pattern_types: merge changed row count — duplicate join keys within "
            "txns matched multiple pattern rows unexpectedly."
        )
    merged.index = txns.index
    return merged


def dataset_stats(df: pd.DataFrame) -> dict:
    """Row count, class balance, date span, unique accounts — the brief's required
    first artifact, logged before any modelling starts.
    """
    unique_accounts = pd.concat([df["from_account_key"], df["to_account_key"]]).nunique()
    return {
        "row_count": len(df),
        "positive_count": int(df["is_laundering"].sum()),
        "positive_rate": float(df["is_laundering"].mean()),
        "date_min": df["timestamp"].min(),
        "date_max": df["timestamp"].max(),
        "unique_accounts": unique_accounts,
    }


if __name__ == "__main__":
    config = load_config()
    raw_dir = Path(config["paths"]["raw_dir"])

    txns = load_transactions(raw_dir / "HI-Small_Trans.csv")
    stats = dataset_stats(txns)
    print("HI-Small_Trans.csv stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    accounts = load_accounts(raw_dir / "HI-Small_accounts.csv")
    n_keys = accounts["account_key"].nunique()
    print(f"\nHI-Small_accounts.csv: {len(accounts)} rows, {n_keys} unique account keys")

    patterns = load_patterns(raw_dir / "HI-Small_Patterns.txt")
    n_typologies = patterns["pattern_type"].nunique()
    print(f"\nHI-Small_Patterns.txt: {len(patterns)} labelled txns, {n_typologies} typologies")
    print(patterns["pattern_type"].value_counts())
