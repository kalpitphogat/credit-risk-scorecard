"""Load the raw Lending Club file into SQLite, build the modeling sample with
SQL, and add the few features that are easier in pandas."""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "LoanStats3a.csv"
SQL_FILE = ROOT / "sql" / "build_sample.sql"
DB_FILE = ROOT / "data" / "loans.sqlite"


def build_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (model_sample, exclusion_waterfall)."""
    # First line of the file is a note, not the header.
    raw = pd.read_csv(RAW_CSV, skiprows=1, dtype=str, low_memory=False)

    with sqlite3.connect(DB_FILE) as con:
        raw.to_sql("raw_loans", con, if_exists="replace", index=False)
        con.executescript(SQL_FILE.read_text())
        sample = pd.read_sql("SELECT * FROM model_sample", con)
        waterfall = pd.read_sql("SELECT * FROM sample_waterfall ORDER BY step", con)

    return add_features(sample), waterfall


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    issue = pd.to_datetime(df["issue_d"], format="%b-%Y")
    first_line = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["issue_date"] = issue
    df["credit_history_months"] = (
        (issue.dt.year - first_line.dt.year) * 12 + (issue.dt.month - first_line.dt.month)
    )

    # "10+ years" -> 10, "< 1 year" -> 0, "n/a" -> NaN
    emp = df["emp_length"].str.extract(r"(\d+)")[0].astype(float)
    emp[df["emp_length"].str.contains("<", na=False)] = 0
    df["emp_years"] = emp

    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER"})
    return df.drop(columns=["emp_length", "earliest_cr_line"])


# Characteristics offered to the model (application-time only).
NUMERIC = [
    "loan_amnt", "annual_inc", "pti", "dti", "revol_util", "revol_bal",
    "inq_last_6mths", "delinq_2yrs", "pub_rec", "open_acc", "total_acc",
    "mths_since_last_delinq", "credit_history_months", "emp_years",
]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]
CANDIDATES = NUMERIC + CATEGORICAL
