# data_processing.py - cleaning and feature eng for the diabetes readmission dataset

import re
import warnings

import numpy as np
import pandas as pd


### Loading

def load_raw_data(filepath):
    # the dataset uses '?' for missing vals which is annoying
    df = pd.read_csv(filepath, na_values=["?"], keep_default_na=True)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns from {filepath}")
    return df


### Profiling
def profile_data(df):
    """Return a per-column summary with dtype, nulls, uniques, and sample values."""
    records = []
    for col in df.columns:
        non_null = df[col].dropna()
        sample_vals = non_null.unique()[:5].tolist() if len(non_null) > 0 else []
        records.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "null_pct": round(df[col].isna().mean() * 100, 2),
                "unique_count": int(df[col].nunique(dropna=True)),
                "sample_values": sample_vals,
            }
        )
    summary = pd.DataFrame(records).set_index("column")
    print(f"Profiled {len(summary)} columns")
    return summary


# Cleaning

HIGH_MISSING_COLS = ["weight", "payer_code", "medical_specialty"]
ID_COLS = ["encounter_id", "patient_nbr"]
NEAR_ZERO_VARIANCE_COLS = ["citoglipton", "examide"]
DEATH_HOSPICE_IDS = {11, 13, 14, 19, 20, 21}

def clean_data(df):
    df = df.copy()

    # drop cols with too many missing values
    cols_to_drop = [c for c in HIGH_MISSING_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"dropped {len(cols_to_drop)} cols with too many missing vals: {cols_to_drop}")

    # drop id columns
    id_cols = [c for c in ID_COLS if c in df.columns]
    df.drop(columns=id_cols, inplace=True)
    print(f"Dropped {len(id_cols)} identifier columns: {id_cols}")

    # near-zero variance cols
    nzv_cols = [c for c in NEAR_ZERO_VARIANCE_COLS if c in df.columns]
    df.drop(columns=nzv_cols, inplace=True)
    print(f"Dropped {len(nzv_cols)} near-zero-variance columns: {nzv_cols}")

    # fill remaining nulls
    if "race" in df.columns:
        n_race = int(df["race"].isna().sum())
        df["race"].fillna("Unknown", inplace=True)
        print(f"Filled {n_race:,} missing 'race' values with 'Unknown'")

    for diag in ["diag_1", "diag_2", "diag_3"]:
        if diag in df.columns:
            n_diag = int(df[diag].isna().sum())
            df[diag].fillna("0", inplace=True)
            if n_diag:
                print(f"Filled {n_diag:,} missing '{diag}' values with '0'")

    # drop rows where patient died (not useful for readmission prediction lol)
    if "discharge_disposition_id" in df.columns:
        mask = df["discharge_disposition_id"].isin(DEATH_HOSPICE_IDS)
        n_removed = int(mask.sum())
        df = df[~mask].reset_index(drop=True)
        print(f"Removed {n_removed:,} death/hospice rows")

    remaining_nulls = int(df.isna().sum().sum())
    print(f"Remaining null cells after cleaning: {remaining_nulls:,}")
    return df
