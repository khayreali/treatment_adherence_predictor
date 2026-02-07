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


### Target encoding

def encode_target(df):
    """Binary target: readmitted <30 days = 1, else 0."""
    df = df.copy()
    if "readmitted" not in df.columns:
        raise KeyError("Column 'readmitted' not found in DataFrame")

    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    positive_rate = df["readmitted_30d"].mean() * 100
    df.drop(columns=["readmitted"], inplace=True)
    print(f"Encoded target: {positive_rate:.1f}% positive class rate (readmitted <30 days)")
    return df


# Feature engineering

def _map_icd9_to_category(code):
    # based on the dataset documentation, these ICD-9 ranges map to broad clinical categories
    if code in ("0", "", "nan") or pd.isna(code):
        return "other"

    code = str(code).strip()

    # diabetes
    if code.startswith("250"):
        return "diabetes"

    if code.startswith("E"):
        return "external_causes"
    if code.startswith("V"):
        return "supplementary"

    # try to parse the numeric part
    try:
        numeric = float(re.match(r"^[\d.]+", code).group())
    except (AttributeError, ValueError):
        return "other"

    if 390 <= numeric <= 459:
        return "circulatory"
    if 460 <= numeric <= 519:
        return "respiratory"
    if 520 <= numeric <= 579:
        return "digestive"
    if 800 <= numeric <= 999:
        return "injury"
    if 580 <= numeric <= 629:
        return "genitourinary"
    if 710 <= numeric <= 739:
        return "musculoskeletal"
    if 140 <= numeric <= 239:
        return "neoplasms"
    if 240 <= numeric <= 279:
        return "endocrine"  # non-diabetes endocrine
    if 680 <= numeric <= 709:
        return "skin"
    if 1 <= numeric <= 139:
        return "infectious"
    if 280 <= numeric <= 289:
        return "blood"
    if 290 <= numeric <= 319:
        return "mental"
    if 320 <= numeric <= 389:
        return "nervous"

    return "other"


def engineer_features(df):
    df = df.copy()

    # age bracket cleanup
    if "age" in df.columns:
        df["age"] = df["age"].str.replace(r"[\[\)']", "", regex=True)
        print(f"Simplified age brackets -> {df['age'].nunique()} groups")

    # total visits
    visit_cols = ["number_inpatient", "number_outpatient", "number_emergency"]
    present = [c for c in visit_cols if c in df.columns]
    if present:
        df["total_visits"] = df[present].sum(axis=1)
        print(f"Created 'total_visits' from {present}")

    # bin num_medications
    if "num_medications" in df.columns:
        df["num_medications_bin"] = pd.cut(
            df["num_medications"],
            bins=[0, 5, 15, np.inf],
            labels=["low", "medium", "high"],
            right=True,
        )
        print(f"Binned 'num_medications' -> {df['num_medications_bin'].value_counts().to_dict()}")

    # map diagnosis codes to categories
    for diag in ["diag_1", "diag_2", "diag_3"]:
        if diag in df.columns:
            df[diag] = df[diag].astype(str).map(_map_icd9_to_category)
            n_cats = df[diag].nunique()
            print(f"Mapped '{diag}' ICD-9 codes to {n_cats} broad categories")

    # A1Cresult -> ordinal
    if "A1Cresult" in df.columns:
        a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
        df["A1Cresult"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)
        print("Mapped 'A1Cresult' to ordinal (None->0, Norm->1, >7->2, >8->3)")

    if "change" in df.columns:
        df["change"] = df["change"].map({"Ch": 1, "No": 0}).fillna(0).astype(int)
        print("Mapped 'change' to binary (Ch->1, No->0)")

    if "diabetesMed" in df.columns:
        df["diabetesMed"] = (
            df["diabetesMed"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        )
        print("Mapped 'diabetesMed' to binary (Yes->1, No->0)")

    # gender -> binary
    if "gender" in df.columns:
        invalid_mask = ~df["gender"].isin(["Male", "Female"])
        n_invalid = int(invalid_mask.sum())
        if n_invalid:
            df = df[~invalid_mask].reset_index(drop=True)
            print(f"Dropped {n_invalid} rows with Unknown/Invalid gender")
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).astype(int)
        print("Mapped 'gender' to binary (Male->1, Female->0)")

    return df
