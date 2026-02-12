# modeling.py - train and evaluate models for readmission prediction

import warnings

import joblib
import numpy as np
import optuna
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# TODO: try LightGBM too


def split_data(X, y, test_size=0.2, random_state=42) -> tuple:
    """Stratified train/test split, prints class distributions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    print("--- Class distribution ---")
    for label, subset, name in [
        (y_train_arr, len(y_train_arr), "Train"),
        (y_test_arr, len(y_test_arr), "Test"),
    ]:
        unique, counts = np.unique(label, return_counts=True)
        dist = ", ".join(
            f"class {int(u)}: {c} ({c / subset:.2%})" for u, c in zip(unique, counts)
        )
        print(f"  {name} (n={subset}): {dist}")

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, random_state=42) -> LogisticRegression:
    """Train logistic regression with balanced weights."""
    # using balanced class weights bc of the class imbalance (~11% positive)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state=42) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# train xgboost with scale_pos_weight computed from the data
def train_xgboost(X_train, y_train, random_state=42):
    y_arr = np.asarray(y_train)
    n_negative = int(np.sum(y_arr == 0))
    n_positive = int(np.sum(y_arr == 1))
    scale_pos_weight = n_negative / max(n_positive, 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report_str = classification_report(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"  Results for {model_name}")
    print(f"{'='*60}")
    print(report_str)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "classification_report_str": report_str,
    }
