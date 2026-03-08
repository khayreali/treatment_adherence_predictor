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


def plot_roc_curves(models, X_test, y_test) -> go.Figure:
    """Plot ROC curves for each model in the dict."""
    # plotly is way nicer than matplotlib for this
    fig = go.Figure()

    # diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Random (AUC = 0.50)",
            showlegend=True,
        )
    )

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc_val:.3f})",
            )
        )

    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.58, y=0.05),
        template="plotly_white",
        width=700,
        height=500,
    )
    return fig


def plot_pr_curves(models, X_test, y_test) -> go.Figure:
    fig = go.Figure()

    # baseline = positive class prevalence
    baseline = np.mean(np.asarray(y_test))
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name=f"Baseline (prevalence = {baseline:.2f})",
            showlegend=True,
        )
    )

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        fig.add_trace(
            go.Scatter(
                x=recall_vals,
                y=precision_vals,
                mode="lines",
                name=f"{name} (AP = {pr_auc:.3f})",
            )
        )

    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(x=0.02, y=0.05),
        template="plotly_white",
        width=700,
        height=500,
    )
    return fig


# FIXME: the SHAP fallback might not work for all model types
def plot_feature_importance_shap(model, X_test, feature_names, top_n=20) -> go.Figure:
    importance_values = None
    importance_label = "Mean |SHAP value|"

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # for binary classifiers shap_values may be a list of two arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        importance_values = np.abs(shap_values).mean(axis=0)
    except Exception:
        warnings.warn(
            "SHAP TreeExplainer failed; falling back to model.feature_importances_.",
            stacklevel=2,
        )

    if importance_values is None:
        if hasattr(model, "feature_importances_"):
            importance_values = np.asarray(model.feature_importances_)
            importance_label = "Feature importance"
        else:
            raise ValueError(
                "Model does not support feature_importances_ and SHAP failed."
            )

    # select top N
    top_indices = np.argsort(importance_values)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importance_values[top_indices]

    fig = go.Figure(
        go.Bar(
            x=top_importances,
            y=top_features,
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title=importance_label,
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, top_n * 25),
        width=700,
        yaxis=dict(autorange="reversed"),
    )
    # reverse so most important feature is at the top
    fig.update_yaxes(autorange=True)
    return fig


# see: sklearn calibration_curve docs
def plot_calibration_curve(model, X_test, y_test, model_name, n_bins=10) -> go.Figure:
    y_proba = model.predict_proba(X_test)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=n_bins
    )

    fig = go.Figure()

    # perfectly calibrated reference
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Perfectly calibrated",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode="lines+markers",
            name=model_name,
        )
    )

    fig.update_layout(
        title=f"Calibration Curve — {model_name}",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        template="plotly_white",
        width=700,
        height=500,
        legend=dict(x=0.58, y=0.05),
    )
    return fig


def tune_with_optuna(X_train, y_train, X_val, y_val, model_type="xgboost", n_trials=50) -> dict:
    """Tune hyperparams with Optuna, maximizing ROC-AUC."""

    if model_type != "xgboost":
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Currently only 'xgboost' is supported."
        )

    y_train_arr = np.asarray(y_train)
    n_negative = int(np.sum(y_train_arr == 0))
    n_positive = int(np.sum(y_train_arr == 1))
    scale_pos_weight = n_negative / max(n_positive, 1)

    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }

        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train, verbose=False)

        y_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_proba)

    # suppress optuna logging, it's super noisy
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"best AUC: {study.best_value:.4f}")
    print(f"best params: {study.best_params}")

    return study.best_params


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model
