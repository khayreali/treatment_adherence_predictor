# dashboard.py - streamlit app for the treatment adherence predictor
# run with: streamlit run src/dashboard.py

# NOTE: need to run notebooks first to generate the model files

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, precision_score, recall_score

OUTPUTS_DIR = Path("outputs/models")

MODEL_PATH = OUTPUTS_DIR / "best_model.joblib"
X_TEST_PATH = OUTPUTS_DIR / "X_test.joblib"
Y_TEST_PATH = OUTPUTS_DIR / "y_test.joblib"
DF_CLEAN_PATH = OUTPUTS_DIR / "df_clean.joblib"
AB_RESULTS_PATH = OUTPUTS_DIR / "ab_results.joblib"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.joblib"

# page config
st.set_page_config(
    page_title="Treatment Adherence Predictor",
    layout="wide",
    page_icon="\U0001f3e5",  # hospital emoji
)

# data loading helpers

@st.cache_data(show_spinner=False)
def load_joblib(path):
    # loads a joblib file, returns None if it doesn't exist
    import joblib
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def get_risk_scores(_model, X_test):
    # get predicted probabilities for the positive class
    if hasattr(_model, "predict_proba"):
        return _model.predict_proba(X_test)[:, 1]
    # fallback to decision_function
    scores = _model.decision_function(X_test)
    # this is a bit hacky but it works
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def _metrics_at_threshold(y_true, scores, threshold):
    # compute precision recall f1 at a given threshold
    y_pred = (scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    high_risk_count = int(y_pred.sum())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "high_risk_count": high_risk_count,
    }


# sidebar
with st.sidebar:
    st.markdown("# \U0001f3e5 Treatment Adherence Predictor")
    st.markdown(
        "Predict which patients are at risk of treatment "
        "non-adherence so care teams can step in early."
    )
    st.divider()
    st.markdown(
        "**Data:** Diabetes 130-US Hospitals Dataset "
        "(1999-2008), 10 years of clinical data from "
        "130 US hospitals."
    )
    st.markdown(
        "**Pipeline:** cleaning, feature engineering, model "
        "training (XGBoost/LightGBM), SHAP explainability, "
        "and A/B testing."
    )

### section 1: risk overview

st.header("1. Patient Risk Overview")

try:
    model = load_joblib(MODEL_PATH)
    X_test = load_joblib(X_TEST_PATH)
    y_test = load_joblib(Y_TEST_PATH)

    if model is None or X_test is None or y_test is None:
        st.warning(
            f"One or more required files are missing "
            f"({MODEL_PATH}, {X_TEST_PATH}, {Y_TEST_PATH}). "
            "Please run the analysis notebooks first to generate the model files in outputs/models/."
        )
    else:
        y_true = np.asarray(y_test)
        scores = get_risk_scores(model, X_test)

        # histogram of predicted risk scores
        fig_hist = px.histogram(
            x=scores,
            nbins=50,
            labels={"x": "Predicted Risk Score", "y": "Patient Count"},
            title="Distribution of Predicted Risk Scores",
            color_discrete_sequence=["#636EFA"],
        )
        fig_hist.update_layout(
            xaxis_title="Predicted Risk Score",
            yaxis_title="Patient Count",
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # threshold slider
        threshold = st.slider(
            "Risk threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Patients with a predicted score at or above this threshold are flagged as high-risk.",
        )

        metrics = _metrics_at_threshold(y_true, scores, threshold)

        # metrics row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", f"{len(y_true):,}")
        col2.metric("High-Risk Count", f"{metrics['high_risk_count']:,}")
        col3.metric("Precision", f"{metrics['precision']:.3f}")
        col4.metric("Recall", f"{metrics['recall']:.3f}")

        st.caption(f"F1 Score at threshold {threshold:.2f}: **{metrics['f1']:.3f}**")

except Exception as exc:
    st.warning("Could not load Patient Risk Overview data. Make sure you've run the notebooks first.")
    st.exception(exc)

st.divider()
