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

### section 2: cohort explorer

st.header("2. Cohort Explorer")

# TODO: add a filter for diagnosis type

try:
    df_clean = load_joblib(DF_CLEAN_PATH)

    if df_clean is None:
        st.warning(
            f"Cleaned dataset not found at `{DF_CLEAN_PATH}`. "
            "Run the analysis notebooks to generate model artifacts."
        )
    else:
        # find a usable readmission column
        readmit_col = None
        for candidate in ("readmitted_30d", "readmitted", "readmitted_binary", "readmit_flag"):
            if candidate in df_clean.columns:
                readmit_col = candidate
                break

        if readmit_col is None:
            st.info(
                "No readmission column found in the cleaned dataset. "
                "Cohort explorer requires a `readmitted` column."
            )
        else:
            # build filter options
            age_options = ["All"]
            if "age" in df_clean.columns:
                age_options += sorted(df_clean["age"].dropna().unique().tolist())

            a1c_options = ["All"]
            if "A1Cresult" in df_clean.columns:
                a1c_options += sorted(
                    df_clean["A1Cresult"].dropna().unique().astype(str).tolist()
                )

            med_options = ["All"]
            if "num_medications_bin" in df_clean.columns:
                med_options += sorted(
                    df_clean["num_medications_bin"]
                    .dropna()
                    .unique()
                    .astype(str)
                    .tolist()
                )
            elif "num_medications" in df_clean.columns:
                # create bins on the fly if needed
                df_clean["num_medications_bin"] = pd.cut(
                    df_clean["num_medications"],
                    bins=[0, 5, 10, 20, 50, 100],
                    labels=["0-5", "6-10", "11-20", "21-50", "51+"],
                )
                med_options += sorted(
                    df_clean["num_medications_bin"]
                    .dropna()
                    .unique()
                    .astype(str)
                    .tolist()
                )

            # dropdowns
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                sel_age = st.selectbox("Age Group", age_options)
            with filter_col2:
                sel_a1c = st.selectbox("A1C Result", a1c_options)
            with filter_col3:
                sel_med = st.selectbox("Number of Medications Bin", med_options)

            # apply filters
            mask = pd.Series(True, index=df_clean.index)
            if sel_age != "All" and "age" in df_clean.columns:
                mask &= df_clean["age"] == sel_age
            if sel_a1c != "All" and "A1Cresult" in df_clean.columns:
                mask &= df_clean["A1Cresult"].astype(str) == sel_a1c
            if sel_med != "All" and "num_medications_bin" in df_clean.columns:
                mask &= df_clean["num_medications_bin"].astype(str) == sel_med

            cohort = df_clean.loc[mask]

            overall_rate = float(df_clean[readmit_col].mean())
            cohort_rate = float(cohort[readmit_col].mean()) if len(cohort) > 0 else 0.0

            # metrics
            m1, m2 = st.columns(2)
            m1.metric("Cohort Size", f"{len(cohort):,}")
            m2.metric(
                "Cohort Readmission Rate",
                f"{cohort_rate:.1%}",
                delta=f"{cohort_rate - overall_rate:+.1%} vs overall",
            )

            # bar chart comparing cohort vs overall
            comparison_df = pd.DataFrame(
                {
                    "Group": ["Overall", "Selected Cohort"],
                    "Readmission Rate": [overall_rate, cohort_rate],
                }
            )
            fig_cohort = px.bar(
                comparison_df,
                x="Group",
                y="Readmission Rate",
                color="Group",
                text_auto=".1%",
                title="Readmission Rate: Cohort vs Overall",
                color_discrete_sequence=["#636EFA", "#EF553B"],
            )
            fig_cohort.update_layout(
                yaxis_title="Readmission Rate",
                yaxis_tickformat=".0%",
                showlegend=False,
            )
            st.plotly_chart(fig_cohort, use_container_width=True)

except Exception as exc:
    st.warning("Could not load Cohort Explorer data. Check that notebooks have been run.")
    st.exception(exc)

st.divider()


### section 3: a/b test results

st.header("3. A/B Test Results")

try:
    ab_results = load_joblib(AB_RESULTS_PATH)

    if ab_results is None:
        st.warning(
            f"A/B test results not found at `{AB_RESULTS_PATH}`. "
            "Run the notebooks first to generate results."
        )
    else:
        # pull out summary metrics from the nested dict structure
        # the notebook saves sim_results, test_results, etc. as sub-dicts
        sim_results = ab_results.get("sim_results", {})
        test_results = ab_results.get("test_results", {})

        sample_control = sim_results.get("control_total", 0)
        sample_treatment = sim_results.get("treatment_total", 0)
        control_rate = sim_results.get("control_rate_observed", 0.0)
        treatment_rate = sim_results.get("treatment_rate_observed", 0.0)
        p_value = test_results.get("p_value", 1.0)

        relative_lift = (
            (treatment_rate - control_rate) / control_rate
            if control_rate > 0
            else 0.0
        )

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Control Sample Size", f"{sample_control:,}")
        mc2.metric("Treatment Sample Size", f"{sample_treatment:,}")
        mc3.metric("Control Rate", f"{control_rate:.2%}")
        mc4.metric("Treatment Rate", f"{treatment_rate:.2%}")
        mc5.metric("p-value", f"{p_value:.4f}")

        st.caption(f"Relative lift: **{relative_lift:+.2%}**")

        # regenerate the charts using our ab_test module helpers when available
        ci_control = ab_results.get("ci_control")
        ci_treatment = ab_results.get("ci_treatment")
        n_per_group = ab_results.get("n_per_group", 0)
        baseline_rate = ab_results.get("baseline_rate", control_rate)
        treat_rate_true = ab_results.get("treatment_rate", treatment_rate)

        # if the ab_test helpers got imported, use them for nicer charts with CIs
        if plot_conversion_rates is not None and ci_control is not None and ci_treatment is not None:
            fig_conv = plot_conversion_rates(sim_results, ci_control, ci_treatment)
        else:
            # fallback: simple bar chart if the module isn't available
            conv_df = pd.DataFrame(
                {
                    "Group": ["Control", "Treatment"],
                    "Conversion Rate": [control_rate, treatment_rate],
                }
            )
            fig_conv = px.bar(
                conv_df,
                x="Group",
                y="Conversion Rate",
                color="Group",
                text_auto=".2%",
                title="Conversion Rate Comparison",
                color_discrete_sequence=["#636EFA", "#00CC96"],
            )
            fig_conv.update_layout(
                yaxis_title="Conversion Rate",
                yaxis_tickformat=".1%",
                showlegend=False,
            )

        if plot_cumulative_lift is not None and n_per_group > 0:
            fig_lift = plot_cumulative_lift(
                baseline_rate, treat_rate_true, n_per_group
            )
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(fig_conv, use_container_width=True)
            with chart_col2:
                st.plotly_chart(fig_lift, use_container_width=True)
        else:
            st.plotly_chart(fig_conv, use_container_width=True)
            st.info("Cumulative lift chart not available -- ab_test module couldn't be imported.")

        # business impact section
        # try to use the pre-computed business_impact from the notebook first
        biz = ab_results.get("business_impact", {})
        with st.expander("Business Impact -- Cost Savings Analysis"):
            if biz:
                st.markdown(
                    f"""
| Metric | Value |
|---|---|
| Annual readmissions (baseline) | {biz.get('annual_readmissions_baseline', 0):,.0f} |
| Annual readmissions (after intervention) | {biz.get('annual_readmissions_after', 0):,.0f} |
| Readmissions prevented | {biz.get('readmissions_prevented', 0):,.0f} |
| Gross cost savings | ${biz.get('annual_cost_savings', 0):,.0f} |
| Intervention cost (${biz.get('cost_per_patient_for_intervention', 50):.0f}/patient) | ${biz.get('annual_cost_savings', 0) - biz.get('net_savings', 0):,.0f} |
| **Net savings** | **${biz.get('net_savings', 0):,.0f}** |
| **ROI** | **{biz.get('roi_percentage', 0):,.1f}%** |
"""
                )
            else:
                # fallback rough estimate if business_impact wasn't saved
                avg_readmission_cost = 15_000.0
                annual_patients = 50_000
                reduction = abs(relative_lift) if relative_lift < 0 else relative_lift
                projected_savings = annual_patients * reduction * avg_readmission_cost
                st.markdown(
                    f"""
| Metric | Value |
|---|---|
| Average readmission cost | ${avg_readmission_cost:,.0f} |
| Annual eligible patients | {annual_patients:,} |
| Estimated reduction in readmission rate | {reduction:.2%} |
| **Projected annual savings** | **${projected_savings:,.0f}** |
"""
                )
            st.markdown(
                "These are rough estimates based on the A/B test lift "
                "applied to the full eligible population. Real savings "
                "will depend on patient mix and how well the intervention "
                "is actually carried out."
            )

except Exception as exc:
    st.warning("Could not load A/B Test results. Have you run the notebooks?")
    st.exception(exc)

st.divider()
