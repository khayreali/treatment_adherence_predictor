# ab_test.py - power analysis, simulation, and statistical tests for A/B experiment

import math

import numpy as np
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def compute_power_analysis(
    baseline_rate: float,
    relative_reduction: float,
    alpha=0.05,
    power=0.80,
) -> int:
    """Compute required sample size per group for a two-proportion z-test."""
    treatment_rate = baseline_rate * (1 - relative_reduction)
    effect_size = proportion_effectsize(baseline_rate, treatment_rate)

    analysis = NormalIndPower()
    n_per_group = math.ceil(
        analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=1.0,
            alternative="two-sided",
        )
    )

    # based on cohen's conventions: small=0.2, medium=0.5, large=0.8
    print(f"Power Analysis Parameters:")
    print(f"  Baseline rate: {baseline_rate:.4f}")
    print(f"  Treatment rate: {treatment_rate:.4f}")
    print(f"  Relative reduction: {relative_reduction:.2%}")
    print(f"  Effect size (h): {effect_size:.4f}")
    print(f"  Alpha: {alpha}")
    print(f"  Power: {power}")
    print(f"  Sample size/group: {n_per_group}")

    return n_per_group


def simulate_ab_test(control_rate, treatment_rate, n_per_group, random_state=42):
    # simulate binomial outcomes for both arms
    rng = np.random.default_rng(random_state)

    control_outcomes = rng.binomial(n=1, p=control_rate, size=n_per_group)
    treatment_outcomes = rng.binomial(n=1, p=treatment_rate, size=n_per_group)

    control_conversions = int(control_outcomes.sum())
    treatment_conversions = int(treatment_outcomes.sum())

    return {
        "control_conversions": control_conversions,
        "control_total": n_per_group,
        "treatment_conversions": treatment_conversions,
        "treatment_total": n_per_group,
        "control_rate_observed": control_conversions / n_per_group,
        "treatment_rate_observed": treatment_conversions / n_per_group,
    }


def run_chi_squared_test(sim_results):
    """Run chi-squared test on the 2x2 contingency table."""
    cc = sim_results["control_conversions"]
    ct = sim_results["control_total"]
    tc = sim_results["treatment_conversions"]
    tt = sim_results["treatment_total"]

    contingency_table = np.array([
        [cc, ct - cc],
        [tc, tt - tc],
    ])
    # print(contingency_table)  # useful for debugging

    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    # TODO: could also try fisher's exact test for small samples

    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "expected_frequencies": expected.tolist(),
        "degrees_of_freedom": int(dof),
    }


# wilson score interval - better than wald for small proportions
def compute_confidence_interval(successes, total, confidence=0.95):
    n = total
    p_hat = successes / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z

    denom = 1 + z2 / n
    centre = p_hat + z2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * n)) / n)

    lower = (centre - margin) / denom
    upper = (centre + margin) / denom

    return (lower, upper)


def compute_effect_size(control_rate, treatment_rate):
    absolute_difference = control_rate - treatment_rate

    if control_rate != 0:
        relative_reduction = absolute_difference / control_rate * 100
    else:
        relative_reduction = 0.0

    risk_ratio = treatment_rate / control_rate if control_rate != 0 else 0.0

    ctrl_odds = control_rate / (1 - control_rate)
    treat_odds = treatment_rate / (1 - treatment_rate)
    odds_ratio = treat_odds / ctrl_odds

    cohens_h = 2 * math.asin(math.sqrt(control_rate)) - 2 * math.asin(
        math.sqrt(treatment_rate)
    )

    return {
        "absolute_difference": absolute_difference,
        "relative_reduction": relative_reduction,
        "risk_ratio": risk_ratio,
        "odds_ratio": odds_ratio,
        "cohens_h": cohens_h,
    }


def compute_confidence_interval_difference(sim_results, confidence=0.95):
    """CI for the difference in proportions (treatment - control)."""
    p_c = sim_results["control_rate_observed"]
    p_t = sim_results["treatment_rate_observed"]
    n_c = sim_results["control_total"]
    n_t = sim_results["treatment_total"]

    point_est = p_t - p_c

    se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    ci_lower = point_est - z * se
    ci_upper = point_est + z * se

    return (point_est, ci_lower, ci_upper)


def plot_conversion_rates(sim_results, ci_control, ci_treatment) -> go.Figure:
    ctrl_rate = sim_results["control_rate_observed"]
    treat_rate = sim_results["treatment_rate_observed"]

    groups = ["Control", "Treatment"]
    rates = [ctrl_rate, treat_rate]
    colors = ["#1f77b4", "#2ca02c"]  # blue, green

    error_lower = [
        ctrl_rate - ci_control[0],
        treat_rate - ci_treatment[0],
    ]
    error_upper = [
        ci_control[1] - ctrl_rate,
        ci_treatment[1] - treat_rate,
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=groups,
                y=rates,
                marker_color=colors,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=error_upper,
                    arrayminus=error_lower,
                    color="rgba(0,0,0,0.6)",
                    thickness=1.5,
                    width=6,
                ),
                text=[f"{r:.4f}" for r in rates],
                textposition="outside",
            )
        ]
    )

    fig.add_annotation(
        x="Control", y=ci_control[1],
        text=f"Rate: {ctrl_rate:.4f}<br>CI: [{ci_control[0]:.4f}, {ci_control[1]:.4f}]",
        showarrow=True, arrowhead=2, ax=80, ay=-30, font=dict(size=11),
    )
    fig.add_annotation(
        x="Treatment", y=ci_treatment[1],
        text=f"Rate: {treat_rate:.4f}<br>CI: [{ci_treatment[0]:.4f}, {ci_treatment[1]:.4f}]",
        showarrow=True, arrowhead=2, ax=80, ay=-30, font=dict(size=11),
    )

    fig.update_layout(
        title="A/B Test: Observed Conversion Rates with 95% Confidence Intervals",
        yaxis_title="Conversion Rate",
        xaxis_title="Group",
        template="plotly_white",
        showlegend=False,
        yaxis=dict(rangemode="tozero"),
    )

    return fig


def plot_cumulative_lift(control_rate, treatment_rate, n_per_group, random_state=42):
    # this shows how the estimate stabilizes - pretty cool visualization
    rng = np.random.default_rng(random_state)

    control_outcomes = rng.binomial(n=1, p=control_rate, size=n_per_group)
    treatment_outcomes = rng.binomial(n=1, p=treatment_rate, size=n_per_group)

    enrollment = np.arange(1, n_per_group + 1)
    cumulative_control = np.cumsum(control_outcomes) / enrollment
    cumulative_treatment = np.cumsum(treatment_outcomes) / enrollment

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=enrollment, y=cumulative_control,
        mode="lines", name="Control (cumulative)",
        line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=enrollment, y=cumulative_treatment,
        mode="lines", name="Treatment (cumulative)",
        line=dict(color="#2ca02c", width=1.5),
    ))

    fig.add_hline(
        y=control_rate, line_dash="dash", line_color="#1f77b4",
        annotation_text=f"True control rate ({control_rate:.4f})",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=treatment_rate, line_dash="dash", line_color="#2ca02c",
        annotation_text=f"True treatment rate ({treatment_rate:.4f})",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Cumulative Conversion Rate by Enrollment Count",
        xaxis_title="Patients Enrolled (per group)",
        yaxis_title="Cumulative Conversion Rate",
        template="plotly_white",
        legend=dict(x=0.65, y=0.98),
    )

    return fig


def compute_business_impact(
    baseline_rate,
    relative_reduction,
    annual_patients=50_000,
    cost_per_readmission=15_000.0,
):
    # using CMS average readmission cost of $15k
    # $50 per patient for the follow-up call seems reasonable based on what I found
    cost_per_patient_for_intervention = 50.0

    annual_readmissions_baseline = baseline_rate * annual_patients
    treatment_rate = baseline_rate * (1 - relative_reduction)
    annual_readmissions_after = treatment_rate * annual_patients

    readmissions_prevented = annual_readmissions_baseline - annual_readmissions_after
    annual_cost_savings = readmissions_prevented * cost_per_readmission

    total_intervention_cost = annual_patients * cost_per_patient_for_intervention
    net_savings = annual_cost_savings - total_intervention_cost

    roi_pct = (net_savings / total_intervention_cost * 100) if total_intervention_cost > 0 else 0.0

    return {
        "annual_readmissions_baseline": annual_readmissions_baseline,
        "annual_readmissions_after": annual_readmissions_after,
        "readmissions_prevented": readmissions_prevented,
        "annual_cost_savings": annual_cost_savings,
        "cost_per_patient_for_intervention": cost_per_patient_for_intervention,
        "net_savings": net_savings,
        "roi_percentage": roi_pct,
    }


def run_full_ab_analysis(baseline_rate, relative_reduction=0.15, random_state=42):
    """Run the full A/B pipeline and return all results."""

    # power analysis
    n_per_group = compute_power_analysis(baseline_rate, relative_reduction)

    # simulation
    treatment_rate = baseline_rate * (1 - relative_reduction)
    sim_results = simulate_ab_test(
        control_rate=baseline_rate,
        treatment_rate=treatment_rate,
        n_per_group=n_per_group,
        random_state=random_state,
    )

    # chi-squared
    chi2_results = run_chi_squared_test(sim_results)

    # effect size
    effect_size = compute_effect_size(
        sim_results["control_rate_observed"],
        sim_results["treatment_rate_observed"],
    )

    # confidence intervals
    ci_control = compute_confidence_interval(
        sim_results["control_conversions"],
        sim_results["control_total"],
    )
    ci_treatment = compute_confidence_interval(
        sim_results["treatment_conversions"],
        sim_results["treatment_total"],
    )
    ci_diff = compute_confidence_interval_difference(sim_results)

    # business impact
    business = compute_business_impact(baseline_rate, relative_reduction)

    return {
        "power_analysis": {
            "n_per_group": n_per_group,
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
            "relative_reduction": relative_reduction,
        },
        "simulation": sim_results,
        "chi_squared_test": chi2_results,
        "effect_size": effect_size,
        "confidence_intervals": {
            "control": ci_control,
            "treatment": ci_treatment,
            "difference": {
                "point_estimate": ci_diff[0],
                "ci_lower": ci_diff[1],
                "ci_upper": ci_diff[2],
            },
        },
        "business_impact": business,
    }
