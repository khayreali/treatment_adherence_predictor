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
