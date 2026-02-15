# Treatment Adherence & Readmission Risk Predictor

## TL;DR

- Built an ML pipeline on **101,766 hospital encounters** to predict 30-day diabetes readmissions
- XGBoost model hits **~0.67 ROC-AUC** -- enough to meaningfully triage high-risk patients
- Simulated A/B test shows a follow-up intervention could prevent **~825 readmissions/year**, saving **$10M+** per 50K-patient cohort
- Top risk drivers: number of inpatient visits, discharge disposition, and medication count

## Key Results

### Model Performance

The tuned XGBoost beat logistic regression and random forest across the board. Not amazing AUC, but for healthcare readmission prediction this is pretty solid -- most published benchmarks on this dataset land in the 0.63-0.68 range.

![ROC Curves](assets/roc_curves.png)

### What Drives Readmission

SHAP values on the final model. Prior inpatient visits dominate everything else, which makes clinical sense -- if you've been admitted before, you're more likely to come back.

![Feature Importance](assets/feature_importance.png)

### Risk Score Distribution

How the model's predicted risk scores distribute across the patient population. The long right tail is where targeted outreach would focus.

![Risk Distribution](assets/risk_distribution.png)

## What This Project Does

Built a pipeline that predicts which diabetes patients will be readmitted within 30 days, then simulated what would happen if we actually intervened on the high-risk ones. I trained three models (logistic regression, random forest, XGBoost), picked the best one, interpreted it with SHAP, and ran an A/B test simulation to estimate real-world cost savings.

## Project Structure

```
treatment-adherence-predictor/
├── assets/                          # Chart PNGs for README
├── data/
│   └── diabetic_data.csv
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_modeling.ipynb            # Model training & evaluation
│   └── 03_ab_test_simulation.ipynb  # A/B test & cost analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py           # Cleaning & feature engineering
│   ├── modeling.py                  # Train/evaluate pipeline
│   ├── ab_test.py                   # Simulation logic
│   └── dashboard.py                 # Streamlit app
├── outputs/
│   ├── figures/                     # Interactive Plotly charts
│   └── models/                      # Saved models & artifacts
├── docs/
│   └── executive_summary.md
├── requirements.txt
└── README.md
```

## How to Run

Make sure you have Python 3.10+ installed.

```bash
git clone https://github.com/yourusername/treatment-adherence-predictor.git
cd treatment-adherence-predictor
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Download `diabetic_data.csv` from [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes) and drop it in `data/`. Then run the notebooks in order (01 -> 02 -> 03).

Or just look at the notebooks on GitHub -- they have all the outputs embedded.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, Plotly, Streamlit, SciPy, joblib

## Data

Uses the [Diabetes 130-US Hospitals (1999-2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) dataset from the UCI ML Repository -- 101,766 encounters across 130 hospitals.
