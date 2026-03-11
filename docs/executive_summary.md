# Reducing Hospital Readmissions Through Predictive Analytics and Targeted Outreach

---

## The Problem

- 30-day hospital readmissions affect roughly 1 in 9 of our diabetes patients.
- Each readmission costs an average of $15,000 in direct healthcare costs.
- Early readmission often signals that patients aren't following their treatment plan after discharge -- something we can actually help with through our platform.
- Currently, we have no systematic way to identify which patients need extra support after they leave the hospital.

## Our Approach

- We analyzed 100,000+ patient encounters from a major hospital network to understand the patterns behind readmission.
- We built a machine learning model that scores each patient's readmission risk at the time of discharge.
- We then simulated a clinical experiment to estimate what would happen if we sent personalized follow-up messages to high-risk patients 48 hours after discharge.

## Key Findings

- Our model identifies approximately 60-70% of patients who will be readmitted within 30 days, allowing us to focus resources on the patients who need them most.
- The strongest predictors of readmission are: number of previous hospitalizations, how the patient was discharged, number of medications, and whether their blood sugar (A1C) is under control.
- Patients aged 60+ with poorly controlled blood sugar and complex medication regimens are 2-3x more likely to be readmitted, and this is the cohort where our intervention will have the greatest impact since they tend to have the most to gain from consistent post-discharge follow-up and medication reminders.

## Simulated Intervention Results

- We simulated sending a personalized follow-up message (medication reminders + symptom check-in + care coordinator access) to half of our patients.
- The simulation projects a 15% reduction in readmissions for the outreach group.
- At our scale (50,000 patients/year), this translates to roughly 825 fewer readmissions annually.
- Estimated annual savings: $10M+ in reduced readmission costs, net of the $50/patient cost of the intervention.

## Recommendations

1. **Launch a pilot A/B test** (3 months) at 2-3 clinic locations to validate these projections with real patients.
2. **Integrate the risk model** into our discharge workflow so care coordinators can see each patient's risk score in real time.
3. **Start with the highest-risk segment** -- patients 60+ with elevated A1C and 3+ prior hospitalizations -- where the model is most confident and the ROI is highest.

## Next Steps

With these findings in hand, the natural next step is to move toward a real-world pilot. Specifically:

- Partner with the clinical team to design the follow-up message content and timing.
- Work with engineering to deploy the model as a real-time scoring API.
- Establish baseline metrics and a monitoring dashboard to track intervention effectiveness.
- Target: pilot launch within 6 weeks.

---

*For technical details, see the full analysis notebooks in the project repository.*
