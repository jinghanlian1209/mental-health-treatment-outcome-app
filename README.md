# MindCare Outcome Lab

**MindCare Outcome Lab** is a Streamlit web app for an individual extra-credit project in biomedical AI / machine learning.

The app uses a structured mental-health diagnosis and treatment dataset to explore whether treatment outcomes can be predicted from patient demographics, symptoms, treatment type, adherence, activity, stress, sleep, and AI-detected emotional state.

## Final output

A public-facing interactive web app that can be shared with classmates, instructors, or recruiters.

## Main app features

- Patient profile simulator with predicted probabilities for **Improved**, **No Change**, and **Deteriorated** outcomes
- What-if simulator for adherence, activity, sleep, stress, mood, and progress
- Similar-patient retrieval from the dataset
- Model comparison across Logistic Regression, Random Forest, and Gradient Boosting
- Permutation importance for model interpretation
- Subgroup audit by gender, diagnosis, medication, therapy type, and AI-detected emotional state
- Responsible AI checklist and limitations

## Dataset

File included in this folder:

```text
mental_health_diagnosis_treatment_.csv
```

The dataset contains 500 rows and 17 columns, including patient age, diagnosis, symptom severity, mood score, sleep quality, physical activity, treatment type, medication, adherence, AI-detected emotional state, and outcome.

## Important modeling note

The project is intentionally honest about model performance. In cross-validation, the best model performs only slightly above a naive baseline, suggesting the dataset has weak individual-level predictive signal. This is framed as a responsible AI lesson: a clinical model should communicate uncertainty rather than overclaim.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload these files:
   - `app.py`
   - `mental_health_diagnosis_treatment_.csv`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Go to Streamlit Community Cloud.
4. Choose the repository and set the main file path to `app.py`.
5. Deploy and copy the public app link into the course sign-up sheet or final submission.

## Suggested sign-up sheet entry

**Project type:** App  
**Description:** MindCare Outcome Lab: an interactive machine learning web app for mental health treatment outcome prediction, model interpretability, what-if simulation, and subgroup safety auditing.  
**Dataset:** Mental Health Diagnosis and Treatment Monitoring Dataset.  
**Comments:** Streamlit app with model comparison, patient simulator, similar-patient retrieval, permutation importance, and fairness/safety model card.

## Not medical advice

This app is only a course project and portfolio demonstration. It is not a diagnostic, treatment, or crisis-response tool.
