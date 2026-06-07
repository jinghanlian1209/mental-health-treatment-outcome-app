# Project Brief: MindCare Outcome Lab

## One-sentence pitch

An interactive biomedical AI web application that predicts mental-health treatment outcome probabilities while also auditing whether the model is reliable enough to trust.

## Why this project fits the extra-credit requirements

- **Biomedical domain:** Mental health diagnosis and treatment monitoring.
- **AI / machine learning:** Supervised multi-class classification, model comparison, permutation importance, subgroup auditing, and what-if simulation.
- **Interactive final product:** A Streamlit web app that users can explore without a live presentation.
- **Portfolio value:** The app demonstrates not only coding and modeling, but also responsible AI thinking, clinical uncertainty, and communication for nontechnical users.

## Research question

Can structured patient information, treatment characteristics, and AI-detected emotional state predict whether a patient improves, deteriorates, or experiences no change during mental-health treatment?

## Methods

1. Load and clean the structured dataset.
2. Engineer date features from treatment start date.
3. Split features into numeric and categorical variables.
4. Build preprocessing pipelines with median imputation, standardization, and one-hot encoding.
5. Compare Logistic Regression, Random Forest, and Gradient Boosting with 5-fold cross-validation.
6. Select the best model by macro F1 score.
7. Build an interactive Streamlit app with prediction, what-if analysis, similar-patient retrieval, feature importance, and subgroup safety audit.

## Key limitation and framing

The dataset has 500 observations and nearly balanced outcome classes. Validation performance is close to chance. Instead of hiding this, the app turns the limitation into a responsible AI message: clinical ML systems must show uncertainty and must not overstate predictive power when the data are weak.

## Suggested final submission text

I built **MindCare Outcome Lab**, an interactive machine-learning web app for mental-health treatment outcome prediction and responsible AI auditing. Users can create a simulated patient profile, view predicted probabilities for treatment outcomes, explore what-if changes in adherence or symptom-related variables, inspect similar patients, compare model performance, and audit subgroup performance across demographic and clinical categories. The app also includes a model-card style safety discussion because the dataset shows weak predictive signal, making uncertainty communication a core part of the project.
