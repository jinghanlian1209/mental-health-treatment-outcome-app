import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

APP_TITLE = "MindCare Outcome Lab"
DATA_PATH = "mental_health_diagnosis_treatment_.csv"
TARGET = "Outcome"
ID_COL = "Patient ID"
DATE_COL = "Treatment Start Date"
RANDOM_STATE = 42

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.big-title {font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.2rem;}
.subtitle {font-size: 1.05rem; color: #5C5470; margin-bottom: 1rem;}
.card {background: white; border: 1px solid #ECE7FF; border-radius: 18px; padding: 1rem 1.1rem; box-shadow: 0 2px 18px rgba(80, 56, 150, 0.06);}
.metric-note {font-size: 0.82rem; color: #6B6477;}
.warning-card {background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 16px; padding: 0.9rem 1rem;}
.green-card {background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 16px; padding: 0.9rem 1rem;}
.purple-chip {display: inline-block; background: #EEE9FF; color: #5B21B6; border-radius: 999px; padding: 0.25rem 0.65rem; font-size: 0.82rem; margin: 0.15rem 0.1rem 0.15rem 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["Treatment Start Month"] = df[DATE_COL].dt.month
    df["Treatment Start DayOfYear"] = df[DATE_COL].dt.dayofyear
    return df


def make_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [c for c in X.columns if c not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


@st.cache_resource
def train_models(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in [TARGET, ID_COL, DATE_COL]]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    preprocessor, numeric_features, categorical_features = make_preprocessor(X)

    model_candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=450,
            max_depth=5,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    for name, model in model_candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"},
            return_train_score=False,
        )
        fitted_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        fitted_pipe.fit(X_train, y_train)
        test_pred = fitted_pipe.predict(X_test)
        fitted[name] = fitted_pipe
        rows.append(
            {
                "Model": name,
                "CV Accuracy": scores["test_accuracy"].mean(),
                "CV Accuracy SD": scores["test_accuracy"].std(),
                "CV Macro F1": scores["test_macro_f1"].mean(),
                "CV Macro F1 SD": scores["test_macro_f1"].std(),
                "Holdout Accuracy": accuracy_score(y_test, test_pred),
                "Holdout Macro F1": f1_score(y_test, test_pred, average="macro"),
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("CV Macro F1", ascending=False).reset_index(drop=True)
    best_model_name = metrics_df.loc[0, "Model"]
    best_pipe = fitted[best_model_name]

    # Fit best model on all data for the deployed demo predictor.
    production_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model_candidates[best_model_name])])
    production_pipe.fit(X, y)

    return {
        "feature_cols": feature_cols,
        "X": X,
        "y": y,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": fitted,
        "metrics": metrics_df,
        "best_model_name": best_model_name,
        "best_pipe": best_pipe,
        "production_pipe": production_pipe,
        "X_test": X_test,
        "y_test": y_test,
    }


@st.cache_data
def get_permutation_importance(_pipe, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    result = permutation_importance(
        _pipe,
        X_test,
        y_test,
        n_repeats=12,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
    )
    out = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": result.importances_mean,
            "SD": result.importances_std,
        }
    ).sort_values("Importance", ascending=False)
    return out


def prediction_badge(pred_label: str) -> str:
    if pred_label == "Improved":
        return "green-card"
    if pred_label == "Deteriorated":
        return "warning-card"
    return "card"


def make_patient_from_sidebar(df: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    st.sidebar.header("Build a patient profile")
    st.sidebar.caption("Use the controls below to explore model behavior. This is a teaching demo, not a clinical tool.")

    values = {}
    for col in numeric_features:
        series = df[col].dropna()
        if col in ["Treatment Start Month", "Treatment Start DayOfYear"]:
            default = int(series.median())
            min_val = int(series.min())
            max_val = int(series.max())
            values[col] = st.sidebar.slider(col, min_val, max_val, default, step=1)
        elif "Adherence" in col:
            values[col] = st.sidebar.slider(col, int(series.min()), int(series.max()), int(series.median()), step=1)
        elif "hrs/week" in col:
            values[col] = st.sidebar.slider(col, int(series.min()), int(series.max()), int(series.median()), step=1)
        else:
            values[col] = st.sidebar.slider(col, int(series.min()), int(series.max()), int(series.median()), step=1)

    st.sidebar.divider()
    for col in categorical_features:
        options = sorted(df[col].dropna().astype(str).unique().tolist())
        default_ix = 0
        values[col] = st.sidebar.selectbox(col, options, index=default_ix)

    # Preserve the same feature order used during training.
    return pd.DataFrame([{c: values[c] for c in numeric_features + categorical_features}])


def generate_summary(patient: pd.DataFrame, pred_label: str, proba_df: pd.DataFrame, similar_df: pd.DataFrame) -> str:
    p = patient.iloc[0]
    top_probability = float(proba_df.loc[proba_df["Outcome"] == pred_label, "Probability"].iloc[0])
    similar_counts = similar_df[TARGET].value_counts(normalize=True).mul(100).round(1).to_dict() if len(similar_df) else {}

    risks = []
    protections = []
    if p["Symptom Severity (1-10)"] >= 8:
        risks.append("high symptom severity")
    if p["Stress Level (1-10)"] >= 8:
        risks.append("high stress")
    if p["Sleep Quality (1-10)"] <= 5:
        risks.append("lower sleep quality")
    if p["Adherence to Treatment (%)"] >= 80:
        protections.append("strong adherence")
    if p["Physical Activity (hrs/week)"] >= 7:
        protections.append("higher physical activity")
    if p["Mood Score (1-10)"] >= 7:
        protections.append("higher mood score")

    risk_text = ", ".join(risks) if risks else "no single high-risk flag from the simple rule screen"
    protection_text = ", ".join(protections) if protections else "no strong protective flag from the simple rule screen"
    cohort_text = "; ".join([f"{k}: {v}%" for k, v in similar_counts.items()]) if similar_counts else "not enough close matches"

    return (
        f"For this simulated profile, the model's top predicted outcome is **{pred_label}** "
        f"with probability **{top_probability:.1%}**. The simple clinical rule screen flags: {risk_text}. "
        f"Potential protective signals include: {protection_text}. Among similar records in the dataset, the outcome mix is: {cohort_text}. "
        "Because the dataset is small and the validation performance is close to chance, this summary should be read as an interpretability exercise rather than medical guidance."
    )


def find_similar_patients(df: pd.DataFrame, patient: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    numeric_cols = [
        "Age",
        "Symptom Severity (1-10)",
        "Mood Score (1-10)",
        "Sleep Quality (1-10)",
        "Physical Activity (hrs/week)",
        "Treatment Duration (weeks)",
        "Stress Level (1-10)",
        "Treatment Progress (1-10)",
        "Adherence to Treatment (%)",
    ]
    available = [c for c in numeric_cols if c in df.columns and c in patient.columns]
    scaled = df[available].copy()
    p = patient[available].iloc[0]
    denom = scaled.std().replace(0, 1)
    distances = (((scaled - p) / denom) ** 2).sum(axis=1) ** 0.5
    out = df.assign(Similarity_Distance=distances).sort_values("Similarity_Distance").head(n)
    keep = [ID_COL, "Age", "Gender", "Diagnosis", "Symptom Severity (1-10)", "Mood Score (1-10)", "Stress Level (1-10)", "Adherence to Treatment (%)", TARGET, "Similarity_Distance"]
    return out[[c for c in keep if c in out.columns]]


def make_probability_plot(proba_df: pd.DataFrame):
    fig = px.bar(
        proba_df,
        x="Outcome",
        y="Probability",
        text=proba_df["Probability"].map(lambda x: f"{x:.1%}"),
        range_y=[0, 1],
        title="Predicted outcome probabilities",
    )
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=360, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def make_confusion_heatmap(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        labels=dict(x="Predicted", y="Observed", color="Count"),
        title="Holdout confusion matrix",
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def subgroup_performance(pipe, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, subgroup: str) -> pd.DataFrame:
    pred = pipe.predict(X)
    temp = df.loc[X.index, [subgroup]].copy()
    temp["Observed"] = y.values
    temp["Predicted"] = pred
    rows = []
    for group, g in temp.groupby(subgroup):
        if len(g) < 10:
            continue
        rows.append(
            {
                "Subgroup": group,
                "N": len(g),
                "Accuracy": accuracy_score(g["Observed"], g["Predicted"]),
                "Macro F1": f1_score(g["Observed"], g["Predicted"], average="macro"),
                "Improved Rate": (g["Observed"] == "Improved").mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("Macro F1", ascending=False)


# ------------------------- App -------------------------
df = load_data()
artifacts = train_models(df)

st.markdown(f'<div class="big-title">🧠 {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">An interactive machine-learning web app for mental health treatment outcome prediction, model interpretability, and safety auditing.</div>',
    unsafe_allow_html=True,
)

patient = make_patient_from_sidebar(df, artifacts["numeric_features"], artifacts["categorical_features"])
# Reorder sidebar-generated patient to training columns exactly.
patient = patient.reindex(columns=artifacts["feature_cols"])

pipe = artifacts["production_pipe"]
classes = pipe.classes_.tolist()
proba = pipe.predict_proba(patient)[0]
pred_label = pipe.predict(patient)[0]
proba_df = pd.DataFrame({"Outcome": classes, "Probability": proba}).sort_values("Probability", ascending=False)

best_model_name = artifacts["best_model_name"]
metrics = artifacts["metrics"]
baseline = df[TARGET].value_counts(normalize=True).max()
best_cv = float(metrics.loc[metrics["Model"] == best_model_name, "CV Macro F1"].iloc[0])

cols = st.columns([1.1, 1, 1])
with cols[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Dataset size", f"{df.shape[0]} patients")
    st.caption("Synthetic/secondary educational dataset. No patient-identifying information is displayed in the app.")
    st.markdown("</div>", unsafe_allow_html=True)
with cols[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Selected ML model", best_model_name)
    st.caption("Chosen by 5-fold cross-validated macro F1.")
    st.markdown("</div>", unsafe_allow_html=True)
with cols[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Best CV macro F1", f"{best_cv:.3f}")
    st.caption(f"Majority-class baseline accuracy: {baseline:.3f}. Low signal is part of the model audit.")
    st.markdown("</div>", unsafe_allow_html=True)

st.info(
    "Safety note: this app is for a course project and portfolio demonstration. "
    "It is not a diagnostic, treatment, or crisis-response tool. The dataset is small and model performance is close to chance, so the app emphasizes uncertainty and responsible model interpretation."
)

tab_predict, tab_explore, tab_model, tab_fairness, tab_about = st.tabs(
    ["🔮 Patient simulator", "📊 Cohort explorer", "🧪 Model lab", "⚖️ Fairness & safety", "📌 Project brief"]
)

with tab_predict:
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Prediction for the selected profile")
        st.markdown(f'<div class="{prediction_badge(pred_label)}">', unsafe_allow_html=True)
        st.markdown(f"### Top predicted outcome: **{pred_label}**")
        st.write("The probability distribution below is more important than the single top label. A flat distribution means the model is unsure.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.plotly_chart(make_probability_plot(proba_df), use_container_width=True)

        st.subheader("AI-style plain-language summary")
        similar = find_similar_patients(df, patient, n=8)
        st.write(generate_summary(patient, pred_label, proba_df, similar))

    with right:
        st.subheader("Current patient profile")
        st.dataframe(patient.T.rename(columns={0: "Value"}), use_container_width=True)
        st.subheader("Most similar records")
        st.dataframe(similar, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("What-if simulator")
    st.caption("Change one actionable variable while keeping the other profile values fixed. This shows sensitivity, not causal effect.")
    what_if_feature = st.selectbox(
        "Choose feature to vary",
        ["Adherence to Treatment (%)", "Physical Activity (hrs/week)", "Sleep Quality (1-10)", "Stress Level (1-10)", "Mood Score (1-10)", "Treatment Progress (1-10)"],
    )
    values = sorted(df[what_if_feature].dropna().unique().tolist())
    scenarios = []
    for v in values:
        row = patient.copy()
        row.loc[0, what_if_feature] = v
        probs = pipe.predict_proba(row)[0]
        for cls, pr in zip(classes, probs):
            scenarios.append({what_if_feature: v, "Outcome": cls, "Probability": pr})
    sim_df = pd.DataFrame(scenarios)
    fig = px.line(sim_df, x=what_if_feature, y="Probability", color="Outcome", markers=True, title="Outcome probabilities across what-if values")
    fig.update_layout(yaxis_tickformat=".0%", height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab_explore:
    st.subheader("Dataset exploration")
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_diagnosis = st.multiselect("Diagnosis", sorted(df["Diagnosis"].unique()), default=sorted(df["Diagnosis"].unique()))
    with c2:
        selected_outcome = st.multiselect("Outcome", sorted(df[TARGET].unique()), default=sorted(df[TARGET].unique()))
    with c3:
        selected_gender = st.multiselect("Gender", sorted(df["Gender"].unique()), default=sorted(df["Gender"].unique()))

    filtered = df[df["Diagnosis"].isin(selected_diagnosis) & df[TARGET].isin(selected_outcome) & df["Gender"].isin(selected_gender)]
    st.caption(f"Showing {len(filtered)} of {len(df)} records")

    cc1, cc2 = st.columns(2)
    with cc1:
        fig = px.histogram(filtered, x=TARGET, color="Diagnosis", barmode="group", title="Outcome distribution by diagnosis")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        numeric_choice = st.selectbox(
            "Numeric feature for boxplot",
            ["Symptom Severity (1-10)", "Mood Score (1-10)", "Sleep Quality (1-10)", "Physical Activity (hrs/week)", "Stress Level (1-10)", "Adherence to Treatment (%)"],
        )
        fig = px.box(filtered, x=TARGET, y=numeric_choice, color=TARGET, points="all", title=f"{numeric_choice} by outcome")
        fig.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation view")
    corr_cols = [
        "Age",
        "Symptom Severity (1-10)",
        "Mood Score (1-10)",
        "Sleep Quality (1-10)",
        "Physical Activity (hrs/week)",
        "Treatment Duration (weeks)",
        "Stress Level (1-10)",
        "Treatment Progress (1-10)",
        "Adherence to Treatment (%)",
    ]
    corr = filtered[corr_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", title="Numeric feature correlation matrix")
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab_model:
    st.subheader("Model comparison")
    st.write("The app compares three baseline models and selects the best one by 5-fold cross-validated macro F1.")
    display_metrics = metrics.copy()
    for col in display_metrics.columns:
        if col != "Model":
            display_metrics[col] = display_metrics[col].map(lambda x: f"{x:.3f}")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        holdout_pipe = artifacts["best_pipe"]
        y_pred = holdout_pipe.predict(artifacts["X_test"])
        st.plotly_chart(make_confusion_heatmap(artifacts["y_test"], y_pred, classes), use_container_width=True)
    with c2:
        imp = get_permutation_importance(artifacts["best_pipe"], artifacts["X_test"], artifacts["y_test"])
        fig = px.bar(imp.head(12).sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Permutation importance on holdout data")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Model audit finding: The best model performs only slightly above a naive baseline. "
        "This suggests that the available variables do not contain enough predictive signal for reliable individual-level treatment outcome prediction. "
        "A stronger project version would add longitudinal symptom trajectories, clinical notes, validated scale items, or a larger real-world cohort."
    )

with tab_fairness:
    st.subheader("Subgroup performance audit")
    subgroup = st.selectbox("Audit subgroup", ["Gender", "Diagnosis", "Medication", "Therapy Type", "AI-Detected Emotional State"])
    perf = subgroup_performance(artifacts["best_pipe"], artifacts["X_test"], artifacts["y_test"], df, subgroup)
    if perf.empty:
        st.write("Not enough observations for this subgroup audit.")
    else:
        st.dataframe(perf, use_container_width=True, hide_index=True)
        fig = px.bar(perf, x="Subgroup", y="Macro F1", hover_data=["N", "Accuracy", "Improved Rate"], title=f"Holdout macro F1 by {subgroup}")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Responsible AI checklist")
    st.markdown(
        """
        <span class="purple-chip">Small dataset warning</span>
        <span class="purple-chip">No clinical deployment</span>
        <span class="purple-chip">Human review required</span>
        <span class="purple-chip">Subgroup performance reported</span>
        <span class="purple-chip">Uncertainty shown</span>
        <span class="purple-chip">No sensitive free-text stored</span>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "A clinically responsible version would require prospective validation, richer clinical variables, privacy review, calibration testing, and crisis escalation pathways."
    )

with tab_about:
    st.subheader("Project concept")
    st.markdown(
        """
        **MindCare Outcome Lab** is an interactive biomedical AI project that asks a practical question:
        can a small structured mental-health treatment dataset support individual-level outcome prediction?

        The app lets users:
        1. Build a patient profile and view predicted treatment outcome probabilities.
        2. Compare similar patients from the dataset.
        3. Explore what-if changes in adherence, activity, sleep, stress, and mood.
        4. Compare several ML models.
        5. Audit performance across subgroups.

        The most important conclusion is not simply the prediction. The project demonstrates responsible AI reasoning:
        when the data signal is weak, a model should say so instead of pretending to be a crystal ball in a lab coat.
        """
    )
    st.subheader("Suggested course sign-up entry")
    st.code(
        "Project type: App\n"
        "Description: MindCare Outcome Lab: an interactive machine learning web app for mental health treatment outcome prediction, model interpretability, what-if simulation, and subgroup safety auditing.\n"
        "Dataset: Mental Health Diagnosis and Treatment Monitoring Dataset (uploaded CSV / Kaggle-style structured clinical dataset)\n"
        "Comments: Streamlit app with model comparison, patient simulator, similar-patient retrieval, permutation importance, and fairness/safety model card."
    )
    st.subheader("Limitations")
    st.write(
        "The dataset has only 500 records and outcome labels are nearly balanced. In validation, models perform close to chance, so this app is best framed as an educational model-audit tool rather than a reliable clinical predictor."
    )
