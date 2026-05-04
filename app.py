# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REST API Abuse Detection",
    layout="wide"
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'data',    'cyber_attacks_dataset.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("API Abuse Detection")
st.sidebar.markdown("**IEEE CloudCom 2021**")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA", "Model Results", "Live Prediction"]
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    try:
        model      = joblib.load(os.path.join(MODELS_DIR, 'best_model_rf.joblib'))
        scaler     = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        le_target  = joblib.load(os.path.join(MODELS_DIR, 'label_encoder_target.joblib'))
        feat_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.joblib'))
        return model, scaler, le_target, feat_names
    except Exception:
        return None, None, None, None

df = load_data()
model, scaler, le_target, feat_names = load_models()

# ── Helper: show saved figure or regenerate ───────────────────────────────────
def show_figure(filename):
    path = os.path.join(FIGURES_DIR, filename)
    if os.path.isfile(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Figure not found: {filename}. Run main.py first.")

# ==============================================================================
# PAGE 1 — Overview
# ==============================================================================
if page == "Overview":
    st.title("REST API Abuse Detection")
    st.subheader("Using Request Pattern Analysis & Machine Learning")
    st.markdown("""
    > **Reference:** IEEE CloudCom 2021 — *"API Security Threat Detection Using Machine Learning"*
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df):,}")
    col2.metric("Attack Types",    df['Attack_Type'].nunique())
    col3.metric("Industries",      df['Target_Industry'].nunique())
    col4.metric("Locations",       df['Location'].nunique())

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Attack Types")
        attack_counts = df['Attack_Type'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.Set3(np.linspace(0, 1, len(attack_counts)))
        ax.barh(attack_counts.index, attack_counts.values, color=colors)
        ax.set_xlabel("Count")
        ax.set_title("Distribution of Attack Types", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Severity Breakdown")
        sev_counts = df['Severity'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(sev_counts.values, labels=sev_counts.index, autopct='%1.1f%%',
               colors=['#66b3ff','#ffcc99','#ff9999','#99ff99'], startangle=140)
        ax.set_title("Severity Distribution", fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("API vs Non-API Attacks")
    API_ATTACKS = ['SQL Injection', 'DDoS', 'Brute Force Attack',
                   'Cross-site Scripting (XSS)', 'Privilege Escalation', 'Man-in-the-Middle']
    api_count    = df['Attack_Type'].isin(API_ATTACKS).sum()
    nonapi_count = len(df) - api_count
    col1, col2 = st.columns(2)
    col1.metric("API-Specific Attacks",     f"{api_count:,}",    f"{api_count/len(df)*100:.1f}%")
    col2.metric("Non-API Attacks",          f"{nonapi_count:,}", f"{nonapi_count/len(df)*100:.1f}%")

    st.subheader("Sample Data")
    st.dataframe(df.sample(10, random_state=42).reset_index(drop=True), use_container_width=True)

# ==============================================================================
# PAGE 2 — EDA
# ==============================================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Distributions", "Numerical", "Boxplots", "Correlation", "Attack x Severity"
    ])

    with tab1:
        st.subheader("Attack Type & Category Distributions")
        show_figure("01_data_distribution.png")

    with tab2:
        st.subheader("Numerical Feature Distributions")
        show_figure("02_numerical_distributions.png")
        st.markdown("""
        - **Log-Damage** follows a roughly normal distribution after log-transform
        - **Affected Systems** is uniformly distributed (1-999)
        """)

    with tab3:
        st.subheader("Boxplots: Damage & Affected Systems")
        show_figure("03_boxplots.png")

    with tab4:
        st.subheader("Feature Correlation Heatmap")
        show_figure("04_correlation_heatmap.png")
        st.info("Low inter-feature correlation confirms the dataset is synthetic (randomly generated).")

    with tab5:
        st.subheader("Attack Type × Severity Heatmap")
        show_figure("05_attack_severity_heatmap.png")
        st.markdown("Each row shows the severity breakdown (%) within that attack type.")

# ==============================================================================
# PAGE 3 — Model Results
# ==============================================================================
elif page == "Model Results":
    st.title("Model Training & Evaluation")

    # Summary table
    results = {
        'Logistic Regression': {'Accuracy': 0.1050, 'F1 Weighted': 0.0943, 'Precision': 0.1010, 'Recall': 0.1050},
        'Decision Tree':       {'Accuracy': 0.1010, 'F1 Weighted': 0.0924, 'Precision': 0.0978, 'Recall': 0.1010},
        'Random Forest':       {'Accuracy': 0.0860, 'F1 Weighted': 0.0858, 'Precision': 0.0861, 'Recall': 0.0860},
        'RF Tuned':            {'Accuracy': 0.0905, 'F1 Weighted': 0.0908, 'Precision': 0.0916, 'Recall': 0.0905},
        'XGBoost':             {'Accuracy': 0.0805, 'F1 Weighted': 0.0802, 'Precision': 0.0802, 'Recall': 0.0805},
        'Gradient Boosting':   {'Accuracy': 0.0800, 'F1 Weighted': 0.0797, 'Precision': 0.0798, 'Recall': 0.0800},
    }
    summary_df = pd.DataFrame(results).T.sort_values('F1 Weighted', ascending=False)
    st.subheader("Model Comparison Table")
    st.dataframe(summary_df.style.format("{:.4f}").highlight_max(axis=0, color='#d4edda'), use_container_width=True)

    st.info("""
    **Why ~10% accuracy?** This dataset is fully synthetic — `Attack_Type` was randomly assigned
    independently of all other features (date, damage, location, severity).
    With 10 balanced classes, random chance = 10%. The full ML pipeline is correct and demonstrates
    all required workflows (tuning, cross-validation, ROC curves, etc.).
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Confusion Matrix", "ROC Curves", "Feature Importance", "XGBoost History", "Cross-Validation"
    ])

    with tab1:
        show_figure("07_confusion_matrices.png")
    with tab2:
        show_figure("08_roc_curves.png")
    with tab3:
        show_figure("09_feature_importance.png")
    with tab4:
        show_figure("10_xgboost_training_history.png")
    with tab5:
        show_figure("12_cross_validation.png")

# ==============================================================================
# PAGE 4 — Live Prediction
# ==============================================================================
elif page == "Live Prediction":
    st.title("Live Attack Type Prediction")

    if model is None:
        st.error("Model files not found. Run `python main.py` first to train and save the models.")
        st.stop()

    st.markdown("Fill in the attack details below to predict the attack type.")

    col1, col2 = st.columns(2)

    with col1:
        target_industry = st.selectbox("Target Industry", sorted(df['Target_Industry'].unique()))
        location        = st.selectbox("Location",        sorted(df['Location'].unique()))
        severity        = st.selectbox("Severity",        ['Low', 'Medium', 'High', 'Critical'])
        damage          = st.slider("Damage Estimate (USD)", 1000, 5000000, 500000, step=1000)

    with col2:
        affected        = st.slider("Affected Systems",   1, 999, 100)
        month           = st.slider("Month",              1, 12, 6)
        day_of_week     = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                                       format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
        year            = st.selectbox("Year", [2019, 2020, 2021, 2022, 2023, 2024])

    if st.button("Predict Attack Type", type="primary"):
        SEVERITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        severity_num = SEVERITY_MAP[severity]
        log_damage   = np.log1p(damage)
        dmg_per_sys  = damage / (affected + 1)
        log_dps      = np.log1p(dmg_per_sys)
        is_weekend   = int(day_of_week >= 5)
        quarter      = (month - 1) // 3 + 1
        high_damage  = int(damage >= df['Damage_Estimate(USD)'].quantile(0.75))

        industry_risk_score = df[df['Target_Industry'] == target_industry]['Severity'].map(SEVERITY_MAP).mean()
        location_risk_score = df[df['Location'] == location]['Severity'].map(SEVERITY_MAP).mean()
        industry_freq_score = (df['Target_Industry'] == target_industry).sum() / len(df)
        location_freq_score = (df['Location'] == location).sum() / len(df)

        from sklearn.preprocessing import LabelEncoder
        le_ind = LabelEncoder().fit(df['Target_Industry'].astype(str))
        le_loc = LabelEncoder().fit(df['Location'].astype(str))
        le_sev = LabelEncoder().fit(df['Severity'].astype(str))

        numerical_vals = [
            year, month, day_of_week, quarter, is_weekend,
            log_damage, affected, log_dps,
            industry_risk_score, location_risk_score,
            industry_freq_score, location_freq_score,
            high_damage, severity_num
        ]
        cat_vals = [
            le_ind.transform([target_industry])[0],
            le_loc.transform([location])[0],
            le_sev.transform([severity])[0],
        ]
        X_input = np.array(numerical_vals + cat_vals).reshape(1, -1)
        X_scaled = scaler.transform(X_input)

        proba      = model.predict_proba(X_scaled)[0]
        pred_class = le_target.classes_[np.argmax(proba)]
        confidence = np.max(proba) * 100

        st.divider()
        col_res1, col_res2 = st.columns(2)
        col_res1.success(f"**Predicted Attack Type:** {pred_class}")
        col_res2.metric("Confidence", f"{confidence:.1f}%")

        st.subheader("Probability Distribution")
        prob_df = pd.DataFrame({
            'Attack Type': le_target.classes_,
            'Probability': proba
        }).sort_values('Probability', ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        colors  = ['#2ecc71' if c == pred_class else '#3498db' for c in prob_df['Attack Type']]
        ax.barh(prob_df['Attack Type'], prob_df['Probability'], color=colors)
        ax.set_xlabel('Probability')
        ax.set_title('Attack Type Probability Distribution', fontweight='bold')
        ax.axvline(1/10, color='red', linestyle='--', alpha=0.5, label='Random chance (10%)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
