# -*- coding: utf-8 -*-
"""
REST API Abuse Detection Using Request Pattern Analysis
=======================================================
End-to-end ML pipeline: EDA -> Feature Engineering ->
Preprocessing -> Training -> Tuning -> Evaluation -> Saving.

Reference: IEEE CloudCom 2021 - "API Security Threat Detection Using ML"
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 100, 'font.size': 10})

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'cyber_attacks_dataset.csv')
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

for d in [FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# SECTION 1 -Data Loading & Inspection
# ==============================================================================
print("\n" + "="*60)
print("SECTION 1: DATA LOADING & INSPECTION")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns        : {list(df.columns)}")
print(f"Missing values :\n{df.isnull().sum()}")
print(f"Duplicate rows : {df.duplicated().sum()}")
print(f"\nNumerical summary:\n{df.describe()}")
print(f"\nAttack type counts:\n{df['Attack_Type'].value_counts()}")
print(f"\nSeverity counts:\n{df['Severity'].value_counts()}")

# ==============================================================================
# SECTION 2 -Exploratory Data Analysis
# ==============================================================================
print("\n" + "="*60)
print("SECTION 2: EDA")
print("="*60)

# ── Figure 01: categorical distributions ──────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Attack type
ax0 = fig.add_subplot(gs[0, 0])
attack_counts = df['Attack_Type'].value_counts()
bars = ax0.barh(attack_counts.index, attack_counts.values,
                color=plt.cm.Set3(np.linspace(0, 1, len(attack_counts))))
ax0.set_xlabel('Count')
ax0.set_title('Attack Type Distribution', fontweight='bold')
for bar, val in zip(bars, attack_counts.values):
    ax0.text(val + 20, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=8)

# Severity pie
ax1 = fig.add_subplot(gs[0, 1])
sev_counts = df['Severity'].value_counts()
wedge_colors = ['#66b3ff', '#ffcc99', '#ff9999', '#99ff99']
ax1.pie(sev_counts.values, labels=sev_counts.index, autopct='%1.1f%%',
        colors=wedge_colors, startangle=140, textprops={'fontsize': 10})
ax1.set_title('Severity Distribution', fontweight='bold')

# Target industry
ax2 = fig.add_subplot(gs[1, 0])
ind_counts = df['Target_Industry'].value_counts()
sns.barplot(y=ind_counts.index, x=ind_counts.values, ax=ax2,
            palette='husl', orient='h')
ax2.set_xlabel('Count')
ax2.set_title('Target Industry Distribution', fontweight='bold')

# Location
ax3 = fig.add_subplot(gs[1, 1])
loc_counts = df['Location'].value_counts().head(15)
sns.barplot(y=loc_counts.index, x=loc_counts.values, ax=ax3,
            palette='coolwarm', orient='h')
ax3.set_xlabel('Count')
ax3.set_title('Top Locations', fontweight='bold')

plt.suptitle('Data Distribution Overview', fontsize=14, fontweight='bold', y=1.01)
plt.savefig(os.path.join(FIGURES_DIR, '01_data_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_data_distribution.png")

# ── Figure 02: numerical distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

log_damage = np.log1p(df['Damage_Estimate(USD)'])
axes[0].hist(log_damage, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(log_damage.mean(), color='red', linestyle='--', label=f'Mean={log_damage.mean():.2f}')
axes[0].axvline(log_damage.median(), color='orange', linestyle='--', label=f'Median={log_damage.median():.2f}')
axes[0].set_xlabel('log(1 + Damage Estimate USD)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Log-Damage Distribution', fontweight='bold')
axes[0].legend()

axes[1].hist(df['Affected_Systems'], bins=40, color='darkorange', edgecolor='white', alpha=0.85)
axes[1].axvline(df['Affected_Systems'].mean(), color='red', linestyle='--',
                label=f"Mean={df['Affected_Systems'].mean():.1f}")
axes[1].set_xlabel('Affected Systems')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Affected Systems Distribution', fontweight='bold')
axes[1].legend()

plt.suptitle('Numerical Feature Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '02_numerical_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_numerical_distributions.png")

# ── Figure 03: boxplots ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

severity_order = ['Low', 'Medium', 'High', 'Critical']
sns.boxplot(data=df, x='Severity', y='Damage_Estimate(USD)', order=severity_order,
            palette='RdYlGn_r', ax=axes[0])
axes[0].set_title('Damage Estimate by Severity', fontweight='bold')
axes[0].set_ylabel('Damage Estimate (USD)')
axes[0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df, x='Attack_Type', y='Affected_Systems',
            palette='Set2', ax=axes[1])
axes[1].set_title('Affected Systems by Attack Type', fontweight='bold')
axes[1].set_ylabel('Affected Systems')
axes[1].tick_params(axis='x', rotation=40)

plt.suptitle('Boxplot Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '03_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_boxplots.png")

# ── Figure 04: correlation heatmap (numerical columns only) ───────────────────
num_df = df[['Damage_Estimate(USD)', 'Affected_Systems']].copy()
num_df['Log_Damage'] = np.log1p(num_df['Damage_Estimate(USD)'])
num_df['Damage_Per_System'] = num_df['Damage_Estimate(USD)'] / (num_df['Affected_Systems'] + 1)
severity_map_num = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
num_df['Severity_Num'] = df['Severity'].map(severity_map_num)
num_df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month
num_df['DayOfWeek'] = pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek

corr = num_df.corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '04_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_correlation_heatmap.png")

# ── Figure 05: attack type × severity heatmap ─────────────────────────────────
crosstab = pd.crosstab(df['Attack_Type'], df['Severity'], normalize='index') * 100
crosstab = crosstab.reindex(columns=severity_order)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': '% of row'})
ax.set_title('Attack Type × Severity Distribution (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Attack Type')
ax.set_xlabel('Severity')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '05_attack_severity_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05_attack_severity_heatmap.png")

# ==============================================================================
# SECTION 3 -Feature Engineering
# ==============================================================================
print("\n" + "="*60)
print("SECTION 3: FEATURE ENGINEERING")
print("="*60)

df_eng = df.copy()

# Temporal
df_eng['Date']      = pd.to_datetime(df_eng['Date'], errors='coerce')
df_eng['Year']      = df_eng['Date'].dt.year
df_eng['Month']     = df_eng['Date'].dt.month
df_eng['DayOfWeek'] = df_eng['Date'].dt.dayofweek
df_eng['Quarter']   = df_eng['Date'].dt.quarter
df_eng['IsWeekend'] = (df_eng['DayOfWeek'] >= 5).astype(int)

# Damage
df_eng['Log_Damage']          = np.log1p(df_eng['Damage_Estimate(USD)'])
df_eng['Damage_Per_System']   = df_eng['Damage_Estimate(USD)'] / (df_eng['Affected_Systems'] + 1)
df_eng['Log_Damage_Per_System'] = np.log1p(df_eng['Damage_Per_System'])
high_thr = df_eng['Damage_Estimate(USD)'].quantile(0.75)
df_eng['High_Damage'] = (df_eng['Damage_Estimate(USD)'] >= high_thr).astype(int)

# Severity numeric
SEVERITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
df_eng['Severity_Numeric'] = df_eng['Severity'].map(SEVERITY_MAP).fillna(1)

# Binary API attack label
API_ATTACKS = ['SQL Injection', 'DDoS', 'Brute Force Attack',
               'Cross-site Scripting (XSS)', 'Privilege Escalation', 'Man-in-the-Middle']
df_eng['Is_API_Attack'] = df_eng['Attack_Type'].isin(API_ATTACKS).astype(int)

# Risk / frequency scores
df_eng['Industry_Risk_Score']       = df_eng['Target_Industry'].map(df_eng.groupby('Target_Industry')['Severity_Numeric'].mean())
df_eng['Location_Risk_Score']       = df_eng['Location'].map(df_eng.groupby('Location')['Severity_Numeric'].mean())
df_eng['Location_Attack_Frequency'] = df_eng['Location'].map(df_eng.groupby('Location').size() / len(df_eng))
df_eng['Industry_Attack_Frequency'] = df_eng['Target_Industry'].map(df_eng.groupby('Target_Industry').size() / len(df_eng))

print(f"Engineered features added: {df_eng.shape[1] - df.shape[1]}")
print(f"New feature list: Year, Month, DayOfWeek, Quarter, IsWeekend, "
      f"Log_Damage, Damage_Per_System, Log_Damage_Per_System, High_Damage, "
      f"Severity_Numeric, Is_API_Attack, Industry_Risk_Score, "
      f"Location_Risk_Score, Location_Attack_Frequency, Industry_Attack_Frequency")

# ==============================================================================
# SECTION 4 -Preprocessing
# ==============================================================================
print("\n" + "="*60)
print("SECTION 4: PREPROCESSING")
print("="*60)

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

categorical_cols = ['Target_Industry', 'Location', 'Severity']
numerical_cols   = [
    'Year', 'Month', 'DayOfWeek', 'Quarter', 'IsWeekend',
    'Log_Damage', 'Affected_Systems', 'Log_Damage_Per_System',
    'Industry_Risk_Score', 'Location_Risk_Score',
    'Industry_Attack_Frequency', 'Location_Attack_Frequency',
    'High_Damage', 'Severity_Numeric'
]

# Encode categoricals
cat_encoders = {}
df_proc = df_eng.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_proc[col + '_enc'] = le.fit_transform(df_proc[col].astype(str))
    cat_encoders[col] = le

enc_cat_cols = [c + '_enc' for c in categorical_cols]
feature_cols = numerical_cols + enc_cat_cols

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(df_proc['Attack_Type'].astype(str))
class_names = list(le_target.classes_)
print(f"Classes ({len(class_names)}): {class_names}")

X = df_proc[feature_cols].values
print(f"Feature matrix: {X.shape}")

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Figure 06: class distribution ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
unique, counts = np.unique(y_train, return_counts=True)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
bars = ax.bar([class_names[i] for i in unique], counts, color=colors, edgecolor='white')
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(cnt), ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Count')
ax.set_title('Training Set Class Distribution', fontweight='bold')
ax.tick_params(axis='x', rotation=35)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '06_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 06_class_distribution.png")

# ==============================================================================
# SECTION 5 -Model Training
# ==============================================================================
print("\n" + "="*60)
print("SECTION 5: MODEL TRAINING")
print("="*60)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, C=1.0, solver='lbfgs'
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10, min_samples_split=5, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5,
        random_state=42, n_jobs=-1, class_weight='balanced'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
}

trained_models = {}
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ['Logistic Regression', 'Random Forest']:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    trained_models[name] = model
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    pre = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    results[name] = {'accuracy': acc, 'f1_weighted': f1,
                     'precision_weighted': pre, 'recall_weighted': rec}
    print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  Precision={pre:.4f}  Recall={rec:.4f}")

# XGBoost with eval_set for training history
print("\nTraining XGBoost (with eval_set)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', random_state=42, n_jobs=-1
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
evals_result = xgb_model.evals_result()
y_pred_xgb = xgb_model.predict(X_test)
acc  = accuracy_score(y_test, y_pred_xgb)
f1   = f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
pre  = precision_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred_xgb, average='weighted', zero_division=0)
results['XGBoost'] = {'accuracy': acc, 'f1_weighted': f1,
                      'precision_weighted': pre, 'recall_weighted': rec}
trained_models['XGBoost'] = xgb_model
print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  Precision={pre:.4f}  Recall={rec:.4f}")

# ==============================================================================
# SECTION 6 -Hyperparameter Tuning (GridSearchCV on Random Forest)
# ==============================================================================
print("\n" + "="*60)
print("SECTION 6: HYPERPARAMETER TUNING")
print("="*60)

from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'n_estimators':    [50, 100, 200],
    'max_depth':       [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
}
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    param_grid_rf, cv=cv5, scoring='f1_weighted', n_jobs=-1, verbose=1
)
print("Running GridSearchCV for Random Forest...")
gs_rf.fit(X_train_sc, y_train)
best_rf = gs_rf.best_estimator_
print(f"Best params : {gs_rf.best_params_}")
print(f"Best CV F1  : {gs_rf.best_score_:.4f}")

y_pred_best_rf = best_rf.predict(X_test_sc)
acc  = accuracy_score(y_test, y_pred_best_rf)
f1   = f1_score(y_test, y_pred_best_rf, average='weighted', zero_division=0)
pre  = precision_score(y_test, y_pred_best_rf, average='weighted', zero_division=0)
rec  = recall_score(y_test, y_pred_best_rf, average='weighted', zero_division=0)
results['RF Tuned'] = {'accuracy': acc, 'f1_weighted': f1,
                       'precision_weighted': pre, 'recall_weighted': rec}
trained_models['RF Tuned'] = best_rf
print(f"RF Tuned -> Accuracy={acc:.4f}  F1={f1:.4f}")

# ==============================================================================
# SECTION 7 -Evaluation & Visualizations
# ==============================================================================
print("\n" + "="*60)
print("SECTION 7: EVALUATION")
print("="*60)

# ── Classification reports ─────────────────────────────────────────────────────
report_models = {
    'RF_Tuned':  (best_rf,    X_test_sc,  y_pred_best_rf),
    'XGBoost':   (xgb_model,  X_test,     y_pred_xgb),
}
for tag, (model, _, y_pred) in report_models.items():
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(f"\n{tag}:\n{report}")
    with open(os.path.join(REPORTS_DIR, f'{tag}_classification_report.txt'), 'w') as f:
        f.write(f"{tag} -Classification Report\n{'='*60}\n{report}")
    print(f"Saved: outputs/reports/{tag}_classification_report.txt")

# ── Figure 07: confusion matrices ─────────────────────────────────────────────
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, (tag, (_, _, y_pred)) in zip(axes, report_models.items()):
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax, cbar=False)
    ax.set_title(f'{tag} -Confusion Matrix (normalised)', fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.tick_params(axis='x', rotation=40)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '07_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_confusion_matrices.png")

# ── Figure 08: ROC curves (multi-class OvR) ───────────────────────────────────
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

n_classes = len(class_names)
y_bin     = label_binarize(y_test, classes=np.arange(n_classes))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
roc_models = [
    ('RF Tuned', best_rf, X_test_sc),
    ('XGBoost',  xgb_model, X_test),
]
for ax, (name, model, X_ev) in zip(axes, roc_models):
    y_score = model.predict_proba(X_ev)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f"{name} -OvR ROC Curves", fontweight='bold')
    ax.legend(loc='lower right', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '08_roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 08_roc_curves.png")

# ── Figure 09: feature importance ─────────────────────────────────────────────
importances = best_rf.feature_importances_
idx         = np.argsort(importances)[::-1]
sorted_names = [feature_cols[i] for i in idx]
sorted_vals  = importances[idx]

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(sorted_names)))
ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
ax.axvline(np.mean(sorted_vals), color='red', linestyle='--', alpha=0.7, label='Mean importance')
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest -Feature Importance (Tuned)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '09_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 09_feature_importance.png")

# ── Figure 10: XGBoost training history ───────────────────────────────────────
train_loss = evals_result['validation_0']['mlogloss']
val_loss   = evals_result['validation_1']['mlogloss']
epochs     = range(1, len(train_loss) + 1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, train_loss, label='Train Log-Loss',      color='royalblue', lw=2)
ax.plot(epochs, val_loss,   label='Validation Log-Loss', color='tomato',    lw=2)
min_epoch = np.argmin(val_loss) + 1
ax.axvline(min_epoch, color='green', linestyle='--', alpha=0.7,
           label=f'Min val loss @ epoch {min_epoch}')
ax.set_xlabel('Boosting Round')
ax.set_ylabel('Log-Loss')
ax.set_title('XGBoost Training History', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '10_xgboost_training_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 10_xgboost_training_history.png")

# ── Figure 11: model comparison ───────────────────────────────────────────────
all_models_compare = {k: v for k, v in results.items()}
model_names_cmp    = list(all_models_compare.keys())
metrics_cmp        = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
metric_labels_cmp  = ['Accuracy', 'F1 Weighted', 'Precision', 'Recall']
x = np.arange(len(model_names_cmp))
width = 0.18

fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(1, 2, 1)
for i, (metric, label) in enumerate(zip(metrics_cmp, metric_labels_cmp)):
    vals = [all_models_compare[m].get(metric, 0) for m in model_names_cmp]
    ax1.bar(x + i * width, vals, width, label=label)
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(model_names_cmp, rotation=25, ha='right', fontsize=9)
ax1.set_ylabel('Score')
ax1.set_title('Model Comparison -Bar Chart', fontweight='bold')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3)

ax2 = fig.add_subplot(1, 2, 2, polar=True)
angles = np.linspace(0, 2 * np.pi, len(metrics_cmp), endpoint=False).tolist()
angles += angles[:1]
colors_cycle = plt.cm.Set2(np.linspace(0, 1, len(model_names_cmp)))
for m_name, color in zip(model_names_cmp, colors_cycle):
    vals = [all_models_compare[m_name].get(metric, 0) for metric in metrics_cmp]
    vals += vals[:1]
    ax2.plot(angles, vals, 'o-', lw=1.8, color=color, label=m_name)
    ax2.fill(angles, vals, alpha=0.07, color=color)
ax2.set_thetagrids(np.degrees(angles[:-1]), metric_labels_cmp, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_title('Radar Chart', fontweight='bold', pad=15)
ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '11_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 11_model_comparison.png")

# ── Figure 12: cross-validation ───────────────────────────────────────────────
print("\nRunning 5-fold cross-validation...")
cv_models = {
    'Logistic Regression': (trained_models['Logistic Regression'], X_train_sc),
    'Decision Tree':       (trained_models['Decision Tree'],       X_train),
    'Random Forest':       (trained_models['Random Forest'],       X_train_sc),
    'RF Tuned':            (best_rf,                               X_train_sc),
    'XGBoost':             (xgb_model,                             X_train),
    'Gradient Boosting':   (trained_models['Gradient Boosting'],  X_train),
}
cv_results = {}
for name, (model, X_cv) in cv_models.items():
    scores = cross_val_score(model, X_cv, y_train, cv=cv5, scoring='f1_weighted', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(12, 5))
data   = list(cv_results.values())
labels = list(cv_results.keys())
bp = ax.boxplot(data, patch_artist=True, labels=labels)
bp_colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
for patch, color in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(color)
ax.set_ylabel('F1 Score (Weighted)')
ax.set_title('5-Fold Cross-Validation Results', fontweight='bold')
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '12_cross_validation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 12_cross_validation.png")

# ==============================================================================
# SECTION 8 -Save Models & Summary
# ==============================================================================
print("\n" + "="*60)
print("SECTION 8: SAVING MODELS")
print("="*60)

joblib.dump(best_rf,    os.path.join(MODELS_DIR, 'best_model_rf.joblib'))
joblib.dump(xgb_model,  os.path.join(MODELS_DIR, 'xgboost_model.joblib'))
joblib.dump(scaler,     os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(le_target,  os.path.join(MODELS_DIR, 'label_encoder_target.joblib'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_names.joblib'))
print("Saved: best_model_rf.joblib, xgboost_model.joblib, scaler.joblib, "
      "label_encoder_target.joblib, feature_names.joblib")

# ── Final summary table ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL MODEL COMPARISON TABLE")
print("="*60)
summary_df = pd.DataFrame(results).T[['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']]
summary_df.columns = ['Accuracy', 'F1 Weighted', 'Precision', 'Recall']
summary_df = summary_df.sort_values('F1 Weighted', ascending=False)
print(summary_df.to_string(float_format='{:.4f}'.format))

best_model_name = summary_df.index[0]
print(f"\n>>> Best model: {best_model_name} (F1 = {summary_df.loc[best_model_name, 'F1 Weighted']:.4f})")
print("\nAll outputs saved to:")
print(f"  Figures : {FIGURES_DIR}")
print(f"  Reports : {REPORTS_DIR}")
print(f"  Models  : {MODELS_DIR}")
print("\nPipeline complete.")
