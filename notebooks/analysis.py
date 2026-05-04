# %% [markdown]
# # REST API Abuse Detection — Exploratory Analysis & Modelling
#
# **Reference:** IEEE CloudCom 2021 — *"API Security Threat Detection Using Machine Learning"*
#
# **Dataset:** 10,000 cyber-attack records | 10 attack types | 4 severity levels
#
# **Notebook sections:**
# 1. Setup & Data Loading
# 2. Exploratory Data Analysis (EDA)
# 3. Feature Engineering
# 4. Preprocessing
# 5. Baseline Models
# 6. Advanced Models + Hyperparameter Tuning
# 7. Evaluation (confusion matrix, ROC, feature importance)
# 8. Model Comparison & Cross-validation

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 100, 'font.size': 10})

BASE_DIR    = os.path.dirname(os.path.abspath('__file__'))
DATA_PATH   = os.path.join(BASE_DIR, '..', 'data', 'cyber_attacks_dataset.csv')
FIGURES_DIR = os.path.join(BASE_DIR, '..', 'outputs', 'figures')
REPORTS_DIR = os.path.join(BASE_DIR, '..', 'outputs', 'reports')
MODELS_DIR  = os.path.join(BASE_DIR, '..', 'models')

for d in [FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
df.head()

# %%
print("Data Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print(f"\nDuplicates: {df.duplicated().sum()}")

# %%
print("Attack Type Distribution:")
print(df['Attack_Type'].value_counts())
print("\nSeverity Distribution:")
print(df['Severity'].value_counts())
print("\nTop Industries:")
print(df['Target_Industry'].value_counts())

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
# ── Attack type & severity distributions ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

attack_counts = df['Attack_Type'].value_counts()
axes[0].barh(attack_counts.index, attack_counts.values,
             color=plt.cm.Set3(np.linspace(0, 1, len(attack_counts))))
axes[0].set_xlabel('Count')
axes[0].set_title('Attack Type Distribution', fontweight='bold')

sev_counts = df['Severity'].value_counts()
axes[1].pie(sev_counts.values, labels=sev_counts.index, autopct='%1.1f%%',
            colors=['#66b3ff','#ffcc99','#ff9999','#99ff99'],
            startangle=140)
axes[1].set_title('Severity Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_01_attack_severity.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── Numerical feature distributions ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

log_dmg = np.log1p(df['Damage_Estimate(USD)'])
axes[0].hist(log_dmg, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(log_dmg.mean(), color='red', linestyle='--', label=f'Mean={log_dmg.mean():.2f}')
axes[0].set_xlabel('log(1 + Damage USD)'); axes[0].set_title('Log-Damage', fontweight='bold')
axes[0].legend()

axes[1].hist(df['Affected_Systems'], bins=40, color='darkorange', edgecolor='white', alpha=0.85)
axes[1].axvline(df['Affected_Systems'].mean(), color='red', linestyle='--',
                label=f"Mean={df['Affected_Systems'].mean():.1f}")
axes[1].set_xlabel('Affected Systems'); axes[1].set_title('Affected Systems', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_02_numerical.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── Boxplots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

severity_order = ['Low', 'Medium', 'High', 'Critical']
sns.boxplot(data=df, x='Severity', y='Damage_Estimate(USD)',
            order=severity_order, palette='RdYlGn_r', ax=axes[0])
axes[0].set_title('Damage by Severity', fontweight='bold')

sns.boxplot(data=df, x='Attack_Type', y='Affected_Systems', palette='Set2', ax=axes[1])
axes[1].tick_params(axis='x', rotation=40)
axes[1].set_title('Affected Systems by Attack Type', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_03_boxplots.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── Correlation heatmap ────────────────────────────────────────────────────────
num_df = pd.DataFrame({
    'Damage_USD':       df['Damage_Estimate(USD)'],
    'Log_Damage':       np.log1p(df['Damage_Estimate(USD)']),
    'Affected_Systems': df['Affected_Systems'],
    'Damage_Per_System': df['Damage_Estimate(USD)'] / (df['Affected_Systems'] + 1),
    'Severity_Num':     df['Severity'].map({'Low':0,'Medium':1,'High':2,'Critical':3}),
    'Month':            pd.to_datetime(df['Date'], errors='coerce').dt.month,
    'DayOfWeek':        pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek,
})

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_04_correlation.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# ── Attack × Severity heatmap ─────────────────────────────────────────────────
crosstab = pd.crosstab(df['Attack_Type'], df['Severity'], normalize='index') * 100
crosstab = crosstab.reindex(columns=severity_order)

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label':'% of row'})
ax.set_title('Attack Type × Severity (%)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_05_attack_severity_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Feature Engineering

# %%
from src.feature_engineering import engineer_features, get_feature_columns

df_eng = engineer_features(df)
print(f"Shape after engineering: {df_eng.shape}")
print(f"New columns: {[c for c in df_eng.columns if c not in df.columns]}")
df_eng[['Log_Damage', 'Severity_Numeric', 'Is_API_Attack',
        'Industry_Risk_Score', 'Location_Risk_Score']].describe()

# %%
# API attack breakdown
api_counts  = df_eng['Is_API_Attack'].value_counts()
labels_api  = ['Non-API Attack', 'API Attack']
fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(api_counts.values, labels=labels_api, autopct='%1.1f%%',
       colors=['#ff9999','#66b3ff'], startangle=90)
ax.set_title('API vs Non-API Attack Distribution', fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Preprocessing

# %%
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

categorical_cols, numerical_cols = get_feature_columns()

cat_encoders = {}
df_proc = df_eng.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_proc[col + '_enc'] = le.fit_transform(df_proc[col].astype(str))
    cat_encoders[col] = le

enc_cat_cols = [c + '_enc' for c in categorical_cols]
feature_cols = numerical_cols + enc_cat_cols

le_target = LabelEncoder()
y = le_target.fit_transform(df_proc['Attack_Type'].astype(str))
class_names = list(le_target.classes_)

X = df_proc[feature_cols].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Classes: {class_names}")
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# %%
# Class distribution plot
unique, counts = np.unique(y_train, return_counts=True)
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar([class_names[i] for i in unique], counts,
              color=plt.cm.tab10(np.linspace(0, 1, len(unique))), edgecolor='white')
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(cnt), ha='center', fontsize=9)
ax.set_ylabel('Count')
ax.set_title('Training Set Class Distribution', fontweight='bold')
ax.tick_params(axis='x', rotation=35)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_06_class_dist.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Baseline Models

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

baseline_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42),
}

baseline_results = {}
for name, model in baseline_models.items():
    X_fit, X_ev = (X_train_sc, X_test_sc) if name == 'Logistic Regression' else (X_train, X_test)
    model.fit(X_fit, y_train)
    y_pred = model.predict(X_ev)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    baseline_results[name] = {'accuracy': acc, 'f1': f1, 'model': model, 'y_pred': y_pred}
    print(f"{name}: Accuracy={acc:.4f}, F1={f1:.4f}")

# %%
print("\nLogistic Regression Report:")
print(classification_report(y_test, baseline_results['Logistic Regression']['y_pred'],
                             target_names=class_names, zero_division=0))

# %% [markdown]
# ## 6. Advanced Models + Hyperparameter Tuning

# %%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
import xgboost as xgb

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5,
                             random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_sc, y_train)
y_pred_rf = rf.predict(X_test_sc)
print(f"Random Forest: Acc={accuracy_score(y_test, y_pred_rf):.4f}, "
      f"F1={f1_score(y_test, y_pred_rf, average='weighted', zero_division=0):.4f}")

# %%
# GridSearchCV tuning for RF
param_grid = {
    'n_estimators':    [50, 100, 200],
    'max_depth':       [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
}
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    param_grid, cv=cv5, scoring='f1_weighted', n_jobs=-1, verbose=1
)
gs_rf.fit(X_train_sc, y_train)
best_rf = gs_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_sc)
print(f"\nBest RF params : {gs_rf.best_params_}")
print(f"Best RF CV F1  : {gs_rf.best_score_:.4f}")
print(f"RF Tuned Test  : Acc={accuracy_score(y_test, y_pred_best_rf):.4f}, "
      f"F1={f1_score(y_test, y_pred_best_rf, average='weighted', zero_division=0):.4f}")

# %%
# XGBoost with training history
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)
evals_result  = xgb_model.evals_result()
y_pred_xgb    = xgb_model.predict(X_test)
print(f"XGBoost: Acc={accuracy_score(y_test, y_pred_xgb):.4f}, "
      f"F1={f1_score(y_test, y_pred_xgb, average='weighted', zero_division=0):.4f}")

# %%
# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print(f"Gradient Boosting: Acc={accuracy_score(y_test, y_pred_gb):.4f}, "
      f"F1={f1_score(y_test, y_pred_gb, average='weighted', zero_division=0):.4f}")

# %% [markdown]
# ## 7. Evaluation

# %%
# Classification reports
for name, y_pred in [('RF Tuned', y_pred_best_rf), ('XGBoost', y_pred_xgb)]:
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# %%
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, (name, y_pred) in zip(axes, [('RF Tuned', y_pred_best_rf), ('XGBoost', y_pred_xgb)]):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax, cbar=False)
    ax.set_title(f'{name} — Confusion Matrix', fontweight='bold')
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    ax.tick_params(axis='x', rotation=40)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_07_confusion.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# ROC curves (multi-class OvR)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

n_classes = len(class_names)
y_bin     = label_binarize(y_test, classes=np.arange(n_classes))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, (name, model, X_ev) in zip(axes, [
        ('RF Tuned', best_rf,    X_test_sc),
        ('XGBoost',  xgb_model,  X_test)]):
    y_score = model.predict_proba(X_ev)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=1.5, label=f"{class_names[i]} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} — OvR ROC', fontweight='bold')
    ax.legend(loc='lower right', fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_08_roc.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# Feature importance
importances = best_rf.feature_importances_
idx         = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(feature_cols)))
ax.barh([feature_cols[i] for i in idx[::-1]],
        importances[idx[::-1]], color=colors[::-1])
ax.axvline(importances.mean(), color='red', linestyle='--', label='Mean')
ax.set_xlabel('Importance')
ax.set_title('RF Tuned — Feature Importance', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_09_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# XGBoost training history
train_loss = evals_result['validation_0']['mlogloss']
val_loss   = evals_result['validation_1']['mlogloss']

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, len(train_loss)+1), train_loss, label='Train', color='royalblue', lw=2)
ax.plot(range(1, len(val_loss)+1),   val_loss,   label='Validation', color='tomato', lw=2)
ax.axvline(np.argmin(val_loss)+1, color='green', linestyle='--', alpha=0.7,
           label=f'Min val @ {np.argmin(val_loss)+1}')
ax.set_xlabel('Boosting Round'); ax.set_ylabel('Log-Loss')
ax.set_title('XGBoost Training History', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_10_xgb_history.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Model Comparison & Cross-Validation

# %%
from sklearn.metrics import precision_score, recall_score

all_models = {
    'Logistic Regression': (baseline_results['Logistic Regression']['model'], X_test_sc),
    'Decision Tree':       (baseline_results['Decision Tree']['model'],       X_test),
    'Random Forest':       (rf,         X_test_sc),
    'RF Tuned':            (best_rf,    X_test_sc),
    'XGBoost':             (xgb_model,  X_test),
    'Gradient Boosting':   (gb,         X_test),
}

summary = {}
for name, (model, X_ev) in all_models.items():
    y_pred = model.predict(X_ev)
    summary[name] = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
    }

summary_df = pd.DataFrame(summary).T.sort_values('F1', ascending=False)
print(summary_df.to_string(float_format='{:.4f}'.format))

# %%
# Grouped bar chart
x     = np.arange(len(summary_df))
width = 0.2
fig, ax = plt.subplots(figsize=(14, 6))
for i, col in enumerate(['Accuracy', 'F1', 'Precision', 'Recall']):
    ax.bar(x + i*width, summary_df[col], width, label=col)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(summary_df.index, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Score'); ax.set_title('Model Comparison', fontweight='bold')
ax.legend(); ax.set_ylim(0, 1.1); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_11_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()

# %%
# Cross-validation
cv_models_cv = {
    'LR':  (baseline_results['Logistic Regression']['model'], X_train_sc),
    'DT':  (baseline_results['Decision Tree']['model'],       X_train),
    'RF':  (rf,       X_train_sc),
    'RF+': (best_rf,  X_train_sc),
    'XGB': (xgb_model, X_train),
    'GB':  (gb,        X_train),
}
cv_results = {}
print("\n5-Fold Cross-Validation (F1 weighted):")
for name, (model, X_cv) in cv_models_cv.items():
    scores = cross_val_score(model, X_cv, y_train, cv=cv5, scoring='f1_weighted', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")

fig, ax = plt.subplots(figsize=(11, 5))
bp = ax.boxplot(list(cv_results.values()), patch_artist=True, labels=list(cv_results.keys()))
bp_colors = plt.cm.Set2(np.linspace(0, 1, len(cv_results)))
for patch, color in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(color)
ax.set_ylabel('F1 Weighted')
ax.set_title('5-Fold Cross-Validation Boxplot', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'nb_12_cv.png'), dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Save Models

# %%
import joblib

joblib.dump(best_rf,      os.path.join(MODELS_DIR, 'best_model_rf.joblib'))
joblib.dump(xgb_model,   os.path.join(MODELS_DIR, 'xgboost_model.joblib'))
joblib.dump(scaler,      os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(le_target,   os.path.join(MODELS_DIR, 'label_encoder_target.joblib'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_names.joblib'))
print("All models saved to models/")
print(f"\nBest model: {summary_df.index[0]} (F1={summary_df['F1'].iloc[0]:.4f})")
