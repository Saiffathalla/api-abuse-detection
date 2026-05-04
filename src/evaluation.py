import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold


def print_classification_report(y_true, y_pred, class_names, model_name: str, output_dir: str) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(f"\n{'='*60}\n{model_name} — Classification Report\n{'='*60}")
    print(report)
    with open(f"{output_dir}/{model_name.replace(' ', '_')}_report.txt", 'w') as f:
        f.write(f"{model_name} — Classification Report\n{'='*60}\n{report}")


def plot_confusion_matrix(y_true, y_pred, class_names, title: str, ax) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, cbar=False)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)


def plot_roc_curves(models_preds: dict, y_test, class_names, figures_dir: str) -> None:
    n_classes = len(class_names)
    classes = np.arange(n_classes)
    y_bin = label_binarize(y_test, classes=classes)

    fig, axes = plt.subplots(1, len(models_preds), figsize=(8 * len(models_preds), 6))
    if len(models_preds) == 1:
        axes = [axes]

    for ax, (model_name, (model, X_test)) in zip(axes, models_preds.items()):
        y_score = model.predict_proba(X_test)
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"{cls} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f"{model_name} — OvR ROC Curves", fontweight='bold')
        ax.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/08_roc_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_roc_curves.png")


def plot_feature_importance(model, feature_names: list, figures_dir: str) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_names)))
    ax.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest — Feature Importance', fontweight='bold')
    ax.axvline(x=np.mean(sorted_vals), color='red', linestyle='--', alpha=0.7, label='Mean importance')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/09_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 09_feature_importance.png")


def plot_xgboost_history(evals_result: dict, figures_dir: str) -> None:
    train_loss = evals_result['validation_0']['mlogloss']
    val_loss = evals_result['validation_1']['mlogloss']
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label='Train Log-Loss', color='royalblue')
    ax.plot(epochs, val_loss, label='Validation Log-Loss', color='tomato')
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('Log-Loss')
    ax.set_title('XGBoost Training History', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/10_xgboost_training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 10_xgboost_training_history.png")


def plot_model_comparison(results: dict, figures_dir: str) -> None:
    model_names = list(results.keys())
    metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    metric_labels = ['Accuracy', 'F1 Weighted', 'Precision', 'Recall']
    x = np.arange(len(model_names))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Grouped bar chart
    ax = axes[0]
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        vals = [results[m].get(metric, 0) for m in model_names]
        ax.bar(x + i * width, vals, width, label=label)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Radar / spider chart
    ax2 = plt.subplot(1, 2, 2, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors_cycle = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
    for m_name, color in zip(model_names, colors_cycle):
        vals = [results[m_name].get(metric, 0) for metric in metrics]
        vals += vals[:1]
        ax2.plot(angles, vals, 'o-', linewidth=1.5, color=color, label=m_name)
        ax2.fill(angles, vals, alpha=0.08, color=color)
    ax2.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_title('Radar Chart', fontweight='bold', pad=15)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/11_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 11_model_comparison.png")


def plot_cross_validation(cv_results: dict, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [scores for scores in cv_results.values()]
    labels = list(cv_results.keys())
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('F1 Score (Weighted)')
    ax.set_title('5-Fold Cross-Validation Results', fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/12_cross_validation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 12_cross_validation.png")


def run_cross_validation(models: dict, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
        cv_results[name] = scores
        print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
    return cv_results
