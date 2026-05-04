import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb


def get_baseline_models() -> dict:
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, C=1.0, solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, min_samples_split=5, random_state=42
        ),
    }


def get_advanced_models() -> dict:
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1, class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }


def tune_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"Best RF params: {gs.best_params_}")
    print(f"Best RF CV F1: {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def train_xgboost_with_eval(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray
) -> tuple:
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    evals_result = model.evals_result()
    return model, evals_result
