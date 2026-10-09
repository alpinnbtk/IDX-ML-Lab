import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """
    Objective function Optuna menggunakan TimeSeriesSplit untuk mengevaluasi hiperparameter.
    """
    # 1. Definisi Search Space Hiperparameter
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'random_state': 42,
        'eval_metric': 'logloss'
    }

    # 2. Time-Series Cross Validation (mencegah data leakage)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        # Menggunakan ROC-AUC sebagai metrik evaluasi tuning
        score = roc_auc_score(y_val, preds)
        scores.append(score)

    return np.mean(scores)


def tune_xgboost(X: pd.DataFrame, y: pd.Series, n_trials: int = 30) -> tuple[XGBClassifier, dict]:
    """
    Jalankan Optuna study dan kembalikan model terbaik yang sudah di-fit pada seluruh data latih.
    """
    logger.info(a := f"Memulai hyperparameter tuning dengan {n_trials} trials...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    print(f"=== Tuning Selesai ===")
    print(f"ROC-AUC Terbaik (CV): {best_score:.4f}")
    print(f"Parameter Terbaik: {best_params}")

    # Train ulang model terbaik menggunakan seluruh dataset (X, y)
    best_model = XGBClassifier(**best_params, random_state=42)
    best_model.fit(X, y)

    return best_model, best_params