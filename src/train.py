"""
src/train.py
Task 5 – Train models, track with MLflow, register best model
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from data_processing import get_processed_dataset_with_target

# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    """Returns a dict of all required metrics."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1"       : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_proba),
    }

def log_run(model_name, model, params, metrics, X_train, y_train):
    """Fits model and logs everything to MLflow."""
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model",
                                 registered_model_name=model_name)
    return model

# ── Main training script ──────────────────────────────────────────────────────

def train(data_path="../data/raw/data.csv"):
    raw_df = pd.read_csv(data_path)
    
    # Get features + target
    df = get_processed_dataset_with_target(raw_df)
    df = df.dropna(subset=["is_high_risk"])
    
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    mlflow.set_experiment("credit_risk_model")
    
    results = {}

    # ── Model 1: Logistic Regression ─────────────────────────────────
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_metrics = evaluate(lr, X_test, y_test)
    lr_params  = {"model": "LogisticRegression", "max_iter": 1000}
    log_run("LogisticRegression", lr, lr_params, lr_metrics, X_train, y_train)
    results["LogisticRegression"] = lr_metrics
    print("Logistic Regression:", lr_metrics)

    # ── Model 2: Random Forest with RandomizedSearch ──────────────────
    rf_param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth"   : [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    }
    rf_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_dist, n_iter=10, cv=3,
        scoring="roc_auc", random_state=42, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_rf      = rf_search.best_estimator_
    rf_metrics   = evaluate(best_rf, X_test, y_test)
    rf_params    = rf_search.best_params_
    log_run("RandomForest", best_rf, rf_params, rf_metrics, X_train, y_train)
    results["RandomForest"] = rf_metrics
    print("Random Forest:", rf_metrics)

    # ── Model 3: Gradient Boosting ────────────────────────────────────
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_metrics = evaluate(gb, X_test, y_test)
    gb_params  = {"model": "GradientBoosting", "n_estimators": 100}
    log_run("GradientBoosting", gb, gb_params, gb_metrics, X_train, y_train)
    results["GradientBoosting"] = gb_metrics
    print("Gradient Boosting:", gb_metrics)

    # ── Pick best model by ROC-AUC ────────────────────────────────────
    best_name = max(results, key=lambda m: results[m]["roc_auc"])
    print(f"\n✅ Best model: {best_name}  ROC-AUC={results[best_name]['roc_auc']:.4f}")
    return best_name

if __name__ == "__main__":
    train()