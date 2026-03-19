"""
Machine Learning Models Module
================================
Classification (BBBP) and Regression (ESOL) pipelines with:
 - Stratified train/val/test splits
 - Cross-validation + RandomizedSearchCV hyperparameter tuning
 - MLflow experiment tracking
 - Evaluation plots (ROC, confusion matrix, calibration)
 - Model comparison CSV
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ML imports
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    StratifiedKFold, KFold,
    RandomizedSearchCV, train_test_split,
)
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
DPI = 150
MLFLOW_EXPERIMENT = "drug-discovery-bbbp"

DESCRIPTOR_COLS = [
    "MolWt", "MolLogP", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "FractionCSP3",
]
FP_PREFIX = "fp_"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    [SAVED] {path}")


def _feature_cols(df: pd.DataFrame) -> List[str]:
    """Return all fp_* + descriptor columns."""
    fp_cols = [c for c in df.columns if c.startswith(FP_PREFIX)]
    desc_cols = [c for c in DESCRIPTOR_COLS if c in df.columns]
    return fp_cols + desc_cols


def _prepare_classification_data(features_path: str) -> Tuple:
    """
    Load BBBP features and split into train/val/test sets (70/15/15).

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    df = pd.read_csv(features_path)
    feature_names = _feature_cols(df)
    X = df[feature_names].values.astype(np.float32)
    y = df["p_np"].values.astype(int)

    # 70 / 30 first split, then 30 → 15 val + 15 test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"  [SPLIT] train={len(y_train)} | val={len(y_val)} | test={len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def _prepare_regression_data(features_path: str) -> Tuple:
    """
    Load ESOL features and split into train/val/test (70/15/15).
    """
    df = pd.read_csv(features_path)
    feature_names = _feature_cols(df)
    X = df[feature_names].values.astype(np.float32)
    y = df["logS"].values.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    print(f"  [SPLIT] train={len(y_train)} | val={len(y_val)} | test={len(y_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


# ── Classification Evaluation Plots ───────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   model_name: str, out_dir: str) -> float:
    """Plot ROC curve; return AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#3498db",
            label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"roc_{model_name.replace(' ', '_').lower()}.png"))
    return auc


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str, out_dir: str) -> None:
    """Plot normalised + raw confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in [
        (axes[0], cm,      "d",    "Counts"),
        (axes[1], cm_norm, ".2f",  "Normalised"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["Non-perm.", "Permeable"],
                    yticklabels=["Non-perm.", "Permeable"],
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.set_title(f"Confusion Matrix ({title}) — {model_name}", fontsize=11)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"cm_{model_name.replace(' ', '_').lower()}.png"))


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                            model_name: str, out_dir: str) -> None:
    """Plot probability calibration curve."""
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=10
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_predicted, fraction_of_positives, "s-",
            color="#e67e22", label=model_name, lw=2)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1.5)
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"Calibration Curve — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"calibration_{model_name.replace(' ', '_').lower()}.png"))


def plot_all_roc_curves(results: List[Dict], out_dir: str) -> None:
    """Overlay ROC curves for all models on one figure."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    for i, r in enumerate(results):
        ax.plot(r["fpr"], r["tpr"], lw=2, color=colors[i % len(colors)],
                label=f"{r['model']} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models (BBBP Test Set)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "roc_all_models.png"))


# ── Classification Models ──────────────────────────────────────────────────────

def get_classification_models() -> Dict[str, Tuple[Any, Dict]]:
    """
    Return (estimator, param_grid) pairs for all classification models.
    All models are wrapped in sklearn Pipelines.
    """
    models = {}

    # 1. Logistic Regression (scaled)
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    lr_params = {
        "clf__C":       [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver":  ["lbfgs"],
    }
    models["Logistic Regression"] = (lr_pipe, lr_params)

    # 2. Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_params = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    }
    models["Random Forest"] = (rf, rf_params)

    # 3. XGBoost
    xgb_cls = xgb.XGBClassifier(
        eval_metric="logloss", use_label_encoder=False,
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
    )
    xgb_params = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7, 9],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha":     [0, 0.1, 1.0],
        "reg_lambda":    [1.0, 2.0, 5.0],
    }
    models["XGBoost"] = (xgb_cls, xgb_params)

    # 4. LightGBM
    lgb_cls = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_params = {
        "n_estimators":   [100, 200, 300],
        "max_depth":      [-1, 7, 15],
        "learning_rate":  [0.01, 0.05, 0.1, 0.2],
        "num_leaves":     [31, 63, 127],
        "subsample":      [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha":      [0, 0.1, 1.0],
    }
    models["LightGBM"] = (lgb_cls, lgb_params)

    return models


def train_and_evaluate_classifier(
    name: str,
    estimator: Any,
    param_grid: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    models_dir: str,
) -> Dict:
    """
    Tune, train, evaluate and log one classification model.

    Steps:
    1. RandomizedSearchCV (5-fold stratified CV) on train set.
    2. Evaluate best model on val + test sets.
    3. Log to MLflow.
    4. Save evaluation plots.
    5. Save model artifact.

    Returns
    -------
    dict with all metrics and plot data.
    """
    print(f"\n  ── {name} ──")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    cv_auc = search.best_score_
    best_params = search.best_params_
    print(f"    CV AUC (5-fold): {cv_auc:.4f}")
    print(f"    Best params: {best_params}")

    # ── Validation metrics ───────────────────────────────────────────────────
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    y_val_pred = best_model.predict(X_val)
    val_auc = roc_auc_score(y_val, y_val_prob)

    # ── Test metrics ─────────────────────────────────────────────────────────
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = best_model.predict(X_test)
    test_auc  = roc_auc_score(y_test, y_test_prob)
    test_f1   = f1_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec  = recall_score(y_test, y_test_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)

    print(f"    Val  AUC={val_auc:.4f}")
    print(f"    Test AUC={test_auc:.4f} | F1={test_f1:.4f} | "
          f"Prec={test_prec:.4f} | Rec={test_rec:.4f} | Acc={test_acc:.4f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plot_roc_curve(y_test, y_test_prob, name, out_dir)
    plot_confusion_matrix(y_test, y_test_pred, name, out_dir)
    plot_calibration_curve(y_test, y_test_prob, name, out_dir)

    # ── Save model ───────────────────────────────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(best_model, model_path)
    print(f"    [SAVED] Model → {model_path}")

    # ── MLflow logging ───────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_auc":     cv_auc,
            "val_auc":    val_auc,
            "test_auc":   test_auc,
            "test_f1":    test_f1,
            "test_prec":  test_prec,
            "test_rec":   test_rec,
            "test_acc":   test_acc,
        })
        mlflow.log_artifact(model_path)

    return {
        "model":      name,
        "cv_auc":     cv_auc,
        "val_auc":    val_auc,
        "test_auc":   test_auc,
        "test_f1":    test_f1,
        "test_prec":  test_prec,
        "test_rec":   test_rec,
        "test_acc":   test_acc,
        "best_params":best_params,
        "fpr":        fpr.tolist(),
        "tpr":        tpr.tolist(),
        "auc":        test_auc,
        "model_path": model_path,
    }


def run_classification_pipeline(
    features_path: str = "data/features_bbbp.csv",
    out_dir:       str = "outputs",
    models_dir:    str = "outputs/models",
) -> pd.DataFrame:
    """
    Full classification pipeline for BBBP.

    Parameters
    ----------
    features_path : str
    out_dir : str
    models_dir : str

    Returns
    -------
    pd.DataFrame — model comparison table
    """
    print("\n" + "="*60)
    print("STEP 4 — Classification Pipeline (BBBP)")
    print("="*60)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = \
        _prepare_classification_data(features_path)

    classif_out = os.path.join(out_dir, "classification")
    os.makedirs(classif_out, exist_ok=True)

    models = get_classification_models()
    all_results = []
    roc_data = []

    for name, (estimator, param_grid) in models.items():
        result = train_and_evaluate_classifier(
            name, estimator, param_grid,
            X_train, y_train,
            X_val,   y_val,
            X_test,  y_test,
            classif_out, models_dir,
        )
        all_results.append(result)
        roc_data.append({"model": name, "fpr": result["fpr"],
                          "tpr": result["tpr"], "auc": result["auc"]})

    # ── Overlay ROC curves ────────────────────────────────────────────────────
    plot_all_roc_curves(roc_data, classif_out)

    # ── Model comparison table ────────────────────────────────────────────────
    metric_cols = ["model", "cv_auc", "val_auc", "test_auc",
                   "test_f1", "test_prec", "test_rec", "test_acc"]
    comparison = pd.DataFrame(all_results)[metric_cols]
    comparison = comparison.sort_values("test_auc", ascending=False).reset_index(drop=True)

    comparison_path = os.path.join(out_dir, "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False, float_format="%.4f")
    print(f"\n  [SAVED] Model comparison → {comparison_path}")
    print("\n" + comparison.to_string(index=False))

    # Identify best XGBoost model path for SHAP
    best_xgb = next((r for r in all_results if r["model"] == "XGBoost"), None)
    if best_xgb:
        xgb_model_path = best_xgb["model_path"]
        # Save path for shap module to pick up
        ref_path = os.path.join(models_dir, "best_xgb_path.txt")
        os.makedirs(models_dir, exist_ok=True)
        with open(ref_path, "w") as f:
            f.write(xgb_model_path)

    print("\n[CLASSIFICATION] Complete.")
    return comparison, all_results


# ── Regression Models ──────────────────────────────────────────────────────────

def get_regression_models() -> Dict[str, Tuple[Any, Dict]]:
    """Return (estimator, param_grid) for regression models."""
    models = {}

    # Ridge Regression (scaled)
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(random_state=RANDOM_STATE)),
    ])
    ridge_params = {"reg__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    models["Ridge Regression"] = (ridge_pipe, ridge_params)

    # XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
    )
    xgb_params = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    }
    models["XGBoost Regressor"] = (xgb_reg, xgb_params)

    # LightGBM Regressor
    lgb_reg = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    lgb_params = {
        "n_estimators":   [100, 200, 300],
        "max_depth":      [-1, 7, 15],
        "learning_rate":  [0.01, 0.05, 0.1],
        "num_leaves":     [31, 63],
        "subsample":      [0.7, 0.85, 1.0],
    }
    models["LightGBM Regressor"] = (lgb_reg, lgb_params)

    return models


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str, out_dir: str) -> None:
    """Scatter plot: predicted vs actual solubility."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#3498db", edgecolors="none")

    lim = [min(y_true.min(), y_pred.min()) - 0.5,
           max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    ax.text(0.05, 0.92, f"RMSE={rmse:.3f}\nR²={r2:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Actual LogS", fontsize=12)
    ax.set_ylabel("Predicted LogS", fontsize=12)
    ax.set_title(f"Predicted vs Actual Solubility — {model_name}",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, f"pred_vs_actual_{model_name.replace(' ', '_').lower()}.png"))


def train_and_evaluate_regressor(
    name: str,
    estimator: Any,
    param_grid: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    models_dir: str,
) -> Dict:
    """Tune, train, evaluate and log one regression model."""
    print(f"\n  ── {name} ──")

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=20,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    cv_rmse = -search.best_score_
    print(f"    CV RMSE (5-fold): {cv_rmse:.4f}")

    # Test metrics
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"    Test  RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

    # Plot
    plot_predicted_vs_actual(y_test, y_pred, name, out_dir)

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{name.replace(' ', '_').lower()}_reg.pkl")
    joblib.dump(best_model, model_path)

    # MLflow
    mlflow.set_experiment("drug-discovery-esol")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics({"cv_rmse": cv_rmse, "test_rmse": rmse,
                             "test_mae": mae, "test_r2": r2})
        mlflow.log_artifact(model_path)

    return {
        "model": name, "cv_rmse": cv_rmse,
        "test_rmse": rmse, "test_mae": mae, "test_r2": r2,
        "model_path": model_path,
    }


def run_regression_pipeline(
    features_path: str = "data/features_esol.csv",
    out_dir:       str = "outputs/regression_results",
    models_dir:    str = "outputs/models",
) -> pd.DataFrame:
    """
    Full regression pipeline for ESOL solubility prediction.
    """
    print("\n" + "="*60)
    print("STEP 5 — Regression Pipeline (ESOL)")
    print("="*60)

    os.makedirs(out_dir, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test, _ = \
        _prepare_regression_data(features_path)

    models = get_regression_models()
    all_results = []

    for name, (estimator, param_grid) in models.items():
        result = train_and_evaluate_regressor(
            name, estimator, param_grid,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            out_dir, models_dir,
        )
        all_results.append(result)

    comparison = pd.DataFrame(all_results)[["model", "cv_rmse", "test_rmse",
                                             "test_mae", "test_r2"]]
    comparison = comparison.sort_values("test_rmse").reset_index(drop=True)
    comp_path = os.path.join(out_dir, "regression_comparison.csv")
    comparison.to_csv(comp_path, index=False, float_format="%.4f")
    print(f"\n  [SAVED] Regression comparison → {comp_path}")
    print("\n" + comparison.to_string(index=False))

    print("\n[REGRESSION] Complete.")
    return comparison


# ── Entry points ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    bbbp_feat = "data/features_bbbp.csv"
    esol_feat = "data/features_esol.csv"
    if len(sys.argv) > 1:
        bbbp_feat = sys.argv[1]
    if len(sys.argv) > 2:
        esol_feat = sys.argv[2]

    run_classification_pipeline(bbbp_feat)
    run_regression_pipeline(esol_feat)
