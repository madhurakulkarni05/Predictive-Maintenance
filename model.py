"""
model.py
========
Train and persist regressors for Remaining Useful Life (RUL) on FD001.

Models (scikit-learn / XGBoost):
- Random Forest Regressor (primary saved artifact for the Streamlit app)
- Linear Support Vector Regression (scalable on ~20k FD001 rows)
- XGBoost Regressor

Artifacts are saved with joblib and include the scaler + feature list for
consistent inference in the web app.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from preprocess import (
    FEATURE_COLS,
    build_training_matrix,
    compute_rul_per_row,
    drop_constant_sensors,
    fit_scaler,
    load_raw_fd001,
    scale_features,
)


# -----------------------------------------------------------------------------
# Paths (project root = directory containing this file)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train_FD001.txt"
MODEL_DIR = ROOT / "models"
RF_PATH = MODEL_DIR / "rf_model.joblib"
BUNDLE_META_PATH = MODEL_DIR / "training_meta.json"


def train_models(
    data_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Load data, engineer RUL, train RF / SVR / XGBoost, evaluate, save RF bundle.

    Returns
    -------
    dict
        Metrics and paths for each model.
    """
    data_path = data_path or DATA_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_fd001(data_path)
    raw, active_features = drop_constant_sensors(raw)
    labeled = compute_rul_per_row(raw, rul_cap=125)
    X_df, y = build_training_matrix(labeled, active_features)

    scaler = fit_scaler(X_df)
    X = scale_features(X_df, scaler)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models: Dict[str, Any] = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        # LinearSVR scales better than SVR(kernel="linear") on ~20k FD001 rows
        "svr": LinearSVR(C=1.0, epsilon=0.1, max_iter=5000, random_state=random_state),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    metrics: Dict[str, Dict[str, float]] = {}

    for name, est in models.items():
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }

    bundle: Dict[str, Any] = {
        "model": models["random_forest"],
        "scaler": scaler,
        "feature_cols": active_features,
        "rul_cap": 125,
    }
    joblib.dump(bundle, RF_PATH)

    meta = {
        "data_path": str(data_path),
        "n_samples": int(len(labeled)),
        "feature_cols": active_features,
        "metrics": metrics,
        "saved_rf_path": str(RF_PATH),
    }
    BUNDLE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"metrics": metrics, "bundle_path": str(RF_PATH), "meta_path": str(BUNDLE_META_PATH)}


def load_model_bundle(path: Path | None = None) -> Dict[str, Any]:
    """Load joblib bundle produced by train_models."""
    path = path or RF_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file at {path}. Run: python model.py"
        )
    return joblib.load(path)


def predict_rul(features: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    """
    Predict RUL for a 2D feature array (n_samples, n_features).

    features must match bundle['feature_cols'] order before scaling.
    """
    scaler = bundle["scaler"]
    model = bundle["model"]
    Xs = scaler.transform(np.asarray(features, dtype=np.float64))
    return model.predict(Xs)


if __name__ == "__main__":
    results = train_models()
    print("Training complete.")
    for name, m in results["metrics"].items():
        print(f"  {name}: RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} R2={m['r2']:.3f}")
    print(f"Saved: {results['bundle_path']}")
