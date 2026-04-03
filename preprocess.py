"""
preprocess.py
=============
Data loading, cleaning, RUL label generation, and feature preparation for
NASA CMAPSS Turbofan FD001 (train split).

The training file contains multivariate time series per engine until failure.
Remaining Useful Life (RUL) for each row is derived as the number of cycles
from the current row until the last recorded cycle for that engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Column layout for FD001 (and typical CMAPSS text exports): 26 columns
# -----------------------------------------------------------------------------
INDEX_COLS = ["unit_id", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

# All columns used as model inputs (settings + sensors)
FEATURE_COLS: List[str] = SETTING_COLS + SENSOR_COLS


def load_raw_fd001(path: Union[str, Path, IO[str]], sep: str = r"\s+") -> pd.DataFrame:
    """
    Load FD001-style whitespace-separated data with no header row.

    Parameters
    ----------
    path : str | Path | IO[str]
        Path to train_FD001.txt, or a text buffer (e.g. StringIO from upload).
    sep : str
        Separator passed to pandas.read_csv; default splits on whitespace.

    Returns
    -------
    pd.DataFrame
        Typed dataframe with named columns.
    """
    col_names = INDEX_COLS + FEATURE_COLS
    df = pd.read_csv(path, sep=sep, header=None, names=col_names, engine="python")
    # Drop accidental empty columns from trailing delimiters
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["unit_id"] = df["unit_id"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    return df


def compute_rul_per_row(df: pd.DataFrame, rul_cap: Optional[int] = 125) -> pd.DataFrame:
    """
    Add RUL column: cycles remaining until failure for each engine row.

    Piecewise RUL capping (default 125) follows common CMAPSS practice and
    stabilizes regression targets for early-life segments.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain unit_id and time_cycles.
    rul_cap : Optional[int]
        If set, RUL is min(raw_rul, rul_cap). Use None to disable capping.

    Returns
    -------
    pd.DataFrame
        Copy of df with column 'RUL'.
    """
    out = df.copy()
    max_cycle = out.groupby("unit_id")["time_cycles"].transform("max")
    raw_rul = max_cycle - out["time_cycles"]
    if rul_cap is not None:
        out["RUL"] = np.minimum(raw_rul.values, rul_cap)
    else:
        out["RUL"] = raw_rul.values
    return out


def drop_constant_sensors(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove sensors (and settings) with zero variance — they carry no signal.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Filtered dataframe and the list of columns kept for modeling.
    """
    usable = []
    for col in FEATURE_COLS:
        if col in df.columns and df[col].nunique(dropna=False) > 1:
            usable.append(col)
    return df, usable


def build_training_matrix(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Split features X and target y (RUL) for supervised training.

    Returns
    -------
    X : pd.DataFrame
    y : np.ndarray
    """
    if "RUL" not in df.columns:
        raise ValueError("DataFrame must contain RUL; run compute_rul_per_row first.")
    X = df[feature_cols].astype(np.float64)
    y = df["RUL"].astype(np.float64).values
    return X, y


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training features (settings + sensors)."""
    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler


def scale_features(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform feature block using a fitted StandardScaler."""
    return scaler.transform(X.values.astype(np.float64))


def health_bucket(rul: float) -> Tuple[str, str]:
    """
    Map predicted RUL to discrete health status and risk label.

    Thresholds (cycles):
    - RUL > 80  -> Healthy / Low
    - 30–80     -> Warning / Medium
    - < 30      -> Critical / High
    """
    if rul > 80:
        return "Healthy", "Low"
    if rul >= 30:
        return "Warning", "Medium"
    return "Critical", "High"


def health_score_from_rul(rul: float, ref_max: float = 125.0) -> float:
    """
    Convert RUL to a 0–100 health score for gauge display.
    Higher score means healthier (more remaining life).
    """
    if ref_max <= 0:
        return 0.0
    return float(np.clip((rul / ref_max) * 100.0, 0.0, 100.0))


def prepare_uploaded_frame(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize user-uploaded CSV column names to internal schema.

    Accepts either our named columns or generic names like s1..s21.
    """
    df = uploaded_df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {}
    lower = {c.lower(): c for c in df.columns}
    if "unit_id" not in df.columns and "id" in lower:
        rename_map[lower["id"]] = "unit_id"
    if "time_cycles" not in df.columns:
        for key, target in [("cycle", "time_cycles"), ("cycles", "time_cycles")]:
            if key in lower:
                rename_map[lower[key]] = "time_cycles"
                break

    for i in range(1, 22):
        for candidate in (f"sensor_{i}", f"s{i}", f"sensor{i}"):
            if candidate in lower:
                rename_map[lower[candidate]] = f"sensor_{i}"
                break

    df = df.rename(columns=rename_map)

    missing = [c for c in INDEX_COLS + FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Uploaded file is missing required columns: "
            + ", ".join(missing)
            + ". Expected unit_id, time_cycles, 3 settings, 21 sensors."
        )

    df["unit_id"] = df["unit_id"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)
    return df
