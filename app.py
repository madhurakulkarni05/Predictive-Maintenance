"""
app.py
======
Streamlit web application for turbofan RUL prediction and health monitoring.

Provides CSV upload, interactive sensor trends (Plotly), metric cards, and a
gauge-style health indicator aligned with the trained Random Forest model.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import load_model_bundle, predict_rul
from preprocess import (
    health_bucket,
    health_score_from_rul,
    load_raw_fd001,
    prepare_uploaded_frame,
)

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance | RUL Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "rf_model.joblib"


def inject_theme_css() -> None:
    """Industrial dark theme: deep blue/black base with orange accents."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
        }

        .stApp {
            background: radial-gradient(1200px 800px at 10% -10%, #0d1b2a 0%, #050608 45%, #020308 100%);
            color: #e8eef7;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #050810 100%);
            border-right: 1px solid rgba(255, 140, 66, 0.25);
        }

        /* Extra top inset so the first line clears the app header / toolbar */
        section[data-testid="stMain"] .block-container,
        .stAppViewContainer .main .block-container,
        .main .block-container {
            padding-top: 3.5rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }

        .pm-hero {
            margin-top: 0.5rem;
            margin-bottom: 0.35rem;
            padding-top: 0.35rem;
            line-height: 1.4;
        }

        /* Metric cards */
        .pm-card {
            background: linear-gradient(145deg, rgba(20, 35, 60, 0.92), rgba(8, 12, 22, 0.95));
            border: 1px solid rgba(255, 140, 66, 0.22);
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 0 0 1px rgba(0,0,0,0.35), 0 18px 40px rgba(0,0,0,0.45);
            min-height: 118px;
        }
        .pm-card h3 {
            margin: 0 0 0.35rem 0;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8fa6c4;
            font-weight: 600;
        }
        .pm-card .value {
            font-size: 2rem;
            font-weight: 700;
            color: #f4f7ff;
            line-height: 1.1;
        }
        .pm-card .sub {
            margin-top: 0.35rem;
            font-size: 0.85rem;
            color: #9db2ce;
        }

        .status-healthy { color: #3ddc84 !important; text-shadow: 0 0 18px rgba(61,220,132,0.35); }
        .status-warning { color: #ffcc66 !important; text-shadow: 0 0 16px rgba(255,204,102,0.35); }
        .status-critical { color: #ff4d4d !important; }

        /* Critical pulsing alert banner */
        .critical-alert {
            background: rgba(80, 10, 10, 0.55);
            border: 1px solid rgba(255, 77, 77, 0.65);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            margin: 0.5rem 0 1rem 0;
            color: #ffd6d6;
            font-weight: 600;
            text-align: center;
            animation: pm-pulse 1.4s ease-in-out infinite;
            box-shadow: 0 0 28px rgba(255, 60, 60, 0.45), inset 0 0 22px rgba(255, 0, 0, 0.12);
        }
        @keyframes pm-pulse {
            0%, 100% { box-shadow: 0 0 22px rgba(255, 60, 60, 0.35); }
            50%      { box-shadow: 0 0 40px rgba(255, 80, 80, 0.75); }
        }

        div[data-testid="stHeader"] {
            background: rgba(5, 8, 16, 0.3);
        }

        .pm-title {
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: #f2f5ff;
            line-height: 1.35;
            padding-top: 0.12em;
        }
        .pm-sub {
            color: #8ea3c2;
            font-size: 0.95rem;
            margin-top: 0.25rem;
            line-height: 1.4;
        }

        /* Streamlit widgets — orange accent */
        .stButton>button {
            background: linear-gradient(90deg, #ff8c42, #ff6b1a);
            color: #0a0e18;
            font-weight: 600;
            border: none;
            border-radius: 10px;
        }
        .stButton>button:hover {
            box-shadow: 0 0 18px rgba(255, 140, 66, 0.55);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_bundle_safe() -> Optional[Dict[str, Any]]:
    """Load RF bundle; return None if missing (sidebar will guide training)."""
    path = DEFAULT_MODEL
    if not path.exists():
        return None
    return load_model_bundle(path)


def pick_primary_unit(df: pd.DataFrame) -> int:
    """Use the engine with the most rows as default for visualization."""
    counts = df.groupby("unit_id").size()
    return int(counts.idxmax())


def status_emoji(status: str) -> str:
    """Traffic-light emoji for health band (per product spec)."""
    if status == "Healthy":
        return "🟢"
    if status == "Warning":
        return "🟡"
    return "🔴"


def build_sensor_figure(
    unit_df: pd.DataFrame,
    sensor_subset: Optional[List[str]] = None,
) -> go.Figure:
    """Smooth multi-sensor time series for the selected engine."""
    sensor_subset = sensor_subset or [c for c in unit_df.columns if c.startswith("sensor_")][:8]
    fig = go.Figure()
    cycles = unit_df["time_cycles"].values
    for col in sensor_subset:
        if col in unit_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=cycles,
                    y=unit_df[col].values,
                    mode="lines",
                    name=col.replace("sensor_", "S"),
                    line=dict(width=2, shape="spline"),
                )
            )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,28,0.65)",
        margin=dict(l=48, r=24, t=48, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(color="#dbe5f5", family="IBM Plex Sans, sans-serif"),
        title=dict(
            text="Sensor trends (selected unit)",
            font=dict(size=18, color="#f2f5ff"),
        ),
        xaxis=dict(
            title="Time cycle",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Reading",
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        hovermode="x unified",
        height=420,
    )
    return fig


def build_gauge_figure(score: float, status: str) -> go.Figure:
    """Animated-style gauge using Plotly indicator (updates on rerun)."""
    # Color steps tied to health score
    if status == "Critical":
        bar_color = "#ff4d4d"
    elif status == "Warning":
        bar_color = "#ffcc66"
    else:
        bar_color = "#3ddc84"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 44, "color": "#f4f7ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6c7a90"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(12,18,32,0.9)",
                "borderwidth": 1,
                "bordercolor": "rgba(255,140,66,0.35)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(120,20,20,0.35)"},
                    {"range": [30, 64], "color": "rgba(120,90,20,0.22)"},
                    {"range": [64, 100], "color": "rgba(20,90,60,0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#ff8c42", "width": 3},
                    "thickness": 0.82,
                    "value": score,
                },
            },
            title={"text": "Health score", "font": {"color": "#8fa6c4", "size": 16}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe5f5", family="IBM Plex Sans, sans-serif"),
        height=320,
        margin=dict(l=30, r=30, t=40, b=20),
    )
    # Subtle transition when Streamlit refreshes
    fig.update_layout(
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    return fig


def render_metric_card(title: str, value: str, subtitle: str, css_class: str = "") -> str:
    """Return HTML for a single KPI card."""
    val_html = f'<div class="value {css_class}">{value}</div>'
    return f"""
    <div class="pm-card">
        <h3>{title}</h3>
        {val_html}
        <div class="sub">{subtitle}</div>
    </div>
    """


def main() -> None:
    inject_theme_css()

    st.markdown(
        '<div class="pm-hero">'
        '<div class="pm-title">Industrial Predictive Maintenance</div>'
        '<div class="pm-sub">NASA CMAPSS FD001 · Remaining Useful Life (RUL) intelligence</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    bundle = load_bundle_safe()

    with st.sidebar:
        st.markdown("### Control room")
        st.caption("Upload engine telemetry (CSV). Columns must match FD001 layout.")
        uploaded = st.file_uploader("Sensor data (CSV or NASA TXT)", type=["csv", "txt"])

        st.markdown("---")
        st.markdown("**Model**")
        if bundle is None:
            st.warning("No trained model found. Run `python model.py` from the project root.")
        else:
            st.success("Random Forest bundle loaded.")
            feats = bundle.get("feature_cols", [])
            st.caption(f"{len(feats)} active features · RUL cap {bundle.get('rul_cap', 125)}")

        st.markdown("---")
        show_sensors = st.multiselect(
            "Sensors to plot",
            options=[f"sensor_{i}" for i in range(1, 22)],
            default=[f"sensor_{i}" for i in (2, 3, 4, 7, 11, 12, 15, 17)],
        )

    if uploaded is None:
        st.info("Upload a CSV file to begin. Use the same schema as `train_FD001.txt` (unit, cycle, settings, 21 sensors).")
        return

    try:
        raw_bytes = uploaded.getvalue()
        name = (uploaded.name or "").lower()
        if name.endswith(".txt"):
            text = raw_bytes.decode("utf-8", errors="replace")
            df = load_raw_fd001(io.StringIO(text))
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes))
            df = prepare_uploaded_frame(df)
    except Exception as exc:  # noqa: BLE001 — surface parse errors to the user
        st.error(f"Could not read or validate file: {exc}")
        return

    if bundle is None:
        st.error("Train the model before running predictions.")
        return

    units_sorted = sorted(df["unit_id"].unique().tolist())
    with st.sidebar:
        unit_id = st.selectbox(
            "Engine unit",
            options=units_sorted,
            index=units_sorted.index(pick_primary_unit(df)) if units_sorted else 0,
            help="Focus charts and headline RUL on one turbofan.",
        )

    feature_cols: List[str] = bundle["feature_cols"]
    missing_preds = [c for c in feature_cols if c not in df.columns]
    if missing_preds:
        st.error(f"CSV is missing model features: {', '.join(missing_preds)}")
        return

    unit_df = df[df["unit_id"] == unit_id].sort_values("time_cycles")
    last_row = unit_df.iloc[[-1]]
    X_last = last_row[feature_cols].values.astype(np.float64)
    rul_pred = float(predict_rul(X_last, bundle)[0])
    status, risk = health_bucket(rul_pred)
    score = health_score_from_rul(rul_pred, ref_max=float(bundle.get("rul_cap", 125)))

    if status == "Critical":
        st.markdown(
            '<div class="critical-alert">CRITICAL · Immediate maintenance recommended</div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    status_class = {
        "Healthy": "status-healthy",
        "Warning": "status-warning",
        "Critical": "status-critical",
    }[status]

    with c1:
        st.markdown(
            render_metric_card(
                "Predicted RUL",
                f"{rul_pred:.1f}",
                "cycles remaining (capped training scale)",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            render_metric_card(
                "Health score",
                f"{score:.0f}%",
                "derived from RUL vs reference horizon",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            render_metric_card(
                "Risk level",
                f"{status_emoji(status)} {risk}",
                f"Status: {status}",
                css_class=status_class,
            ),
            unsafe_allow_html=True,
        )

    g1, g2 = st.columns([1, 1.35])
    with g1:
        st.plotly_chart(build_gauge_figure(score, status), use_container_width=True)
    with g2:
        st.plotly_chart(
            build_sensor_figure(unit_df, sensor_subset=show_sensors),
            use_container_width=True,
        )

    st.markdown("#### Full trajectory predictions")
    X_all = unit_df[feature_cols].values.astype(np.float64)
    preds = predict_rul(X_all, bundle)
    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=unit_df["time_cycles"],
            y=preds,
            mode="lines",
            name="RUL prediction",
            line=dict(color="#ff8c42", width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(255,140,66,0.12)",
        )
    )
    trend_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,16,28,0.65)",
        title=dict(text=f"Estimated RUL over cycles · Unit {unit_id}", font=dict(color="#f2f5ff")),
        xaxis=dict(title="Cycle", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="RUL", gridcolor="rgba(255,255,255,0.06)"),
        height=360,
        margin=dict(l=48, r=24, t=48, b=40),
        font=dict(color="#dbe5f5"),
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    with st.expander("Raw preview (uploaded)"):
        st.dataframe(df.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
