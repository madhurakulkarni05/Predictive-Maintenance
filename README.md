# Predictive Maintenance Web Application

End-to-end Streamlit application that predicts Remaining Useful Life (RUL) for NASA CMAPSS turbofan engines (FD001) and visualizes equipment health for proactive maintenance decisions.

## Features

- Data preprocessing pipeline for CMAPSS FD001
- RUL label generation and sensor normalization
- ML models: Random Forest, LinearSVR, XGBoost
- Streamlit dashboard with:
  - Upload for CSV/TXT telemetry files
  - Real-time RUL prediction
  - Health status (Healthy/Warning/Critical)
  - Plotly sensor trends and RUL trend
  - Gauge chart and KPI cards

## Tech Stack

- Python
- Streamlit
- scikit-learn
- XGBoost
- pandas / NumPy
- Plotly
- joblib

## Project Structure

```text
predictive_maintenance/
├── app.py
├── model.py
├── preprocess.py
├── requirements.txt
├── models/
│   └── rf_model.joblib
├── data/
│   └── train_FD001.txt
└── screenshots/
    ├── dashboard-overview.png
    └── dashboard-rul-trend.png
```

## Quick Start

```bash
pip install -r requirements.txt
python model.py
streamlit run app.py
```

## Dashboard Screenshots

![Dashboard Overview](screenshots/dashboard-overview.png)
![RUL Trend View](screenshots/dashboard-rul-trend.png)

