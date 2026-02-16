from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Custom imports (kept as-is for your environment)
from src.data_loader import load_clean_data
from src.monitoring.drift import KSTest

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Monitoring Experiments", page_icon="📈", layout="wide")

MONITORED_FEATURES = [
    "Country", "StockCode", "Hour", "Month", "Day",
    "is_weekend", "day_of_week", "hour_sin", "hour_cos",
    "quarter", "country_freq", "stockcode_freq", "InvoiceDate",
]
NUMERIC_MONITORED_FEATURES = [
    "Hour", "Month", "Day",
    "is_weekend", "day_of_week",
    "hour_sin", "hour_cos",
    "quarter",
    "country_freq", "stockcode_freq",
]

DEFAULT_REFERENCE_START = pd.to_datetime("2010-04-30").date()
DEFAULT_REFERENCE_END = pd.to_datetime("2010-06-30").date()
DEFAULT_WINDOW_START = pd.to_datetime("2011-11-30").date()
DEFAULT_WINDOW_END = pd.to_datetime("2011-12-09").date()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_processed() -> pd.DataFrame:
    candidate_paths = [
        "data/processed/data.csv",
        "data/processed/train.csv",
    ]
    for path in candidate_paths:
        if Path(path).exists():
            df = load_clean_data(path)
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
            return df
            
    # Fallback to first path to trigger standard error if not found
    df = load_clean_data(candidate_paths[0])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

def normalize_range(value: tuple[date, date] | tuple[date] | date) -> tuple[date, date]:
    """Safely handles Streamlit's date_input tuple lengths during user selection."""
    if isinstance(value, tuple):
        if len(value) == 2:
            start, end = value
        elif len(value) == 1:
            start = end = value[0]
        else:
            start = end = date.today()
    else:
        start = end = value
        
    if start > end:
        start, end = end, start
    return start, end

def highlight_drift(val):
    """Pandas styler to highlight drifted features in red."""
    if val == "drift detected":
        return "background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; font-weight: bold;"
    return ""

# -----------------------------------------------------------------------------
# Data Loading & Validation
# -----------------------------------------------------------------------------
data = load_processed()

if data.empty:
    st.error("No processed data found. Please check `data/processed/data.csv`.")
    st.stop()

missing = set(MONITORED_FEATURES) - set(data.columns)
if missing:
    st.error(f"Missing monitored columns: {sorted(missing)}")
    st.stop()

min_date = data["InvoiceDate"].min().date()
max_date = data["InvoiceDate"].max().date()

ref_start_default = max(min_date, min(DEFAULT_REFERENCE_START, max_date))
ref_end_default = max(min_date, min(DEFAULT_REFERENCE_END, max_date))
window_start_default = max(min_date, min(DEFAULT_WINDOW_START, max_date))
window_end_default = max(min_date, min(DEFAULT_WINDOW_END, max_date))

# -----------------------------------------------------------------------------
# Sidebar Navigation & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    
    with st.expander("Model Parameters", expanded=True):
        alpha = st.slider("KS alpha (Significance)", 0.01, 0.20, 0.05, 0.01)

    with st.expander("Time Windows", expanded=True):
        ref_range = st.date_input(
            "Reference Window",
            value=(ref_start_default, ref_end_default),
            min_value=min_date, max_value=max_date,
        )
        window_range = st.date_input(
            "Monitoring Window",
            value=(window_start_default, window_end_default),
            min_value=min_date, max_value=max_date,
        )

ref_start, ref_end = normalize_range(ref_range)
window_start, window_end = normalize_range(window_range)

reference_mask = data["InvoiceDate"].between(pd.Timestamp(ref_start), pd.Timestamp(ref_end))
window_mask = data["InvoiceDate"].between(pd.Timestamp(window_start), pd.Timestamp(window_end))

reference_data = data.loc[reference_mask, MONITORED_FEATURES].copy()
window_data = data.loc[window_mask, MONITORED_FEATURES].copy()

if reference_data.empty or window_data.empty:
    st.warning("Reference or monitoring window is empty. Please adjust the dates in the sidebar.")
    st.stop()

reference_numeric = reference_data.drop(columns=["InvoiceDate"])
window_numeric = window_data.drop(columns=["InvoiceDate"])

numeric_features = [f for f in NUMERIC_MONITORED_FEATURES if f in reference_numeric.columns]

if not numeric_features:
    st.warning("No numeric monitored features available for KS testing.")
    st.stop()

# -----------------------------------------------------------------------------
# Calculations
# -----------------------------------------------------------------------------
# Run Drift Detection
ks_test = KSTest(alpha=alpha).fit(reference_numeric, numeric_features)
alerts, stats, p_values = ks_test.feature_drift_detection(window_numeric, return_details=True)

results = pd.DataFrame([
    {
        "Feature": feature,
        "Alert": alerts.get(feature, "missing"),
        "KS Statistic": stats.get(feature, np.nan),
        "P-Value": p_values.get(feature, np.nan),
    }
    for feature in numeric_features
])

drifted_count = int((results["Alert"] == "drift detected").sum())

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
st.title("Feature Drift Monitoring")
st.caption("Compare reference and monitoring windows to detect feature distribution shifts.")

# Top-level metrics
col1, col2, col3 = st.columns(3)
col1.metric("Reference Rows", f"{len(reference_data):,}")
col2.metric("Monitoring Rows", f"{len(window_data):,}")
col3.metric("Drifted Features", f"{drifted_count} / {len(numeric_features)}")

st.divider()

# Tabbed Navigation
tab1, tab2, tab3 = st.tabs(["📊 Overview & Results", "📉 P-Value Analysis", "🔍 Distribution Deep Dive"])

with tab1:
    st.subheader("Kolmogorov-Smirnov (KS) Test Results")
    # Apply conditional formatting using map (for Pandas >= 2.1.0) or applymap (older versions)
    try:
        styled_df = results.style.map(highlight_drift, subset=["Alert"])
    except AttributeError:
        styled_df = results.style.applymap(highlight_drift, subset=["Alert"])
        
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        hide_index=True
    )

with tab2:
    st.subheader("P-Values by Feature")
    st.markdown("Features with a p-value below the $\\alpha$ threshold trigger a drift alert.")
    
    chart_df = results.dropna(subset=["P-Value"]).set_index("Feature")["P-Value"]
    if not chart_df.empty:
        st.bar_chart(chart_df, color="#4b8bfe")

with tab3:
    st.subheader("Feature Distribution Comparison")
    
    # Put controls inside a horizontal layout to save vertical space
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_feature = st.selectbox("Select Feature to Inspect", numeric_features)
    with c2:
        bins = st.slider("Histogram Bins", 10, 100, 30)

    # Calculate distributions
    ref_hist, bin_edges = np.histogram(reference_numeric[selected_feature], bins=bins, density=True)
    window_hist, _ = np.histogram(window_numeric[selected_feature], bins=bin_edges, density=True)

    hist_df = pd.DataFrame({
        "Bin": bin_edges[:-1],
        "Reference": ref_hist,
        "Monitoring": window_hist,
    }).set_index("Bin")

    # Area chart provides a much better visual for density comparisons than line charts
    st.area_chart(hist_df, color=["#1f77b4", "#ff7f0e"])