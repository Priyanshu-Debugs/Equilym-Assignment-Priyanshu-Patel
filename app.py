"""
Auto-Analytics Engine — Streamlit Web Application
A production-grade interface for automated statistical insight generation,
data validation, and interactive health indicator analysis.
"""

import io
import json
import os
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from engine.config import AnalyticsConfig, load_indicator_meta
from engine.loader import load_data
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import detect_correlations
from engine.breaches import detect_breaches
from engine.insights import generate_insights

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Auto-Analytics Engine | District Healthcare",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern typography, sleek cards, and badge accents
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .badge-high {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #f59e0b;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #3b82f6;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .caveat-banner {
        background: rgba(245, 158, 11, 0.12);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar: File Upload & Configuration Sliders
# ---------------------------------------------------------
st.sidebar.title("⚙️ Engine Controls")

indicator_meta = load_indicator_meta()

st.sidebar.subheader("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
default_csv_path = "data/district_health.csv"

# Load Data
try:
    if uploaded_file is not None:
        raw_df, validation_report = load_data(uploaded_file, indicator_meta)
        source_name = uploaded_file.name
    elif os.path.exists(default_csv_path):
        raw_df, validation_report = load_data(default_csv_path, indicator_meta)
        source_name = "Default Sample CSV"
    else:
        st.error("No dataset available. Please upload a CSV file.")
        st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# ---------------------------------------------------------
# Dynamic Filters
# ---------------------------------------------------------
st.sidebar.subheader("2. Filters")

all_districts = sorted(raw_df["district"].dropna().unique().tolist())
selected_districts = st.sidebar.multiselect(
    "Districts",
    options=all_districts,
    default=all_districts,
    help="Filter by specific districts.",
)

all_months = sorted(raw_df["month"].dropna().unique().tolist())
selected_months = st.sidebar.multiselect(
    "Months",
    options=all_months,
    default=all_months,
    help="Filter time periods. Note: Trend detection requires at least 2 consecutive months.",
)

all_indicators = validation_report["indicators"]
selected_indicators = st.sidebar.multiselect(
    "Indicators",
    options=all_indicators,
    default=all_indicators,
    help="Select numerical metrics to analyze.",
)

# Filtered DataFrame
filtered_df = raw_df[
    raw_df["district"].isin(selected_districts) & raw_df["month"].isin(selected_months)
].copy()

# Month context warning
if len(selected_months) < 2:
    st.sidebar.warning("⚠️ Fewer than 2 months selected: Trend detection will be inactive.")

# ---------------------------------------------------------
# Detection Parameters
# ---------------------------------------------------------
st.sidebar.subheader("3. Detection Thresholds")

trend_threshold = st.sidebar.slider(
    "Trend Threshold (±%)",
    min_value=1.0,
    max_value=50.0,
    value=10.0,
    step=1.0,
    help="Minimum percentage change required to flag a month-over-month trend.",
)

outlier_method = st.sidebar.radio(
    "Outlier Detection Method",
    options=["IQR", "Z-score"],
    index=0,
    help="IQR is robust for small sample sizes; Z-score flags extreme standard deviations.",
)

if outlier_method == "IQR":
    iqr_k = st.sidebar.slider(
        "IQR Fence Multiplier (k)",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Outlier fences are Q1 - k*IQR and Q3 + k*IQR.",
    )
    z_thresh = 3.0
else:
    z_thresh = st.sidebar.slider(
        "Z-score Threshold (|z|)",
        min_value=1.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        help="Standard deviations from the mean to qualify as an outlier.",
    )
    iqr_k = 1.5

corr_threshold = st.sidebar.slider(
    "Correlation Threshold (|r|)",
    min_value=0.30,
    max_value=1.00,
    value=0.70,
    step=0.05,
    help="Pearson coefficient magnitude to flag pair relationship.",
)

st.sidebar.subheader("4. Policy & Severity Controls")

breach_floor = st.sidebar.slider(
    "Policy Floor (% indicators)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=5.0,
    help="Coverage percentage below which an alert is triggered.",
)

breach_ceiling_pct = st.sidebar.slider(
    "Ceiling Percentile (counts)",
    min_value=50,
    max_value=99,
    value=90,
    step=1,
    help="Upper percentile threshold for inverse/count metrics.",
)

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    sev_med_mult = st.number_input("Med Multiplier", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
with col_s2:
    sev_high_mult = st.number_input("High Multiplier", min_value=1.1, max_value=5.0, value=2.0, step=0.1)

# Package into Config
config = AnalyticsConfig(
    trend_threshold_pct=trend_threshold,
    min_months_for_trend=2,
    outlier_method=outlier_method,
    iqr_multiplier=iqr_k,
    z_threshold=z_thresh,
    corr_threshold=corr_threshold,
    breach_floor_pct=breach_floor,
    breach_ceiling_percentile=float(breach_ceiling_pct),
    severity_medium_x=sev_med_mult,
    severity_high_x=sev_high_mult,
)

# ---------------------------------------------------------
# Execution Engine Pipeline
# ---------------------------------------------------------
trends_df = detect_trends(filtered_df, config, selected_indicators)
outliers_df = detect_outliers(filtered_df, config, selected_indicators)
corr_flagged_df, corr_matrix, corr_meta = detect_correlations(filtered_df, config, selected_indicators)
breaches_df = detect_breaches(filtered_df, config, selected_indicators, indicator_meta)

insights_df = generate_insights(
    trends_df,
    outliers_df,
    corr_flagged_df,
    breaches_df,
    config,
    indicator_meta,
)

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.title("Automated Insight Generation Engine")
st.caption(f"Ingested `{source_name}` · Grain: District-Month · Pure Algorithmic Pipeline")

# Metric Summary Cards
total_insights = len(insights_df)
high_count = len(insights_df[insights_df["severity"] == "High"]) if not insights_df.empty else 0
med_count = len(insights_df[insights_df["severity"] == "Medium"]) if not insights_df.empty else 0
low_count = len(insights_df[insights_df["severity"] == "Low"]) if not insights_df.empty else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Insights", total_insights)
with col2:
    st.metric("🚨 High Severity", high_count)
with col3:
    st.metric("⚠️ Medium Severity", med_count)
with col4:
    st.metric("ℹ️ Low Severity", low_count)

st.markdown("---")

# ---------------------------------------------------------
# Row 1: Insights Table & Severity Distribution
# ---------------------------------------------------------
st.subheader("📋 Structured Insights Stream")

col_table, col_dist = st.columns([2, 1])

with col_dist:
    st.markdown("##### Severity Breakdown")
    if total_insights > 0:
        sev_counts = insights_df["severity"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig_sev = px.bar(
            x=sev_counts.index,
            y=sev_counts.values,
            color=sev_counts.index,
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#3b82f6"},
            labels={"x": "Severity", "y": "Count"},
            text=sev_counts.values,
        )
        fig_sev.update_layout(
            showlegend=False,
            height=280,
            margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sev, use_container_width=True)
    else:
        st.info("No insights flagged with current filter and threshold parameters.")

with col_table:
    # Quick filter on table
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        type_filter = st.multiselect(
            "Filter Type",
            options=["trend", "outlier", "correlation", "threshold_breach"],
            default=["trend", "outlier", "correlation", "threshold_breach"],
        )
    with filter_col2:
        sev_filter = st.multiselect(
            "Filter Severity",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

    display_df = insights_df[
        insights_df["type"].isin(type_filter) & insights_df["severity"].isin(sev_filter)
    ] if not insights_df.empty else pd.DataFrame()

    st.dataframe(
        display_df,
        use_container_width=True,
        height=280,
        column_config={
            "insight_id": st.column_config.TextColumn("ID", width="small"),
            "severity": st.column_config.TextColumn("Severity", width="small"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "explanation": st.column_config.TextColumn("Narrative Explanation", width="large"),
        },
    )

# Downloads
col_dl1, col_dl2, _ = st.columns([1, 1, 2])
if not insights_df.empty:
    with col_dl1:
        csv_data = insights_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Insights CSV",
            data=csv_data,
            file_name="insights.csv",
            mime="text/csv",
        )
    with col_dl2:
        json_data = json.dumps(insights_df.to_dict(orient="records"), indent=2).encode("utf-8")
        st.download_button(
            label="📥 Download Insights JSON",
            data=json_data,
            file_name="insights.json",
            mime="application/json",
        )

st.markdown("---")

# ---------------------------------------------------------
# Row 2: Visualizations (Correlation Heatmap + Line Chart)
# ---------------------------------------------------------
col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.subheader("🔬 Pearson Correlation Matrix")
    if not corr_matrix.empty and len(selected_indicators) >= 2:
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            aspect="auto",
        )
        fig_corr.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        if corr_meta.get("is_small_sample"):
            st.markdown(
                f"""
                <div class="caveat-banner">
                    ⚠️ <b>Small Sample Caveat:</b> {corr_meta.get("warning")}
                    Showing matrix transparently across n={corr_meta.get("n")} rows.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Select at least 2 numerical indicators to compute correlations.")

with col_viz2:
    st.subheader("📈 Multi-District Indicator Trajectories")
    if not filtered_df.empty and selected_indicators:
        chart_indicator = st.selectbox(
            "Select Indicator for Line Chart",
            options=selected_indicators,
            index=0,
        )
        
        # Sort values
        chart_data = filtered_df.sort_values(by=["district", "month"])
        
        fig_line = px.line(
            chart_data,
            x="month",
            y=chart_indicator,
            color="district",
            markers=True,
            title=f"Trajectory: {chart_indicator.replace('_', ' ').title()}",
        )
        
        # Add visual reference lines if percentage
        ind_unit = indicator_meta.get(chart_indicator, {}).get("unit", "%")
        if ind_unit == "%":
            fig_line.add_hline(
                y=config.breach_floor_pct,
                line_dash="dash",
                line_color="rgba(239, 68, 68, 0.7)",
                annotation_text=f"Floor ({config.breach_floor_pct}%)",
            )
            
        fig_line.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("No data available to display line charts.")

# ---------------------------------------------------------
# Row 3: Data Quality & Diagnostics Expander
# ---------------------------------------------------------
st.markdown("---")
with st.expander("🔍 Data Quality & Ingestion Diagnostics (Part A Rubric)", expanded=False):
    st.markdown("##### Dataset Head (First 5 Rows)")
    st.code(validation_report["head"], language="text")

    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.markdown("##### Missing Values Count")
        missing_df = pd.DataFrame(
            list(validation_report["missing"].items()),
            columns=["Column", "Missing Count"],
        )
        st.dataframe(missing_df, use_container_width=True)

    with col_diag2:
        st.markdown("##### Validation Warnings & Audits")
        if validation_report["warnings"]:
            for w in validation_report["warnings"]:
                st.warning(w)
        else:
            st.success("No validation anomalies detected. All columns conform to schema.")

    st.markdown("##### Technical Column Types (df.info())")
    st.code(validation_report["info"], language="text")
