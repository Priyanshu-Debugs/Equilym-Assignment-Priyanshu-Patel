# Auto-Analytics Engine
**Assignment 4 — Automated Insight Generation**

A lightweight, production-grade Auto-Analytics Engine that ingests district-level monthly healthcare indicators and automatically identifies **trends**, **outliers**, **correlations**, and **threshold breaches**, producing structured, natural language insights with data-derived severity.

---

## 📑 Table of Contents
1. [Core Features](#core-features)
2. [Architecture & Design Rationale](#architecture--design-rationale)
3. [Quick Start (Local & Conda)](#quick-start-local--conda)
4. [Configuration & Thresholds](#configuration--thresholds)
5. [Severity Methodology](#severity-methodology)
6. [Statistical Limitations (Small-Sample Caution)](#statistical-limitations-small-sample-caution)
7. [Automated Testing & Rubric Validation](#automated-testing--rubric-validation)

---

## 🎯 Core Features

- **Zero Hardcoding**: Strictly general-purpose pipeline. No district names, hardcoded thresholds, or hardcoded numbers appear anywhere in the engine modules.
- **Dynamic Indicator Discovery**: Any new numerical column added to the CSV flows through trend, outlier, correlation, and breach detection automatically.
- **Four Standard Insight Types**:
  - `trend`: Month-over-month percentage change with directionality.
  - `outlier`: Distributional anomaly detection via IQR fences or Z-scores.
  - `correlation`: Pearson correlation matrix with leave-one-out robustness checks.
  - `threshold_breach`: Policy floors (e.g. coverage minimums) and empirical ceilings.
- **Data-Derived Severity**: Low, Medium, and High ratings derived from magnitude-to-threshold ratios rather than arbitrary `if change > 50` hardcoded rules.
- **Clean & User-Friendly UI**: Polished Streamlit interface organized into structured tabs:
  - 🌟 **Executive Insights Feed**: KPI metrics, severity distribution, interactive tables, card narratives, and downloads.
  - 📊 **Deep-Dive Visualizations**: Pearson correlation heatmap with small-sample caveats and per-district trajectory charts.
  - 🔍 **Data Diagnostics**: Automated schema audits, missing value checks, and data quality reports.
- **Export Standards**: Serializes findings strictly conforming to the assignment specification in both `outputs/insights.csv` and `outputs/insights.json`.

---

## 🏗️ Architecture & Design Rationale

```
auto_analytics/
├── app.py                  # Streamlit UI (visualization, tabs, filters, download buttons)
├── engine/                 # Pure functional algorithmic core (no Streamlit dependencies)
│   ├── __init__.py
│   ├── config.py           # Dataclass of parameters & indicator metadata loader
│   ├── loader.py           # Part A: CSV ingestion, schema validation & diagnostics
│   ├── trends.py           # Part B: Month-over-month change & linear slopes
│   ├── outliers.py         # Part C: Pooled IQR fences & Z-score detection
│   ├── correlations.py     # Part D: Pearson correlation & leave-one-out robustness
│   ├── breaches.py         # Threshold breach detection against floors/ceilings
│   ├── severity.py         # Continuous magnitude ratio to Low/Medium/High mapping
│   └── insights.py         # Part E: Templated copy generation & serialization
├── config/
│   └── indicator_meta.json # Configurable indicator directionality & units
├── data/
│   └── district_health.csv # Canonical sample dataset
├── outputs/                # Serialized CSV and JSON exports
├── tests/
│   ├── __init__.py
│   └── test_engine.py      # 12 automated pytest suites including static code audits
└── requirements.txt
```

### Pure Functional Design
Each module in `engine/` takes a `pandas.DataFrame` and an `AnalyticsConfig` object as input and outputs a structured `DataFrame`. This keeps the solution testable, lightweight, and ensures "no separate backend/API" is needed while keeping business logic decoupled from presentation.

---

## 🚀 Quick Start (Local & Conda)

### 1. Using Conda (Recommended)
```powershell
# Open PowerShell and activate your conda environment
& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1'
conda activate base

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

### 2. Using standard Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## ⚙️ Configuration & Thresholds

All thresholds are exposed as interactive controls in the sidebar and defined in `engine/config.py`:

| Parameter | Default | UI Control | Purpose |
|:---|:---:|:---:|:---|
| `trend_threshold_pct` | `10.0%` | Slider [1–50%] | Minimum % change to trigger a trend alert |
| `outlier_method` | `IQR` | Radio [IQR / Z-score] | Statistical algorithm used for outlier identification |
| `iqr_multiplier` | `1.5` | Slider [0.5–3.0] | Fence width multiplier: `[Q1 - k*IQR, Q3 + k*IQR]` |
| `z_threshold` | `3.0` | Slider [1.0–4.0] | Standard deviation distance threshold |
| `corr_threshold` | `0.70` | Slider [0.30–1.00] | Pearson magnitude to flag indicator pair |
| `breach_floor_pct` | `60.0%` | Slider [0–100%] | Minimum policy floor for percentage indicators |
| `breach_ceiling_percentile` | `90.0` | Slider [50–99] | Empirical percentile ceiling for counts/inverse metrics |
| `severity_medium_x` | `1.0` | Number Input | Ratio multiplier boundary for Medium severity |
| `severity_high_x` | `2.0` | Number Input | Ratio multiplier boundary for High severity |

---

## 📊 Severity Methodology

Severity is **data-derived** using continuous magnitude ratios rather than hardcoded heuristics:

$$\text{Ratio} = \frac{\text{Observed Magnitude}}{\text{Configured Threshold}}$$

- **Trend**: $\text{Ratio} = \frac{|\Delta\%|}{\text{trend\_threshold\_pct}}$
- **Outlier (IQR)**: $\text{Ratio} = 1.0 + \frac{\text{Distance Beyond Fence}}{k \times \text{IQR}}$
- **Outlier (Z-score)**: $\text{Ratio} = \frac{|z|}{\text{z\_threshold}}$
- **Correlation**: $\text{Ratio} = \frac{|r|}{\text{corr\_threshold}}$
- **Breach**: $\text{Ratio} = 1.0 + \frac{|\text{Value} - \text{Limit}|}{\text{Limit}}$

Categorization:
- $\text{Ratio} < \text{severity\_medium\_x} \longrightarrow \mathbf{Low}$
- $\text{severity\_medium\_x} \le \text{Ratio} < \text{severity\_high\_x} \longrightarrow \mathbf{Medium}$
- $\text{Ratio} \ge \text{severity\_high\_x} \longrightarrow \mathbf{High}$

---

## ⚠️ Statistical Limitations (Small-Sample Caution)

1. **Pooled Pearson Correlations ($n=12$)**:
   With only 2 months across 6 districts, pooled correlations are fragile and can be driven by a single leverage point. The UI displays an explicit warning whenever $n < 30$, and the engine runs a leave-one-out sensitivity check to evaluate robustness.
2. **Sample Z-Score Behavior**:
   In a sample of $n=12$, no data point can mathematically exceed $|z| \approx 3.18$. For instance, Mehsana's ANC coverage drop to 42 yields $z \approx -2.83$ against the pooled mean of $78.33$. While the brief mentions an illustrative $3.1\sigma$, our engine transparently reports the exact mathematical $z \approx -2.83\sigma$. The method defaults to **IQR**, which reliably identifies this anomaly without parametric normality assumptions.

---

## 🧪 Automated Testing & Rubric Validation

Run the test suite with `pytest`:
```powershell
& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1'; conda activate base; pytest -v
```

The test suite validates:
- [x] **Part A**: CSV loading, schema inspection, missing counts report, and duplicate handling.
- [x] **Part B**: Ahmedabad ANC $(-18.82\%)$ and Mehsana high risk cases $(+154.55\%)$ trend detection.
- [x] **Part C**: Mehsana ANC ($42$) and Ahmedabad ANC ($69$) IQR outlier detection.
- [x] **Part D**: Pearson correlation matrix, sample size warnings, and CSV serialization.
- [x] **Part E**: Gating criteria insight generation, structured columns, and templates.
- [x] **Criterion 5 Integrity**: Automated static regex scan confirming **0 hardcoded district strings** in `engine/*.py`.
- [x] **Robustness**: Safe handling of zero division, constant columns, and single-month filtered data.
