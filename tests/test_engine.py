"""
Unit and integration test suite for the Auto-Analytics Engine.
Verifies all functional criteria, edge cases, output schemas, and tests against hardcoding.
"""

import os
import glob
import re
import pytest
import pandas as pd
import numpy as np

from engine.config import AnalyticsConfig, load_indicator_meta
from engine.loader import load_data
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import detect_correlations
from engine.breaches import detect_breaches
from engine.severity import assign_severity
from engine.insights import generate_insights


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "data", "district_health.csv")


@pytest.fixture
def sample_data():
    meta = load_indicator_meta()
    df, report = load_data(SAMPLE_CSV_PATH, meta)
    return df, report, meta


def test_loader_sample_data(sample_data):
    df, report, meta = sample_data
    assert len(df) == 12
    assert df["district"].nunique() == 6
    assert df["month"].nunique() == 2
    assert set(report["indicators"]) == {
        "anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"
    }
    assert "head" in report and "info" in report and "missing" in report


def test_ahmedabad_anc_trend(sample_data):
    df, report, _ = sample_data
    indicators = report["indicators"]

    # At 10% threshold: Ahmedabad ANC (85 -> 69, -18.82%) should be flagged
    config_10 = AnalyticsConfig(trend_threshold_pct=10.0)
    trends_10 = detect_trends(df, config_10, indicators)
    
    ahmedabad_anc = trends_10[
        (trends_10["district"] == "Ahmedabad") & (trends_10["indicator"] == "anc_coverage")
    ]
    assert not ahmedabad_anc.empty
    row = ahmedabad_anc.iloc[0]
    assert pytest.approx(row["pct_change"], abs=0.1) == -18.82
    assert row["prev_value"] == 85.0
    assert row["value"] == 69.0

    # At 20% threshold: -18.82% should NOT be flagged
    config_20 = AnalyticsConfig(trend_threshold_pct=20.0)
    trends_20 = detect_trends(df, config_20, indicators)
    ahmedabad_anc_20 = trends_20[
        (trends_20["district"] == "Ahmedabad") & (trends_20["indicator"] == "anc_coverage")
    ]
    assert ahmedabad_anc_20.empty


def test_mehsana_high_risk_trend(sample_data):
    df, report, _ = sample_data
    config = AnalyticsConfig(trend_threshold_pct=10.0)
    trends = detect_trends(df, config, report["indicators"])
    
    mehsana_cases = trends[
        (trends["district"] == "Mehsana") & (trends["indicator"] == "high_risk_cases")
    ]
    assert not mehsana_cases.empty
    row = mehsana_cases.iloc[0]
    # 11 -> 28 is +154.55%
    assert pytest.approx(row["pct_change"], abs=0.1) == 154.55
    assert row["prev_value"] == 11.0
    assert row["value"] == 28.0


def test_mehsana_anc_outlier_iqr(sample_data):
    df, report, _ = sample_data
    config = AnalyticsConfig(outlier_method="IQR", iqr_multiplier=1.5)
    outliers = detect_outliers(df, config, report["indicators"])

    mehsana_anc = outliers[
        (outliers["district"] == "Mehsana") & (outliers["indicator"] == "anc_coverage")
    ]
    assert not mehsana_anc.empty
    row = mehsana_anc.iloc[0]
    assert row["value"] == 42.0
    # Q1 is ~78.5, Q3 is ~84.25, IQR is ~5.75, lower fence is ~69.875
    assert row["value"] < row["lower_fence"]


def test_ahmedabad_anc_outlier_iqr(sample_data):
    df, report, _ = sample_data
    config = AnalyticsConfig(outlier_method="IQR", iqr_multiplier=1.5)
    outliers = detect_outliers(df, config, report["indicators"])

    # 69 is below lower fence (~69.875)
    ahmedabad_anc = outliers[
        (outliers["district"] == "Ahmedabad") & (outliers["indicator"] == "anc_coverage") & (outliers["month"] == "2026-08")
    ]
    assert not ahmedabad_anc.empty
    assert ahmedabad_anc.iloc[0]["value"] == 69.0


def test_zscore_behavior(sample_data):
    df, report, _ = sample_data
    # At default z_threshold = 3.0, 42 has z ~ -2.83, so it does not reach 3.0
    config_z3 = AnalyticsConfig(outlier_method="Z-score", z_threshold=3.0)
    outliers_z3 = detect_outliers(df, config_z3, report["indicators"])
    mehsana_z3 = outliers_z3[
        (outliers_z3["district"] == "Mehsana") & (outliers_z3["indicator"] == "anc_coverage")
    ]
    assert mehsana_z3.empty

    # At configurable z_threshold = 2.5, it IS detected
    config_z25 = AnalyticsConfig(outlier_method="Z-score", z_threshold=2.5)
    outliers_z25 = detect_outliers(df, config_z25, report["indicators"])
    mehsana_z25 = outliers_z25[
        (outliers_z25["district"] == "Mehsana") & (outliers_z25["indicator"] == "anc_coverage")
    ]
    assert not mehsana_z25.empty


def test_correlations_and_matrix(sample_data):
    df, report, _ = sample_data
    config = AnalyticsConfig(corr_threshold=0.70)
    flagged, matrix, meta = detect_correlations(df, config, report["indicators"])

    assert matrix.shape == (4, 4)
    assert meta["n"] == 12
    assert meta["is_small_sample"] is True
    assert "fragile" in meta["warning"].lower()
    assert os.path.exists("outputs/correlation_matrix.csv")


def test_breaches_detection(sample_data):
    df, report, meta = sample_data
    config = AnalyticsConfig(breach_floor_pct=60.0, breach_ceiling_percentile=90.0)
    breaches = detect_breaches(df, config, report["indicators"], meta)

    # Mehsana ANC=42 is below floor 60
    anc_breach = breaches[
        (breaches["district"] == "Mehsana") & (breaches["indicator"] == "anc_coverage")
    ]
    assert not anc_breach.empty
    assert anc_breach.iloc[0]["value"] == 42.0


def test_severity_dynamic_mapping():
    config_default = AnalyticsConfig(severity_medium_x=1.0, severity_high_x=2.0)
    # Ratio 2.5 -> High, Ratio 1.5 -> Medium, Ratio 0.8 -> Low
    df_ratios = pd.DataFrame({"raw_ratio": [2.5, 1.5, 0.8]})
    result = assign_severity(df_ratios, config_default)
    assert result["severity"].tolist() == ["High", "Medium", "Low"]

    # Changing sliders alters categorization without code changes
    config_strict = AnalyticsConfig(severity_medium_x=2.0, severity_high_x=4.0)
    result_strict = assign_severity(df_ratios, config_strict)
    assert result_strict["severity"].tolist() == ["Medium", "Low", "Low"]


def test_insight_schema_and_generation(sample_data):
    df, report, meta = sample_data
    config = AnalyticsConfig()
    
    trends = detect_trends(df, config, report["indicators"])
    outliers = detect_outliers(df, config, report["indicators"])
    corr_flagged, _, _ = detect_correlations(df, config, report["indicators"])
    breaches = detect_breaches(df, config, report["indicators"], meta)

    insights = generate_insights(trends, outliers, corr_flagged, breaches, config, meta)
    
    expected_cols = [
        "insight_id", "type", "indicator", "entity", "period",
        "value", "prev_value", "change_pct", "severity", "explanation"
    ]
    assert list(insights.columns) == expected_cols
    assert not insights.empty
    assert insights["insight_id"].str.startswith("INS-").all()
    assert set(insights["severity"].unique()).issubset({"Low", "Medium", "High"})
    assert set(insights["type"].unique()).issubset({"trend", "outlier", "correlation", "threshold_breach"})


def test_edge_cases_no_crash():
    config = AnalyticsConfig()
    # 1. Division by zero check (previous value = 0)
    df_zero = pd.DataFrame({
        "month": ["2026-07", "2026-08"],
        "district": ["DistrictA", "DistrictA"],
        "indicator1": [0.0, 50.0]
    })
    trends_zero = detect_trends(df_zero, config, ["indicator1"])
    assert isinstance(trends_zero, pd.DataFrame)

    # 2. Constant column (std = 0, IQR = 0)
    df_const = pd.DataFrame({
        "month": ["2026-07", "2026-08", "2026-09", "2026-10"],
        "district": ["DistA", "DistA", "DistB", "DistB"],
        "indicator1": [10.0, 10.0, 10.0, 10.0]
    })
    outliers_const = detect_outliers(df_const, config, ["indicator1"])
    assert outliers_const.empty

    # 3. Single month input
    df_single = pd.DataFrame({
        "month": ["2026-07", "2026-07"],
        "district": ["DistA", "DistB"],
        "indicator1": [50.0, 60.0]
    })
    trends_single = detect_trends(df_single, config, ["indicator1"])
    assert trends_single.empty


def test_no_hardcoded_district_strings_in_engine():
    """
    CRITICAL STATIC ANALYSIS TEST:
    Ensures that no sample district names appear as string literals inside engine/*.py.
    Violating this is an automatic fail according to the assignment specification.
    """
    forbidden_districts = [
        "Ahmedabad", "Mehsana", "Surat", "Vadodara", "Rajkot", "Bhavnagar"
    ]
    engine_files = glob.glob("engine/*.py")
    assert len(engine_files) > 0, "No engine files found"

    violations = []
    for filepath in engine_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            for district in forbidden_districts:
                # Check case-insensitive match for forbidden district names
                if re.search(r'\b' + re.escape(district) + r'\b', content, re.IGNORECASE):
                    violations.append((filepath, district))

    assert not violations, f"Hardcoded district names found in engine code: {violations}"
