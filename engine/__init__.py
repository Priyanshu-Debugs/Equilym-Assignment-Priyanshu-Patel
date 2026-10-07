"""
Auto-Analytics Engine Package
Pure functional modules for data loading, trend detection, outlier analysis,
correlation analysis, threshold breach detection, severity calculation, and insight generation.
"""

from engine.config import AnalyticsConfig
from engine.loader import load_data
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import detect_correlations
from engine.breaches import detect_breaches
from engine.severity import assign_severity
from engine.insights import generate_insights

__all__ = [
    "AnalyticsConfig",
    "load_data",
    "detect_trends",
    "detect_outliers",
    "detect_correlations",
    "detect_breaches",
    "assign_severity",
    "generate_insights",
]
