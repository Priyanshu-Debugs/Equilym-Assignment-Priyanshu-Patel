"""
Data-Derived Severity Assignment.
Maps continuous statistical magnitude ratios to categorical severity levels (Low, Medium, High).
No magic numbers or arbitrary thresholds.
"""

from typing import Union
import pandas as pd
from engine.config import AnalyticsConfig


def calculate_severity_level(
    ratio: float,
    config: AnalyticsConfig
) -> str:
    """
    Categorizes a continuous magnitude ratio into Low, Medium, or High.
    
    Rule:
    - ratio < severity_medium_x -> 'Low'
    - ratio < severity_high_x -> 'Medium'
    - ratio >= severity_high_x -> 'High'
    """
    if ratio < config.severity_medium_x:
        return "Low"
    elif ratio < config.severity_high_x:
        return "Medium"
    else:
        return "High"


def assign_severity(
    insights_df: pd.DataFrame,
    config: AnalyticsConfig
) -> pd.DataFrame:
    """
    Appends a data-derived 'severity' column to the insights DataFrame.
    """
    if insights_df.empty or "raw_ratio" not in insights_df.columns:
        if not insights_df.empty:
            insights_df["severity"] = "Medium"
        return insights_df

    df = insights_df.copy()
    df["severity"] = df["raw_ratio"].apply(lambda r: calculate_severity_level(r, config))
    return df
