"""
Configuration dataclass and metadata loader for the Auto-Analytics Engine.
All user-configurable thresholds and detection parameters are centralized here.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Dict, Any, Optional


@dataclass
class AnalyticsConfig:
    # Trend Detection Parameters
    trend_threshold_pct: float = 10.0
    min_months_for_trend: int = 2

    # Outlier Detection Parameters
    outlier_method: str = "IQR"  # "IQR" or "Z-score"
    iqr_multiplier: float = 1.5
    z_threshold: float = 3.0

    # Correlation Detection Parameters
    corr_threshold: float = 0.70

    # Threshold Breach Parameters
    breach_floor_pct: float = 60.0
    breach_ceiling_percentile: float = 90.0

    # Severity Threshold Multipliers (Data-Derived Ratio)
    severity_medium_x: float = 1.0
    severity_high_x: float = 2.0


def load_indicator_meta(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads indicator metadata specifying unit and directionality.
    Defaults to unit='%' and higher_is_better=True for unlisted indicators.
    """
    if config_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(base_dir, "config", "indicator_meta.json")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
