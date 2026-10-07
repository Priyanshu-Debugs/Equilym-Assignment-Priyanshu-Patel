"""
Part B — Trend Detection.
Pure functional module to detect significant month-over-month shifts and slopes.
"""

from typing import List
import pandas as pd
import numpy as np
from scipy import stats
from engine.config import AnalyticsConfig


def detect_trends(
    df: pd.DataFrame,
    config: AnalyticsConfig,
    indicators: List[str]
) -> pd.DataFrame:
    """
    Identifies significant month-over-month indicator shifts per district.
    
    Formula:
    pct_change = ((current - previous) / previous) * 100
    is_significant = abs(pct_change) >= config.trend_threshold_pct
    """
    if df.empty or len(indicators) == 0:
        return pd.DataFrame()

    results: List[dict] = []
    
    # Ensure ordered chronologically
    sorted_df = df.sort_values(by=["district", "month"]).copy()

    for district, group in sorted_df.groupby("district"):
        for indicator in indicators:
            valid_series = group[["month", indicator]].dropna().sort_values("month")
            n_points = len(valid_series)
            
            if n_points < config.min_months_for_trend:
                continue
            
            # Optional linear regression slope if >= 3 data points
            slope_val = None
            if n_points >= 3:
                try:
                    x_axis = np.arange(n_points)
                    y_axis = valid_series[indicator].values
                    reg = stats.linregress(x_axis, y_axis)
                    slope_val = float(reg.slope)
                except Exception:
                    slope_val = None

            # Evaluate consecutive monthly pairs (focusing especially on the latest transition)
            months = valid_series["month"].tolist()
            vals = valid_series[indicator].tolist()

            for i in range(1, len(vals)):
                prev_val = vals[i - 1]
                curr_val = vals[i]
                prev_month = months[i - 1]
                curr_month = months[i]

                if prev_val == 0 or np.isnan(prev_val) or np.isnan(curr_val):
                    continue

                pct_change = ((curr_val - prev_val) / abs(prev_val)) * 100.0
                abs_pct = abs(pct_change)

                if abs_pct >= config.trend_threshold_pct:
                    ratio = abs_pct / max(config.trend_threshold_pct, 1e-6)
                    period_str = f"{prev_month}..{curr_month}"

                    results.append({
                        "district": district,
                        "indicator": indicator,
                        "period": period_str,
                        "month": curr_month,
                        "prev_month": prev_month,
                        "value": float(curr_val),
                        "prev_value": float(prev_val),
                        "pct_change": float(pct_change),
                        "abs_change": float(curr_val - prev_val),
                        "is_significant": True,
                        "raw_ratio": float(ratio),
                        "slope": slope_val,
                    })

    if not results:
        return pd.DataFrame(columns=[
            "district", "indicator", "period", "month", "prev_month",
            "value", "prev_value", "pct_change", "abs_change", "is_significant",
            "raw_ratio", "slope"
        ])

    return pd.DataFrame(results)
