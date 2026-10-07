"""
Threshold Breach Detection.
Identifies indicator values that cross predefined policy floors or empirical distribution ceilings.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from engine.config import AnalyticsConfig


def detect_breaches(
    df: pd.DataFrame,
    config: AnalyticsConfig,
    indicators: List[str],
    indicator_meta: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Flags rows that breach policy floors (for percentage coverage metrics)
    or empirical ceilings (for counts or inverse indicators).
    """
    if df.empty or len(indicators) == 0:
        return pd.DataFrame()

    meta = indicator_meta or {}
    breaches: List[dict] = []

    for indicator in indicators:
        if indicator not in df.columns:
            continue

        series = df[["district", "month", indicator]].dropna().copy()
        if series.empty:
            continue

        ind_info = meta.get(indicator, {})
        higher_is_better = ind_info.get("higher_is_better", True)
        unit = ind_info.get("unit", "%")

        vals = series[indicator].to_numpy(dtype=float)

        if higher_is_better and unit == "%":
            # Policy floor breach: value falls below minimum acceptable threshold
            floor_limit = config.breach_floor_pct
            for _, row in series.iterrows():
                val = float(row[indicator])
                if val < floor_limit:
                    gap = floor_limit - val
                    # Ratio: relative distance below floor
                    ratio = 1.0 + (gap / max(floor_limit, 1.0))
                    breaches.append({
                        "district": row["district"],
                        "indicator": indicator,
                        "month": row["month"],
                        "value": val,
                        "limit": floor_limit,
                        "limit_type": "policy floor",
                        "direction": "below",
                        "gap": gap,
                        "raw_ratio": float(ratio),
                    })
        else:
            # Ceiling breach: value exceeds empirical upper percentile threshold
            if len(vals) < 4:
                continue
            ceiling_limit = float(np.percentile(vals, config.breach_ceiling_percentile))
            for _, row in series.iterrows():
                val = float(row[indicator])
                if val > ceiling_limit:
                    gap = val - ceiling_limit
                    ratio = 1.0 + (gap / max(ceiling_limit, 1.0))
                    breaches.append({
                        "district": row["district"],
                        "indicator": indicator,
                        "month": row["month"],
                        "value": val,
                        "limit": ceiling_limit,
                        "limit_type": "percentile ceiling",
                        "direction": "above",
                        "gap": gap,
                        "raw_ratio": float(ratio),
                    })

    if not breaches:
        return pd.DataFrame(columns=[
            "district", "indicator", "month", "value", "limit",
            "limit_type", "direction", "gap", "raw_ratio"
        ])

    return pd.DataFrame(breaches)
