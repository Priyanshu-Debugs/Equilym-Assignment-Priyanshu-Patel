"""
Part C — Outlier Detection.
Pure functional module to detect statistical outliers via pooled IQR or Z-score methods.
"""

from typing import List
import pandas as pd
import numpy as np
from engine.config import AnalyticsConfig


def detect_outliers(
    df: pd.DataFrame,
    config: AnalyticsConfig,
    indicators: List[str]
) -> pd.DataFrame:
    """
    Identifies outliers pooled across all district-months for each indicator.
    
    Supports:
    - IQR method: flags values outside [Q1 - k*IQR, Q3 + k*IQR]
    - Z-score method: flags values where |z| >= z_threshold
    """
    if df.empty or len(indicators) == 0:
        return pd.DataFrame()

    outliers: List[dict] = []

    for indicator in indicators:
        if indicator not in df.columns:
            continue
            
        series = df[["district", "month", indicator]].dropna().copy()
        n = len(series)
        
        # Guard: Need minimum data points for meaningful distribution estimation
        if n < 4:
            continue

        vals = series[indicator].to_numpy(dtype=float)

        if config.outlier_method == "IQR":
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            iqr = q3 - q1
            median_val = float(np.median(vals))
            
            # Guard against zero spread
            if iqr <= 0:
                continue

            k = config.iqr_multiplier
            lower_fence = q1 - (k * iqr)
            upper_fence = q3 + (k * iqr)

            for _, row in series.iterrows():
                val = float(row[indicator])
                is_outlier = False
                dist_beyond = 0.0

                if val < lower_fence:
                    is_outlier = True
                    dist_beyond = (lower_fence - val) / (k * iqr)
                elif val > upper_fence:
                    is_outlier = True
                    dist_beyond = (val - upper_fence) / (k * iqr)

                if is_outlier:
                    # Normalized ratio for severity: 1.0 at fence edge, higher as it extends
                    ratio = 1.0 + dist_beyond
                    outliers.append({
                        "district": row["district"],
                        "indicator": indicator,
                        "month": row["month"],
                        "value": val,
                        "score": dist_beyond,
                        "raw_ratio": float(ratio),
                        "reference": median_val,
                        "method": "IQR",
                        "lower_fence": lower_fence,
                        "upper_fence": upper_fence,
                        "z_score": None,
                        "iqr": iqr,
                        "q1": q1,
                        "q3": q3,
                    })

        elif config.outlier_method == "Z-score":
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1))

            # Guard against zero variance
            if std_val <= 0:
                continue

            z_thresh = config.z_threshold

            for _, row in series.iterrows():
                val = float(row[indicator])
                z = (val - mean_val) / std_val
                abs_z = abs(z)

                if abs_z >= z_thresh:
                    ratio = abs_z / max(z_thresh, 1e-6)
                    outliers.append({
                        "district": row["district"],
                        "indicator": indicator,
                        "month": row["month"],
                        "value": val,
                        "score": abs_z,
                        "raw_ratio": float(ratio),
                        "reference": mean_val,
                        "method": "Z-score",
                        "lower_fence": None,
                        "upper_fence": None,
                        "z_score": float(z),
                        "iqr": None,
                        "q1": None,
                        "q3": None,
                    })

    if not outliers:
        return pd.DataFrame(columns=[
            "district", "indicator", "month", "value", "score", "raw_ratio",
            "reference", "method", "lower_fence", "upper_fence", "z_score",
            "iqr", "q1", "q3"
        ])

    return pd.DataFrame(outliers)
