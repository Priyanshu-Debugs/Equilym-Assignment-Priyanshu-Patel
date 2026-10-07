"""
Part D — Correlation Detection.
Pure functional module to compute Pearson correlation matrices, flag significant pairs,
and state statistical caveats for small-sample stability.
"""

from typing import List, Tuple, Dict, Any
import os
import pandas as pd
import numpy as np
from scipy import stats
from engine.config import AnalyticsConfig


def detect_correlations(
    df: pd.DataFrame,
    config: AnalyticsConfig,
    indicators: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Computes Pearson correlation matrix across numerical indicators and flags significant pairs.
    
    Returns:
    - flagged_df: DataFrame of flagged indicator pairs meeting |r| >= corr_threshold
    - corr_matrix: Standard pandas DataFrame.corr() matrix
    - metadata: Dict containing sample size, warnings, and limitations
    """
    if df.empty or len(indicators) < 2:
        empty_matrix = pd.DataFrame()
        empty_flagged = pd.DataFrame(columns=[
            "indicator_a", "indicator_b", "r_value", "p_value", "n",
            "entity", "raw_ratio", "is_robust", "caveat"
        ])
        return empty_flagged, empty_matrix, {"n": 0, "caveat": "Insufficient indicators or data"}

    # Drop NaNs across the evaluated indicators for synchronized pairwise analysis
    clean_df = df[indicators].dropna()
    n_sample = len(clean_df)

    corr_matrix = clean_df.corr(method="pearson")

    # Save to outputs directory if accessible
    try:
        os.makedirs("outputs", exist_ok=True)
        corr_matrix.to_csv("outputs/correlation_matrix.csv")
    except Exception:
        pass

    flagged_pairs: List[dict] = []
    
    # Check if data represents a single district or pooled districts
    districts = df["district"].dropna().unique() if "district" in df.columns else []
    entity_label = districts[0] if len(districts) == 1 else "All districts (pooled)"

    # Small sample caveat text
    small_sample_warning = None
    if n_sample < 30:
        small_sample_warning = (
            f"Correlation computed on n={n_sample} observations; fragile and susceptible to leverage points."
        )

    # Scan upper-triangle
    num_indicators = len(indicators)
    for i in range(num_indicators):
        for j in range(i + 1, num_indicators):
            ind_a = indicators[i]
            ind_b = indicators[j]

            series_a = clean_df[ind_a].to_numpy(dtype=float)
            series_b = clean_df[ind_b].to_numpy(dtype=float)

            # Check for non-zero variance
            if np.std(series_a) <= 0 or np.std(series_b) <= 0 or len(series_a) < 3:
                continue

            r_val, p_val = stats.pearsonr(series_a, series_b)

            if abs(r_val) >= config.corr_threshold:
                # Severity ratio: |r| / threshold
                ratio = abs(r_val) / max(config.corr_threshold, 1e-6)

                # Leave-one-out robustness check
                is_robust = True
                loo_corrs = []
                for idx in range(len(series_a)):
                    sub_a = np.delete(series_a, idx)
                    sub_b = np.delete(series_b, idx)
                    if np.std(sub_a) > 0 and np.std(sub_b) > 0:
                        sub_r, _ = stats.pearsonr(sub_a, sub_b)
                        loo_corrs.append(sub_r)

                if loo_corrs:
                    # If removing a single point drops below 0.50 or flips sign, it is fragile
                    min_loo = min(abs(c) for c in loo_corrs)
                    if min_loo < 0.50 or any(np.sign(c) != np.sign(r_val) for c in loo_corrs):
                        is_robust = False

                flagged_pairs.append({
                    "indicator_a": ind_a,
                    "indicator_b": ind_b,
                    "r_value": float(r_val),
                    "p_value": float(p_val),
                    "n": int(n_sample),
                    "entity": entity_label,
                    "raw_ratio": float(ratio),
                    "is_robust": is_robust,
                    "caveat": small_sample_warning or "",
                })

    flagged_df = pd.DataFrame(flagged_pairs) if flagged_pairs else pd.DataFrame(columns=[
        "indicator_a", "indicator_b", "r_value", "p_value", "n",
        "entity", "raw_ratio", "is_robust", "caveat"
    ])

    meta = {
        "n": n_sample,
        "is_small_sample": n_sample < 30,
        "warning": small_sample_warning,
    }

    return flagged_df, corr_matrix, meta
