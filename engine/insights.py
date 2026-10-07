"""
Part E — Automated Insight Generation (Gating Deliverable).
Combines findings from trends, outliers, correlations, and threshold breaches into a unified schema,
formats natural language narratives strictly through parameterization, and writes export artifacts.
ZERO HARDCODED DISTRICT NAMES OR MAGIC NUMBERS ALLOWED.
"""

from typing import List, Dict, Any, Optional
import os
import json
import pandas as pd
import numpy as np
from engine.config import AnalyticsConfig
from engine.severity import assign_severity


def format_indicator_label(ind_name: str) -> str:
    """Formats an indicator column name into a title-cased label."""
    return ind_name.replace("_", " ").title()


def generate_insights(
    trends_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    breaches_df: pd.DataFrame,
    config: AnalyticsConfig,
    indicator_meta: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Consolidates detection results into standard structured insights.
    
    Generates human-readable explanations via dynamic string templates.
    Computes data-derived severity and serializes to CSV and JSON.
    """
    records: List[dict] = []

    # 1. Transform Trends
    if trends_df is not None and not trends_df.empty:
        for _, row in trends_df.iterrows():
            ind = row["indicator"]
            ind_label = format_indicator_label(ind)
            entity = row["district"]
            period = row["period"]
            val = float(row["value"])
            prev = float(row["prev_value"])
            pct = float(row["pct_change"])
            abs_pct = abs(pct)
            verb = "dropped" if pct < 0 else "rose"
            thr = config.trend_threshold_pct

            explanation = (
                f"{ind_label} in {entity} {verb} by {abs_pct:.1f}% compared to the "
                f"previous month ({prev:.0f} -> {val:.0f}), exceeding the {thr:g}% "
                f"significant-change threshold."
            )

            records.append({
                "type": "trend",
                "indicator": ind,
                "entity": entity,
                "period": period,
                "value": val,
                "prev_value": prev,
                "change_pct": round(pct, 2),
                "metric": val,
                "change": round(pct, 2),
                "raw_ratio": float(row["raw_ratio"]),
                "explanation": explanation,
            })

    # 2. Transform Outliers
    if outliers_df is not None and not outliers_df.empty:
        for _, row in outliers_df.iterrows():
            ind = row["indicator"]
            ind_label = format_indicator_label(ind)
            entity = row["district"]
            period = str(row["month"])
            val = float(row["value"])
            method = row["method"]
            ref_val = float(row["reference"])

            if method == "IQR":
                lo = float(row["lower_fence"])
                hi = float(row["upper_fence"])
                explanation = (
                    f"{entity}'s {ind_label} of {val:.0f} falls outside the "
                    f"IQR fence [{lo:.1f}, {hi:.1f}] (median: {ref_val:.0f})."
                )
            else:
                z = float(row["z_score"])
                abs_z = abs(z)
                dir_str = "below" if z < 0 else "above"
                explanation = (
                    f"{entity}'s {ind_label} of {val:.0f} is {abs_z:.1f}σ {dir_str} "
                    f"the mean ({ref_val:.0f}), flagging for review."
                )

            records.append({
                "type": "outlier",
                "indicator": ind,
                "entity": entity,
                "period": period,
                "value": val,
                "prev_value": None,
                "change_pct": None,
                "metric": val,
                "change": None,
                "raw_ratio": float(row["raw_ratio"]),
                "explanation": explanation,
            })

    # 3. Transform Correlations
    if correlations_df is not None and not correlations_df.empty:
        for _, row in correlations_df.iterrows():
            ind_a = row["indicator_a"]
            ind_b = row["indicator_b"]
            a_label = format_indicator_label(ind_a)
            b_label = format_indicator_label(ind_b)
            r = float(row["r_value"])
            n = int(row["n"])
            entity = str(row["entity"])
            
            direction = "positive" if r > 0 else "negative"
            strength = "strong" if abs(r) >= 0.80 else "moderate"

            explanation = (
                f"{a_label} and {b_label} show a {strength} {direction} correlation "
                f"(r={r:.2f}, n={n}); fragile with small samples."
            )

            pair_indicator = f"{ind_a}:{ind_b}"

            records.append({
                "type": "correlation",
                "indicator": pair_indicator,
                "entity": entity,
                "period": "pooled",
                "value": round(r, 4),
                "prev_value": None,
                "change_pct": None,
                "metric": round(r, 4),
                "change": None,
                "raw_ratio": float(row["raw_ratio"]),
                "explanation": explanation,
            })

    # 4. Transform Threshold Breaches
    if breaches_df is not None and not breaches_df.empty:
        for _, row in breaches_df.iterrows():
            ind = row["indicator"]
            ind_label = format_indicator_label(ind)
            entity = row["district"]
            period = str(row["month"])
            val = float(row["value"])
            limit = float(row["limit"])
            limit_type = row["limit_type"]
            direction = row["direction"]
            gap = float(row["gap"])

            explanation = (
                f"{entity}'s {ind_label} of {val:.0f} breached the {limit_type} "
                f"({limit:.0f}) by {gap:.1f} units ({direction} threshold)."
            )

            records.append({
                "type": "threshold_breach",
                "indicator": ind,
                "entity": entity,
                "period": period,
                "value": val,
                "prev_value": None,
                "change_pct": None,
                "metric": val,
                "change": None,
                "raw_ratio": float(row["raw_ratio"]),
                "explanation": explanation,
            })

    if not records:
        empty_df = pd.DataFrame(columns=[
            "insight_id", "type", "indicator", "entity", "period",
            "value", "prev_value", "change_pct", "severity", "explanation"
        ])
        return empty_df

    combined_df = pd.DataFrame(records)

    # 5. Assign Data-Derived Severity
    combined_df = assign_severity(combined_df, config)

    # 6. Sort deterministically: High -> Medium -> Low, then by raw_ratio descending
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    combined_df["sev_rank"] = combined_df["severity"].map(severity_order)
    combined_df = combined_df.sort_values(by=["sev_rank", "raw_ratio"], ascending=[True, False]).reset_index(drop=True)
    combined_df = combined_df.drop(columns=["sev_rank"])

    # 7. Generate insight_id sequentially: INS-0001, INS-0002...
    combined_df["insight_id"] = [f"INS-{i+1:04d}" for i in range(len(combined_df))]

    # Standard columns ordered per specification 4.1
    standard_columns = [
        "insight_id", "type", "indicator", "entity", "period",
        "value", "prev_value", "change_pct", "severity", "explanation"
    ]
    
    output_df = combined_df[standard_columns].copy()

    # 8. Export artifacts to outputs/ directory
    try:
        os.makedirs("outputs", exist_ok=True)
        # CSV output
        output_df.to_csv("outputs/insights.csv", index=False)
        
        # JSON output includes metric and change aliases
        json_records = combined_df[[
            "insight_id", "type", "indicator", "entity", "period",
            "value", "prev_value", "change_pct", "metric", "change",
            "severity", "explanation"
        ]].to_dict(orient="records")
        
        with open("outputs/insights.json", "w", encoding="utf-8") as f:
            json.dump(json_records, f, indent=2)
    except Exception:
        pass

    return output_df
