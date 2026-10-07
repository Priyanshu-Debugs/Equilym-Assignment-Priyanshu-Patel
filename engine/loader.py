"""
Part A — Data Loading and Validation.
Pure functional module to ingest, validate, and sanitize district-month healthcare data.
"""

import io
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np


def load_data(
    path_or_buffer: Any,
    indicator_meta: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ingests and validates district-month healthcare data.
    
    Parameters:
    - path_or_buffer: CSV filepath or file-like buffer (e.g., Streamlit upload).
    - indicator_meta: Optional metadata dict specifying unit and higher_is_better.
    
    Returns:
    - (df, validation_report): Cleaned DataFrame and validation details.
    """
    warnings: List[str] = []
    
    # 1. Load CSV
    df = pd.read_csv(path_or_buffer)
    
    # Strip whitespace from column names
    df.columns = [col.strip() for col in df.columns]
    
    # 2. Check required columns
    required_cols = {"month", "district"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns in dataset: {missing_required}")

    # 3. Discover numeric indicators dynamically (all columns other than month & district)
    indicator_candidates = [c for c in df.columns if c not in ["month", "district"]]
    if not indicator_candidates:
        raise ValueError("No numeric indicators found in dataset.")

    # 4. Standardize district: strip whitespace and title-case
    df["district"] = df["district"].astype(str).str.strip().str.title()

    # 5. Standardize month: parse to date and normalize to 'YYYY-MM'
    # Handles both '2026-07' and '2026-07-01'
    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.strftime("%Y-%m")
    if df["month"].isna().any():
        warnings.append("Some month values could not be parsed and were coerced to NaT.")

    # 6. Coerce indicators to numeric without crashing
    discovered_indicators: List[str] = []
    meta = indicator_meta or {}

    for col in indicator_candidates:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        discovered_indicators.append(col)
        
        # Check domain constraints per metadata if available
        col_meta = meta.get(col, {})
        is_pct = col_meta.get("unit") == "%" or ("rate" in col.lower() or "coverage" in col.lower() or "delivery" in col.lower() or "immunization" in col.lower())
        
        if is_pct:
            out_of_bounds = df[(df[col] < 0) | (df[col] > 100)][col]
            if not out_of_bounds.empty:
                warnings.append(
                    f"Column '{col}' has {len(out_of_bounds)} values outside percentage bounds [0, 100]."
                )
        else:
            negative_vals = df[df[col] < 0][col]
            if not negative_vals.empty:
                warnings.append(f"Column '{col}' has {len(negative_vals)} negative values.")

    # 7. Check duplicates on (district, month)
    duplicate_mask = df.duplicated(subset=["district", "month"], keep=False)
    if duplicate_mask.any():
        num_dupes = duplicate_mask.sum()
        warnings.append(
            f"Found {num_dupes} duplicate (district, month) records. Keeping the last record."
        )
        df = df.drop_duplicates(subset=["district", "month"], keep="last")

    # Sort deterministically
    df = df.sort_values(by=["district", "month"]).reset_index(drop=True)

    # 8. Capture diagnostics: head(), info(), isna().sum()
    head_str = df.head().to_string()

    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()

    missing_series = df.isna().sum()
    missing_dict = missing_series.to_dict()

    validation_report: Dict[str, Any] = {
        "head": head_str,
        "info": info_str,
        "missing": missing_dict,
        "warnings": warnings,
        "indicators": discovered_indicators,
        "row_count": len(df),
        "district_count": df["district"].nunique(),
        "month_count": df["month"].nunique(),
    }

    return df, validation_report
