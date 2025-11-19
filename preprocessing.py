"""
Data preprocessing pipelines for Stack Overflow survey analysis.

This module provides two main preprocessing functions:
- preprocess_eur(): Prepares data for EUR compensation regression analysis
- preprocess_aiaccbin(): Prepares data for AI trust/distrust classification

Both functions handle missing values, outliers, plausibility checks, and feature
engineering with comprehensive data loss logging at each step.
"""
import pandas as pd

from utils import AGE_MAP, ED_MAP

# Preprocessing constants
MAX_WORK_EXPERIENCE_YEARS = 60  # Maximum plausible years of work experience
MAX_YEARS_CODE = 60  # Maximum plausible years of coding experience
IQR_MULTIPLIER = 1.5  # Standard multiplier for IQR outlier detection
DEFAULT_TOP_N_CATEGORIES = 10  # Default number of top categories to keep


def preprocess_eur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processing per la regressione su EUR ed EDA rilevante:
    - Filtra `Currency` che inizia con "EUR"
    - Mantiene `WorkExp`, `CompTotal`, `Country`, `Employment`, `EdLevel`, `Industry`
    - Converte WorkExp/CompTotal a numerico e rimuove NaN
    - Plausibilità: 0 < WorkExp ≤ 60, CompTotal > 0
    - Rimuove outlier alti su `CompTotal` (1.5×IQR)
    - Rimuove duplicati
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    required_cols = ["Currency", "WorkExp", "CompTotal", "Country", "Employment", "EdLevel", "Industry"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    df = df[df["Currency"].astype(str).str.startswith("EUR")].copy()
    print(f"[preprocess_eur] Filtered to EUR currency: {len(df)} rows")

    numeric_cols = ["WorkExp", "CompTotal"]
    cat_cols = ["Country", "Employment", "EdLevel", "Industry"]
    cols = numeric_cols + cat_cols
    df = df[cols].copy()

    print(f"[preprocess_eur] Starting with {len(df)} rows")
    for c in numeric_cols:
        before = len(df)
        before_valid = df[c].notna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        after_valid = df[c].notna().sum()
        coerced = before_valid - after_valid
        if coerced > 0:
            print(f"  {c}: {coerced} values coerced to NaN (non-numeric)")

    before_dropna = len(df)
    df = df.dropna(subset=numeric_cols)
    dropped_na = before_dropna - len(df)
    print(f"  Dropped {dropped_na} rows with NaN in numeric columns")

    before_bounds = len(df)
    df = df[(df["WorkExp"] > 0) & (df["WorkExp"] <= MAX_WORK_EXPERIENCE_YEARS)]
    df = df[df["CompTotal"] > 0]
    dropped_bounds = before_bounds - len(df)
    print(f"  Dropped {dropped_bounds} rows outside plausibility bounds")

    q1 = df["CompTotal"].quantile(0.25)
    q3 = df["CompTotal"].quantile(0.75)
    iqr = q3 - q1
    high = q3 + IQR_MULTIPLIER * iqr
    before_iqr = len(df)
    df = df[df["CompTotal"] <= high]
    dropped_outliers = before_iqr - len(df)
    print(f"  Dropped {dropped_outliers} high outliers (IQR method)")

    before_dedup = len(df)
    df = df.drop_duplicates()
    dropped_dups = before_dedup - len(df)
    print(f"  Dropped {dropped_dups} duplicate rows")
    print(f"[preprocess_eur] Final dataset: {len(df)} rows")

    # Output validation
    if len(df) == 0:
        raise ValueError("Preprocessing resulted in empty DataFrame. Try less strict filtering.")

    return df


def preprocess_aiaccbin(
    df: pd.DataFrame,
    industry_top_n: int = DEFAULT_TOP_N_CATEGORIES,
    devtype_top_n: int = DEFAULT_TOP_N_CATEGORIES
) -> pd.DataFrame:
    """
    Pre-processing per classificazione AIAcc → AIAccBin:
    - Filtra AIAcc su quattro etichette (trust/distrust) e crea AIAccBin {0,1}
    - Converte WorkExp/YearsCode a numerico e applica bound di plausibilità
    - Mappa Age→AgeMapped e EdLevel→EdLevelOrd e rimuove NaN risultanti
    - Riduce cardinalità: IndustryTop/DevTypeTop con top-10
    - Deduplica
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if industry_top_n < 1:
        raise ValueError(f"industry_top_n must be >= 1, got {industry_top_n}")

    if devtype_top_n < 1:
        raise ValueError(f"devtype_top_n must be >= 1, got {devtype_top_n}")

    required_cols = ["AIAcc", "WorkExp", "YearsCode", "Age", "EdLevel"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty")

    trust = {"Highly trust", "Somewhat trust"}
    distrust = {"Highly distrust", "Somewhat distrust"}
    df = df[df["AIAcc"].isin(trust | distrust)].copy()
    df["AIAccBin"] = (df["AIAcc"].isin(trust)).astype(int)
    print(f"[preprocess_aiaccbin] Filtered to trust/distrust responses: {len(df)} rows")

    print(f"[preprocess_aiaccbin] Starting numeric conversion")
    for c in ["WorkExp", "YearsCode"]:
        before_valid = df[c].notna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        after_valid = df[c].notna().sum()
        coerced = before_valid - after_valid
        if coerced > 0:
            print(f"  {c}: {coerced} values coerced to NaN")

    before_dropna = len(df)
    df = df.dropna(subset=["WorkExp", "YearsCode"]).copy()
    dropped_na = before_dropna - len(df)
    print(f"  Dropped {dropped_na} rows with NaN in WorkExp/YearsCode")

    before_bounds = len(df)
    df = df[(df["WorkExp"] > 0) & (df["WorkExp"] <= MAX_WORK_EXPERIENCE_YEARS)]
    df = df[(df["YearsCode"] >= 0) & (df["YearsCode"] <= MAX_YEARS_CODE)]
    dropped_bounds = before_bounds - len(df)
    print(f"  Dropped {dropped_bounds} rows outside plausibility bounds")

    before_mapping = len(df)
    df["AgeMapped"] = df["Age"].map(AGE_MAP)
    df["EdLevelOrd"] = df["EdLevel"].map(ED_MAP)
    df = df.dropna(subset=["AgeMapped", "EdLevelOrd"]).copy()
    dropped_mapping = before_mapping - len(df)
    print(f"  Dropped {dropped_mapping} rows with unmapped Age/EdLevel")

    if "Industry" in df.columns:
        top_ind = df["Industry"].value_counts().index[:industry_top_n]
        df["IndustryTop"] = df["Industry"].where(df["Industry"].isin(top_ind), "Other")
    if "DevType" in df.columns:
        top_dt = df["DevType"].value_counts().index[:devtype_top_n]
        df["DevTypeTop"] = df["DevType"].where(df["DevType"].isin(top_dt), "Other")

    before_dedup = len(df)
    df = df.drop_duplicates()
    dropped_dups = before_dedup - len(df)
    print(f"  Dropped {dropped_dups} duplicate rows")
    print(f"[preprocess_aiaccbin] Final dataset: {len(df)} rows")

    # Output validation
    if len(df) == 0:
        raise ValueError("Preprocessing resulted in empty DataFrame. Try less strict filtering.")

    return df
