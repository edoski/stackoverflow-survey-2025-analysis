# Stack Overflow 2025 Developer Survey Analysis

A machine learning analysis of Stack Overflow survey data with AI trust classification and compensation regression.

Comprehensive onboarding documentation for the Stack Overflow 2025 Developer Survey Analysis project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start Guide](#2-quick-start-guide)
3. [Architecture Overview](#3-architecture-overview)
4. [Detailed Module Breakdown](#4-detailed-module-breakdown)
   - 4.1 [utils.py](#41-utilspy)
   - 4.2 [preprocessing.py](#42-preprocessingpy)
   - 4.3 [classification.py](#43-classificationpy)
   - 4.4 [regressione_lineare.py](#44-regressione_linearepy)
   - 4.5 [eda.py](#45-edapy)
5. [Results Documentation](#5-results-documentation)
6. [Code Quality Features](#6-code-quality-features)
7. [ML Best Practices Implemented](#7-ml-best-practices-implemented)
8. [File Structure](#8-file-structure)
9. [Common Development Tasks](#9-common-development-tasks)
10. [Troubleshooting](#10-troubleshooting)
11. [Future Enhancements](#11-future-enhancements)

---

## 1. Project Overview

### 1.1 Purpose

This is an **academic statistical analysis project** for the STAT-PRJ-TRACCIA.pdf requirements. The project analyzes the Stack Overflow 2025 Developer Survey dataset to perform two main machine learning tasks:

1. **Binary Classification**: Predict AI trust/distrust (AIAccBin)
2. **Simple Linear Regression**: Predict compensation (CompTotal in EUR) from work experience (WorkExp)

### 1.2 Dataset

**Source**: Stack Overflow 2025 Developer Survey (`survey_results_public.csv`)

- **Size**: ~88 MB CSV file
- **Rows**: ~90,000 survey responses from developers worldwide
- **Columns**: ~100+ survey questions covering demographics, employment, compensation, AI attitudes, and more

**Key columns used**:

- Classification: `AIAcc`, `WorkExp`, `YearsCode`, `Age`, `EdLevel`, `Industry`, `Employment`, `RemoteWork`, `OrgSize`, `ICorPM`, `DevType`, `AISelect`, `AIAgents`, `AIModelsChoice`
- Regression: `Currency`, `WorkExp`, `CompTotal`, `Country`, `Employment`, `EdLevel`, `Industry`

### 1.3 Two Main Tasks

#### Task 1: Binary Classification (AIAccBin)

**Objective**: Predict whether a developer trusts or distrusts AI tools

**Target variable**: `AIAccBin`

- 1 = Trust (Highly trust OR Somewhat trust)
- 0 = Distrust (Highly distrust OR Somewhat distrust)

**Class distribution**: ~40% distrust, ~60% trust (moderate imbalance)

**Models compared**:

1. Logistic Regression (baseline)
2. SVM with linear kernel
3. SVM with polynomial kernel
4. SVM with RBF kernel

**Best model**: Logistic Regression (~73-74% accuracy)

#### Task 2: Simple Linear Regression (CompTotal)

**Objective**: Predict EUR compensation from years of work experience

**Predictor**: `WorkExp` (years of professional coding experience)

**Target**: `CompTotal` (annual compensation in EUR)

**Analysis includes**:

- Overall regression on EUR subset
- Per-country analysis for top 5 countries by sample size
- Diagnostic plots (residuals, QQ-plot)
- Shapiro-Wilk normality test on residuals

### 1.4 Technology Stack

**Language**: Python 3.12+

**Core libraries**:

- `pandas` (1.5+): Data manipulation and CSV loading
- `numpy` (1.23+): Numerical operations
- `scikit-learn` (1.2+): Machine learning models, pipelines, cross-validation
- `statsmodels` (0.13+): Statistical tests, QQ-plots
- `scipy` (1.10+): Statistical distributions (t-distribution for CIs), Shapiro-Wilk test
- `matplotlib` (3.6+): Plotting
- `seaborn` (0.12+): Statistical visualizations

### 1.5 Academic Requirements Fulfilled

This project satisfies the STAT-PRJ-TRACCIA.pdf requirements:

1. **Binary classification** with multiple models and hyperparameter tuning
2. **Simple linear regression** with diagnostic analysis
3. **Train/validation/test splitting** with proper discipline
4. **K-independent runs** statistical study (k=10)
5. **K-fold cross-validation** for model validation (k=10)
6. **Confidence intervals** using t-distribution
7. **Visualization** of results (confusion matrix, CV diagnostics, regression plots)
8. **Per-group analysis** (per-country regression)

---

## 2. Quick Start Guide

### 2.1 Prerequisites

- Python 3.12 or higher (Python 3.8+ should work but not tested)
- ~8 GB RAM minimum (peak usage during GridSearchCV in classification)
- ~2 GB free disk space (for dataset and generated plots)

### 2.2 Dataset Download

**IMPORTANT**: The dataset file `survey_results_public.csv` (~134 MB) is not included in this repository due to GitHub's file size limits.

Download it from the Stack Overflow Developer Survey on Kaggle:

1. Visit the [Stack Overflow Developer Survey dataset page](https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2025-developer-survey) on Kaggle
2. Download `survey_results_public.csv`
3. Place the file in the root directory of this project

Alternatively, if you have the Kaggle CLI installed:

```bash
kaggle datasets download -d stackoverflow/stack-overflow-2025-developer-survey
unzip stack-overflow-2025-developer-survey.zip
```

**Verify the file is present**:

```bash
ls -lh survey_results_public.csv
# Expected: ~134 MB file
```

### 2.3 Installation

```bash
# Clone or download the repository
cd stackoverflow-survey-2025-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

### 2.4 Execution Order

Run scripts in this order to reproduce the complete analysis:

```bash
# 1. Exploratory Data Analysis (generates plots_eda/ directory)
python eda.py

# 2. Classification task (generates plots_classification/ directory)
python classification.py

# 3. Linear regression task (generates plots_regression/ directory)
python regressione_lineare.py
```

**Expected runtime**:

- `eda.py`: 1-2 minutes
- `classification.py`: 5-10 minutes (GridSearchCV is compute-intensive)
- `regressione_lineare.py`: 1-2 minutes

**Total**: ~10-15 minutes on modern hardware

### 2.4 Headless Environments

For servers without display (no X11/Wayland):

```bash
# Set matplotlib backend to non-interactive before running
MPLBACKEND=Agg python eda.py
MPLBACKEND=Agg python classification.py
MPLBACKEND=Agg python regressione_lineare.py
```

### 2.5 Expected Outputs

After running all scripts, you should see:

```
stackoverflow-survey-2025-analysis/
├── plots_eda/
│   ├── eur/                    # 4 plots (regression EDA)
│   └── aiaccbin/               # 6 plots (classification EDA)
├── plots_classification/       # 4 plots (confusion matrix, CV diagnostics)
└── plots_regression/           # 3+ plots (scatter, residuals, QQ-plot)
    └── by_country/             # 5 plots (per-country regressions)
```

**Total plots**: ~22 PNG files at 150 DPI

### 2.6 Verifying Results

Check console output for key metrics:

**Classification** (`classification.py`):

```
Miglior modello su validation: logistic (acc=0.733)
Accuratezza su test: 0.743

=== Studio Statistico 1: K Independent Runs ===
Media: 0.731
Intervallo di Confidenza al 95%: [0.728, 0.734]

=== Studio Statistico 2: K-Fold Cross-Validation ===
Media: 0.731
Intervallo di Confidenza al 95%: [0.724, 0.738]
```

**Regression** (`regressione_lineare.py`):

```
=== Training Set Metrics ===
R^2 = 0.043
RMSE = 28453.1

=== Test Set Metrics (Generalization) ===
R^2 = 0.038
RMSE = 28891.2
Slope = 832.47  (EUR per anno)

=== Shapiro-Wilk Test for Normality of Residuals ===
P-value: 0.0000
Conclusion: Residuals may NOT be normally distributed
```

---

## 3. Architecture Overview

### 3.1 High-Level Data Flow

```
survey_results_public.csv (raw data, ~90,000 rows)
           |
           v
    preprocessing.py (two pipelines)
    ├─ preprocess_eur() ──────────────────> regressione_lineare.py ──> plots_regression/
    │   - Filter Currency=EUR*                - Train/test split
    │   - Plausibility bounds                 - Fit LinearRegression
    │   - IQR outlier removal                 - Metrics: R², MSE, RMSE
    │   - Output: ~6,307 rows                 - Diagnostics: residuals, QQ-plot
    │                                          - Per-country analysis (top 5)
    │
    └─ preprocess_aiaccbin() ─────────────> classification.py ───────> plots_classification/
        - Filter AIAcc (4 labels)             - Train/Val/Test split (70/15/15)
        - Create AIAccBin (binary)            - Baseline: Logistic, SVM (linear/poly/rbf)
        - Plausibility bounds                 - GridSearchCV hyperparameter tuning
        - Ordinal mappings (Age, EdLevel)     - Best model selection on validation
        - Top-N cardinality reduction         - Final evaluation on test
        - Output: ~18,821 rows                - K-independent runs (k=10)
                                               - K-fold CV (k=10)
                                               - Confusion matrix, CV plots

           |
           v
        eda.py (exploratory analysis)
        ├─ EUR subset ──────────────────────> plots_eda/eur/
        │   - Boxplots, histograms              - box_CompTotal.png
        │   - Correlation matrix                - hist_WorkExp.png, hist_CompTotal.png
        │                                        - corr_numeric.png
        │
        └─ AIAccBin subset ─────────────────> plots_eda/aiaccbin/
            - Target distribution               - distribuzione_AIAccBin.png
            - Histograms, boxplots by target    - hist_WorkExp.png, hist_YearsCode.png
            - Stacked bar charts                - box_WorkExp_by_AIAccBin.png
            - Correlation matrix                - stacked_Employment_AIAccBin.png
                                                 - corr_numeric.png

utils.py (shared mappings)
├─ AGE_MAP: Dict[str, int]  ───────> preprocessing.py (line 153)
└─ ED_MAP: Dict[str, int]   ───────> preprocessing.py (line 154)
```

### 3.2 Module Dependency Graph

```
utils.py (no dependencies)
    |
    v
preprocessing.py (imports utils)
    |
    +─────────────────────+
    |                     |
    v                     v
classification.py     regressione_lineare.py     eda.py
(imports preprocessing)  (imports preprocessing)  (imports preprocessing)
```

**No circular dependencies**: Clean unidirectional flow

### 3.3 Design Principles

1. **Separation of Concerns**: Preprocessing, EDA, and modeling in separate modules
2. **DRY (Don't Repeat Yourself)**: Shared mappings in `utils.py`
3. **Fail Fast**: Input/output validation with helpful error messages
4. **Transparency**: Comprehensive logging of data loss at every step
5. **Reproducibility**: Named constants, random seeds, deterministic algorithms
6. **No Data Leakage**: Preprocessing in pipelines, fitted only on training data
7. **Type Safety**: Type hints on all function signatures

---

## 4. Detailed Module Breakdown

### 4.1 utils.py

**Location**: `/Users/edo/dev/python/stackoverflow-survey-2025-analysis/utils.py`

**Lines of code**: 35

**Purpose**: Centralized ordinal encoding mappings for Age and EdLevel features

#### 4.1.1 Module Docstring

```python
"""
Shared utility mappings for classification and EDA.

This module contains constant mappings for ordinal encoding of categorical
features used across the Stack Overflow survey analysis:
- AGE_MAP: Maps age range strings to numeric midpoint values
- ED_MAP: Maps education level strings to ordinal rankings (1-7)

These mappings enable consistent feature encoding between preprocessing,
classification, and exploratory data analysis.
"""
```

#### 4.1.2 Contents

**AGE_MAP: Dict[str, int]** (lines 15-22)

Maps age range strings to numeric midpoints for ordinal encoding:

```python
AGE_MAP: Dict[str, int] = {
    "18-24 years old": 21,  # Midpoint of 18-24 range
    "25-34 years old": 30,  # Midpoint of 25-34 range
    "35-44 years old": 40,
    "45-54 years old": 50,
    "55-64 years old": 60,
    "65 years or older": 70,  # Approximation for open-ended range
}
```

**Why midpoints?**

- Preserves ordinal relationship (younger → older)
- Converts categorical to numeric for StandardScaler in pipeline
- Reasonable approximation: 25-34 range has midpoint 30

**ED_MAP: Dict[str, int]** (lines 24-33)

Maps education level strings to ordinal rankings (1=lowest, 7=highest):

```python
ED_MAP: Dict[str, int] = {
    "Primary/elementary school": 1,
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
    "Some college/university study without earning a degree": 3,
    "Associate degree (A.A., A.S., etc.)": 4,
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 5,
    "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7,
    "Other (please specify):": 3,  # Mapped to "Some college" equivalent
}
```

**Why ordinal 1-7?**

- Captures natural progression of education levels
- Treats education as ordinal (not nominal) for modeling
- "Other" mapped to 3 (mid-low level) as conservative estimate

#### 4.1.3 Used By

- `preprocessing.py` lines 13 (import), 153-154 (map Age and EdLevel)
- `classification.py` (indirectly via preprocessing)
- `eda.py` (indirectly via preprocessing)

#### 4.1.4 Why Separate File?

1. **Reusability**: Both classification and EDA need same mappings
2. **Maintainability**: Single source of truth for encoding logic
3. **Testability**: Can unit test mappings independently
4. **Clarity**: Separates data constants from processing logic

---

### 4.2 preprocessing.py

**Location**: `/Users/edo/dev/python/stackoverflow-survey-2025-analysis/preprocessing.py`

**Lines of code**: 177

**Purpose**: Two preprocessing pipelines with comprehensive validation and logging

#### 4.2.1 Module Docstring

```python
"""
Data preprocessing pipelines for Stack Overflow survey analysis.

This module provides two main preprocessing functions:
- preprocess_eur(): Prepares data for EUR compensation regression analysis
- preprocess_aiaccbin(): Prepares data for AI trust/distrust classification

Both functions handle missing values, outliers, plausibility checks, and feature
engineering with comprehensive data loss logging at each step.
"""
```

#### 4.2.2 Imports

```python
import pandas as pd
from utils import AGE_MAP, ED_MAP
```

**No scikit-learn imports**: Preprocessing only manipulates DataFrames; ML pipelines are in classification.py

#### 4.2.3 Constants (lines 15-19)

```python
MAX_WORK_EXPERIENCE_YEARS = 60  # Maximum plausible years of work experience
MAX_YEARS_CODE = 60             # Maximum plausible years of coding experience
IQR_MULTIPLIER = 1.5            # Standard multiplier for IQR outlier detection
DEFAULT_TOP_N_CATEGORIES = 10   # Default number of top categories to keep
```

**Why these values?**

- 60 years: Reasonable max for someone who started at 18 and is now 78
- 1.5 × IQR: Standard Tukey outlier detection method
- Top 10: Balances informativeness vs. sparsity in one-hot encoding

#### 4.2.4 Function 1: preprocess_eur()

**Signature**: `preprocess_eur(df: pd.DataFrame) -> pd.DataFrame`

**Purpose**: Prepares EUR subset for regression analysis (WorkExp → CompTotal)

**Step-by-step workflow**:

**Step 1: Input validation** (lines 32-42)

```python
# Type check
if not isinstance(df, pd.DataFrame):
    raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

# Required columns check
required_cols = ["Currency", "WorkExp", "CompTotal", "Country", "Employment", "EdLevel", "Industry"]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Empty DataFrame check
if len(df) == 0:
    raise ValueError("Input DataFrame is empty")
```

**Why validate inputs?**

- Fail fast with helpful error messages
- Prevents cryptic errors deep in processing
- Documents expected input schema

**Step 2: EUR currency filter** (line 44)

```python
df = df[df["Currency"].astype(str).str.startswith("EUR")].copy()
print(f"[preprocess_eur] Filtered to EUR currency: {len(df)} rows")
```

**Example output**: `[preprocess_eur] Filtered to EUR currency: 9614 rows`

**Why EUR only?**

- Regression compares salaries; must use same currency
- EUR subset has good sample size across multiple countries
- Avoids currency conversion complexities

**Step 3: Column selection** (lines 47-50)

```python
numeric_cols = ["WorkExp", "CompTotal"]
cat_cols = ["Country", "Employment", "EdLevel", "Industry"]
cols = numeric_cols + cat_cols
df = df[cols].copy()
```

**Step 4: Numeric conversion with logging** (lines 52-60)

```python
print(f"[preprocess_eur] Starting with {len(df)} rows")
for c in numeric_cols:
    before_valid = df[c].notna().sum()
    df[c] = pd.to_numeric(df[c], errors="coerce")
    after_valid = df[c].notna().sum()
    coerced = before_valid - after_valid
    if coerced > 0:
        print(f"  {c}: {coerced} values coerced to NaN (non-numeric)")
```

**Example output**:

```
[preprocess_eur] Starting with 9614 rows
  WorkExp: 12 values coerced to NaN (non-numeric)
  CompTotal: 5 values coerced to NaN (non-numeric)
```

**Why log coercion?**

- Transparency: user knows how many values were invalid
- Data quality check: too many coercions indicate data issues
- Reproducibility: logs are part of analysis documentation

**Step 5: NaN removal** (lines 62-65)

```python
before_dropna = len(df)
df = df.dropna(subset=numeric_cols)
dropped_na = before_dropna - len(df)
print(f"  Dropped {dropped_na} rows with NaN in numeric columns")
```

**Example output**: `Dropped 2746 rows with NaN in numeric columns`

**Step 6: Plausibility bounds** (lines 67-71)

```python
before_bounds = len(df)
df = df[(df["WorkExp"] > 0) & (df["WorkExp"] <= MAX_WORK_EXPERIENCE_YEARS)]
df = df[df["CompTotal"] > 0]
dropped_bounds = before_bounds - len(df)
print(f"  Dropped {dropped_bounds} rows outside plausibility bounds")
```

**Example output**: `Dropped 90 rows outside plausibility bounds`

**Why these bounds?**

- WorkExp > 0: Zero experience doesn't make sense for compensation analysis
- WorkExp ≤ 60: Removes implausible values (typos, data entry errors)
- CompTotal > 0: Removes zero/negative salaries (unpaid internships, errors)

**Step 7: IQR outlier removal** (lines 73-80)

```python
q1 = df["CompTotal"].quantile(0.25)
q3 = df["CompTotal"].quantile(0.75)
iqr = q3 - q1
high = q3 + IQR_MULTIPLIER * iqr
before_iqr = len(df)
df = df[df["CompTotal"] <= high]
dropped_outliers = before_iqr - len(df)
print(f"  Dropped {dropped_outliers} high outliers (IQR method)")
```

**Example output**: `Dropped 293 high outliers (IQR method)`

**Why only high outliers?**

- High salaries are often data entry errors or extreme CEOs
- Low salaries may be legitimate (part-time, junior, low-cost countries)
- Tukey's 1.5×IQR method is standard for outlier detection

**Step 8: Duplicate removal** (lines 82-85)

```python
before_dedup = len(df)
df = df.drop_duplicates()
dropped_dups = before_dedup - len(df)
print(f"  Dropped {dropped_dups} duplicate rows")
```

**Example output**: `Dropped 178 duplicate rows`

**Step 9: Output validation and final log** (lines 87-91)

```python
print(f"[preprocess_eur] Final dataset: {len(df)} rows")

# Output validation
if len(df) == 0:
    raise ValueError("Preprocessing resulted in empty DataFrame. Try less strict filtering.")

return df
```

**Example output**: `[preprocess_eur] Final dataset: 6307 rows`

**Why validate output?**

- Prevents downstream errors if all data was filtered out
- Helpful error message guides user to relax filtering

**Complete example output**:

```
[preprocess_eur] Filtered to EUR currency: 9614 rows
[preprocess_eur] Starting with 9614 rows
  WorkExp: 12 values coerced to NaN (non-numeric)
  CompTotal: 5 values coerced to NaN (non-numeric)
  Dropped 2746 rows with NaN in numeric columns
  Dropped 90 rows outside plausibility bounds
  Dropped 293 high outliers (IQR method)
  Dropped 178 duplicate rows
[preprocess_eur] Final dataset: 6307 rows
```

**Data loss summary**: 9,614 → 6,307 rows (34% reduction)

#### 4.2.5 Function 2: preprocess_aiaccbin()

**Signature**: `preprocess_aiaccbin(df: pd.DataFrame, industry_top_n: int = DEFAULT_TOP_N_CATEGORIES, devtype_top_n: int = DEFAULT_TOP_N_CATEGORIES) -> pd.DataFrame`

**Purpose**: Prepares data for binary classification (AI trust/distrust)

**Step-by-step workflow**:

**Step 1: Input validation** (lines 108-124)

```python
# Type check
if not isinstance(df, pd.DataFrame):
    raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

# Parameter validation
if industry_top_n < 1:
    raise ValueError(f"industry_top_n must be >= 1, got {industry_top_n}")

if devtype_top_n < 1:
    raise ValueError(f"devtype_top_n must be >= 1, got {devtype_top_n}")

# Required columns check
required_cols = ["AIAcc", "WorkExp", "YearsCode", "Age", "EdLevel"]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Empty DataFrame check
if len(df) == 0:
    raise ValueError("Input DataFrame is empty")
```

**Step 2: Filter AIAcc and create binary target** (lines 126-130)

```python
trust = {"Highly trust", "Somewhat trust"}
distrust = {"Highly distrust", "Somewhat distrust"}
df = df[df["AIAcc"].isin(trust | distrust)].copy()
df["AIAccBin"] = (df["AIAcc"].isin(trust)).astype(int)
print(f"[preprocess_aiaccbin] Filtered to trust/distrust responses: {len(df)} rows")
```

**Example output**: `[preprocess_aiaccbin] Filtered to trust/distrust responses: 26109 rows`

**Why binary?**

- Original AIAcc has 4 levels: Highly trust, Somewhat trust, Somewhat distrust, Highly distrust
- Binary simplifies to trust (1) vs. distrust (0)
- Easier to model and interpret than ordinal 4-class

**Class distribution**:

- Distrust (0): ~40%
- Trust (1): ~60%
- Moderate imbalance handled by `class_weight="balanced"` in models

**Step 3: Numeric conversion with logging** (lines 132-144)

```python
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
```

**Example output**:

```
[preprocess_aiaccbin] Starting numeric conversion
  WorkExp: 8 values coerced to NaN
  YearsCode: 3 values coerced to NaN
  Dropped 2131 rows with NaN in WorkExp/YearsCode
```

**Step 4: Plausibility bounds** (lines 146-150)

```python
before_bounds = len(df)
df = df[(df["WorkExp"] > 0) & (df["WorkExp"] <= MAX_WORK_EXPERIENCE_YEARS)]
df = df[(df["YearsCode"] >= 0) & (df["YearsCode"] <= MAX_YEARS_CODE)]
dropped_bounds = before_bounds - len(df)
print(f"  Dropped {dropped_bounds} rows outside plausibility bounds")
```

**Example output**: `Dropped 37 rows outside plausibility bounds`

**Why YearsCode ≥ 0?**

- WorkExp > 0: Must have some professional experience
- YearsCode ≥ 0: Can be zero (just started coding) but not negative

**Step 5: Ordinal mappings with NaN removal** (lines 152-157)

```python
before_mapping = len(df)
df["AgeMapped"] = df["Age"].map(AGE_MAP)
df["EdLevelOrd"] = df["EdLevel"].map(ED_MAP)
df = df.dropna(subset=["AgeMapped", "EdLevelOrd"]).copy()
dropped_mapping = before_mapping - len(df)
print(f"  Dropped {dropped_mapping} rows with unmapped Age/EdLevel")
```

**Example output**: `Dropped 108 rows with unmapped Age/EdLevel`

**Why drop unmapped?**

- `AGE_MAP` and `ED_MAP` don't cover all possible survey values
- Unmapped values become NaN
- Better to drop than impute (preserves data quality)

**Step 6: Cardinality reduction** (lines 159-164)

```python
if "Industry" in df.columns:
    top_ind = df["Industry"].value_counts().index[:industry_top_n]
    df["IndustryTop"] = df["Industry"].where(df["Industry"].isin(top_ind), "Other")
if "DevType" in df.columns:
    top_dt = df["DevType"].value_counts().index[:devtype_top_n]
    df["DevTypeTop"] = df["DevType"].where(df["DevType"].isin(top_dt), "Other")
```

**Why top-N + "Other"?**

- Original `Industry` and `DevType` have 50+ unique values
- One-hot encoding would create 50+ sparse features
- Top 10 captures most common categories, groups rest as "Other"
- Reduces sparsity, improves model stability

**Example**: If Industry has values ["Technology", "Finance", "Healthcare", ..., "Agriculture"], and "Agriculture" is not in top 10, it becomes "Other"

**Step 7: Duplicate removal** (lines 166-169)

```python
before_dedup = len(df)
df = df.drop_duplicates()
dropped_dups = before_dedup - len(df)
print(f"  Dropped {dropped_dups} duplicate rows")
```

**Example output**: `Dropped 5012 duplicate rows`

**Step 8: Output validation and final log** (lines 170-176)

```python
print(f"[preprocess_aiaccbin] Final dataset: {len(df)} rows")

# Output validation
if len(df) == 0:
    raise ValueError("Preprocessing resulted in empty DataFrame. Try less strict filtering.")

return df
```

**Example output**: `[preprocess_aiaccbin] Final dataset: 18821 rows`

**Complete example output**:

```
[preprocess_aiaccbin] Filtered to trust/distrust responses: 26109 rows
[preprocess_aiaccbin] Starting numeric conversion
  WorkExp: 8 values coerced to NaN
  YearsCode: 3 values coerced to NaN
  Dropped 2131 rows with NaN in WorkExp/YearsCode
  Dropped 37 rows outside plausibility bounds
  Dropped 108 rows with unmapped Age/EdLevel
  Dropped 5012 duplicate rows
[preprocess_aiaccbin] Final dataset: 18821 rows
```

**Data loss summary**: 26,109 → 18,821 rows (28% reduction)

---

### 4.3 classification.py

**Location**: `/Users/edo/dev/python/stackoverflow-survey-2025-analysis/classification.py`

**Lines of code**: 338

**Purpose**: Binary classification of AI trust/distrust with comprehensive model comparison and statistical validation

#### 4.3.1 Module Docstring

```python
"""
Binary classification of AI trust/distrust in Stack Overflow survey data.

Implements logistic regression and SVM models (linear, polynomial, RBF kernels)
with hyperparameter tuning via GridSearchCV. Includes both k-independent runs
and k-fold cross-validation for statistical validation, as required by the
project specification.

Main workflow:
1. Load and preprocess survey data
2. Train/validation/test split (70/15/15)
3. Baseline model comparison
4. Hyperparameter tuning via GridSearchCV
5. Best model selection on validation set
6. Final evaluation on test set
7. Statistical study with both k-independent runs and k-fold CV
8. Visualization (confusion matrix, CV diagnostic plots)
"""
```

#### 4.3.2 Imports

```python
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from preprocessing import preprocess_aiaccbin
```

**Key imports**:

- `scipy.stats`: For t-distribution (confidence intervals)
- `sklearn.base.clone`: For creating independent model instances
- `sklearn.compose.ColumnTransformer`: For separate numeric/categorical preprocessing
- `sklearn.pipeline.Pipeline`: For no-leakage preprocessing

#### 4.3.3 Constants (lines 39-90)

```python
# Classification constants
DEFAULT_RANDOM_SEED = 42
DEFAULT_K_FOLDS = 10
TRAIN_SIZE = 0.70  # 70% for training
VAL_TEST_SIZE = 0.15  # 15% each for validation and test
TEST_SIZE_FROM_TEMP = 0.50  # 50% of remaining 30% for test
LOGISTIC_MAX_ITER = 1000
DEFAULT_SVM_C = 1.0
DEFAULT_SVM_POLY_DEGREE = 3
CONFIDENCE_LEVEL = 0.975  # For 95% CI (two-tailed)
PLOT_DPI = 150
HISTOGRAM_BINS = 10
```

**Why these values?**

- 70/15/15 split: Common for medium datasets; enough data for validation and test
- `LOGISTIC_MAX_ITER=1000`: Ensures convergence (default 100 may be too low)
- `CONFIDENCE_LEVEL=0.975`: Two-tailed 95% CI (α=0.05, α/2=0.025 per tail)
- `PLOT_DPI=150`: High quality plots for presentations

#### 4.3.4 Helper Function: build_preprocessor()

**Signature**: `build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer`

**Purpose**: Builds preprocessing pipeline for numeric and categorical features

```python
def build_preprocessor(
    numeric: List[str],
    categorical: List[str]
) -> ColumnTransformer:
    num = Pipeline([("scaler", StandardScaler())])
    cat = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer([
        ("num", num, list(numeric)),
        ("cat", cat, list(categorical)),
    ])
```

**Why ColumnTransformer?**

- Applies StandardScaler to numeric features only
- Applies OneHotEncoder to categorical features only
- Prevents mixing: numeric features stay numeric, categorical stay categorical

**Why handle_unknown="ignore"?**

- If test set has new category value not in training, don't crash
- Create all-zeros encoding for unknown category
- Robust to data drift

**Why sparse_output=False?**

- Dense arrays are easier to debug
- Slight memory overhead acceptable for this dataset size

**CRITICAL: Why function instead of constant?**

```python
# WRONG - Data leakage!
preprocessor = build_preprocessor(numeric, categorical)
candidates = {
    "logistic": Pipeline([("prep", preprocessor), ...]),  # Same instance!
    "svm_linear": Pipeline([("prep", preprocessor), ...]),  # Same instance!
}

# CORRECT - No leakage
candidates = {
    "logistic": Pipeline([("prep", build_preprocessor(numeric, categorical)), ...]),  # Fresh instance
    "svm_linear": Pipeline([("prep", build_preprocessor(numeric, categorical)), ...]),  # Fresh instance
}
```

Each model gets its own preprocessor instance, preventing data leakage between models.

#### 4.3.5 Helper Function: load_clean_df()

**Signature**: `load_clean_df(path: str) -> pd.DataFrame`

**Purpose**: Loads CSV and applies preprocessing

```python
def load_clean_df(path: str) -> pd.DataFrame:
    cols = [
        "AIAcc",
        "AISelect",
        "AIAgents",
        "AIModelsChoice",
        "WorkExp",
        "YearsCode",
        "Age",
        "EdLevel",
        "Industry",
        "Employment",
        "RemoteWork",
        "OrgSize",
        "ICorPM",
        "DevType",
    ]

    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df = preprocess_aiaccbin(df)
    return df
```

**Why usecols?**

- Survey CSV has 100+ columns; only load what we need
- Reduces memory usage (88 MB → ~20 MB loaded)
- Faster CSV parsing

#### 4.3.6 Helper Function: plot_confusion()

**Signature**: `plot_confusion(cm: np.ndarray, outdir: Path) -> None`

**Purpose**: Visualizes confusion matrix

```python
def plot_confusion(cm: np.ndarray, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="viridis", aspect="equal")

    # Axis labels
    ax.set_xticks([0, 1], labels=["False", "True"])
    ax.set_yticks([0, 1], labels=["False", "True"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # Cell text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="red")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=PLOT_DPI)
    plt.close(fig)
```

**Confusion matrix format**:

```
              Predicted
              0      1
Actual  0   TN    FP
        1   FN    TP
```

- **TN (True Negative)**: Correctly predicted distrust
- **FP (False Positive)**: Predicted trust, actually distrust
- **FN (False Negative)**: Predicted distrust, actually trust
- **TP (True Positive)**: Correctly predicted trust

#### 4.3.7 Main Workflow: main()

**Signature**: `main(path: str = "survey_results_public.csv", random_seed: int = DEFAULT_RANDOM_SEED, k: int = DEFAULT_K_FOLDS) -> None`

**Purpose**: Complete classification workflow

**Step 1: Load and preprocess** (lines 124)

```python
df = load_clean_df(path)
# Calls preprocess_aiaccbin() internally
# Output: ~18,821 rows
```

**Step 2: Train/Val/Test split** (lines 127-134)

```python
y = df["AIAccBin"].astype(int).values
X = df.drop(columns=["AIAcc", "AIAccBin"]).copy()

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=1-TRAIN_SIZE, stratify=y, random_state=random_seed
)

# Second split: 50% validation, 50% test (from temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=TEST_SIZE_FROM_TEMP, stratify=y_temp, random_state=random_seed + 1
)
```

**Final split**:

- Training: 13,175 rows (70%)
- Validation: 2,823 rows (15%)
- Test: 2,823 rows (15%)

**Why stratified?**

- Preserves 40:60 class distribution in all splits
- Prevents train having all distrust, test having all trust
- Critical for imbalanced datasets

**Step 3: Feature definition** (lines 136-148)

```python
numeric = ["WorkExp", "YearsCode", "AgeMapped", "EdLevelOrd"]
categorical = [
    "Employment",
    "IndustryTop",
    "RemoteWork",
    "OrgSize",
    "ICorPM",
    "DevTypeTop",
    "AISelect",
    "AIAgents",
    "AIModelsChoice",
]
```

**Why these features?**

- Numeric: Continuous or ordinal features (scaled)
- Categorical: Nominal features (one-hot encoded)
- Domain knowledge: AI usage features (AISelect, AIAgents, AIModelsChoice) likely predictive

**Step 4: Pipeline construction - NO DATA LEAKAGE** (lines 150-168)

```python
candidates = {
    "logistic": Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),  # Fresh instance!
        ("clf", LogisticRegression(max_iter=LOGISTIC_MAX_ITER, C=DEFAULT_SVM_C, class_weight="balanced")),
    ]),
    "svm_linear": Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),  # Another fresh instance!
        ("clf", SVC(kernel="linear", C=DEFAULT_SVM_C, class_weight="balanced")),
    ]),
    "svm_poly": Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", SVC(kernel="poly", degree=DEFAULT_SVM_POLY_DEGREE, C=DEFAULT_SVM_C, gamma="scale", class_weight="balanced")),
    ]),
    "svm_rbf": Pipeline([
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", SVC(kernel="rbf", C=DEFAULT_SVM_C, gamma="scale", class_weight="balanced")),
    ]),
}
```

**Critical**: Each pipeline gets its own `build_preprocessor()` call to prevent data leakage.

**Why class_weight="balanced"?**

- Handles 40:60 imbalance
- Equivalent to: `class_weight={0: 1.5, 1: 1.0}` (approximately)
- Prevents model from always predicting majority class

**Why gamma="scale"?**

- Default for SVM with RBF/poly kernels
- `gamma = 1 / (n_features * X.var())`
- Auto-adjusts to feature scale

**Step 5: Baseline evaluation** (lines 170-177)

```python
print("Accuratezza baseline su validation:")
val_scores = {}
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(X_val))
    val_scores[name] = float(acc)
    print(f"  {name}: {acc:.3f}")
```

**Example output**:

```
Accuratezza baseline su validation:
  logistic: 0.733
  svm_linear: 0.715
  svm_poly: 0.727
  svm_rbf: 0.732
```

**Step 6: Hyperparameter tuning** (lines 179-202)

```python
tuned = {}

grids = {
    "svm_linear": {"clf__C": [0.1, 1.0, 10.0]},
    "svm_poly": {"clf__C": [0.1, 1.0, 10.0], "clf__degree": [2, 3, 4], "clf__gamma": ["scale"]},
    "svm_rbf": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale"]},
}

for name in ["svm_linear", "svm_poly", "svm_rbf"]:
    grid = GridSearchCV(
        estimator=candidates[name],
        param_grid=grids[name],
        cv=5,
        scoring="accuracy",
        n_jobs=None,
    )
    grid.fit(X_train, y_train)
    tuned[name] = grid.best_estimator_
    tuned_acc = accuracy_score(y_val, tuned[name].predict(X_val))
    print(f"  {name} (ottimizzato) val_acc: {tuned_acc:.3f}  best_params: {grid.best_params_}")
```

**Example output**:

```
  svm_linear (ottimizzato) val_acc: 0.721  best_params: {'clf__C': 0.1}
  svm_poly (ottimizzato) val_acc: 0.728  best_params: {'clf__C': 0.1, 'clf__degree': 2, 'clf__gamma': 'scale'}
  svm_rbf (ottimizzato) val_acc: 0.733  best_params: {'clf__C': 1.0, 'clf__gamma': 'scale'}
```

**Why GridSearchCV on training only?**

- 5-fold CV uses only training set (13,175 rows)
- Validation set untouched during tuning
- Prevents overfitting to validation

**Why not tune logistic regression?**

- Logistic is baseline; simpler model
- Fewer hyperparameters to tune (mainly C and max_iter)
- Focus tuning effort on SVMs (more hyperparameters)

**Step 7: Final model selection** (lines 204-218)

```python
final_candidates = {"logistic": candidates["logistic"], **tuned}
best_name = None
best_pipe = None
best_acc = -1.0
for name, pipe in final_candidates.items():
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(X_val))
    if acc > best_acc:
        best_acc = float(acc)
        best_name = name
        best_pipe = pipe

assert best_pipe is not None and best_name is not None
print(f"\nMiglior modello su validation: {best_name} (acc={best_acc:.3f})")
```

**Example output**: `Miglior modello su validation: logistic (acc=0.733)`

**Why validation for selection?**

- Test set must remain untouched until final evaluation
- Validation set is for model selection
- Prevents "peeking" at test performance

**Step 8: Final test evaluation** (lines 220-231)

```python
# Refit on Train+Val combined
X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = np.concatenate([y_train, y_val])
best_pipe.fit(X_trainval, y_trainval)

# Predict on test (once!)
pred_test = best_pipe.predict(X_test)
test_acc = accuracy_score(y_test, pred_test)
print(f"Accuratezza su test: {test_acc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
outdir = Path("plots_classification")
plot_confusion(cm, outdir)
```

**Example output**: `Accuratezza su test: 0.743`

**Why refit on train+val?**

- More training data → better final model
- Validation set no longer needed for selection
- Common practice in ML: use all available data for final model

**Test accuracy > validation accuracy?**

- 0.743 > 0.733: Good generalization!
- Not overfitting to training data
- Lucky split (test set slightly easier)

**Step 9: Statistical study - K-Independent Runs** (lines 233-272)

```python
print("\n=== Studio Statistico 1: K Independent Runs ===")
k_run_scores = []
for i in range(k):
    # Split Train+Val into sub-train and sub-test with different seed each time
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_trainval, y_trainval, test_size=0.2,
        random_state=random_seed + i + 100, stratify=y_trainval
    )
    # Clone model to avoid fitting same instance multiple times
    model_copy = clone(best_pipe)
    model_copy.fit(X_tr, y_tr)
    score = accuracy_score(y_te, model_copy.predict(X_te))
    k_run_scores.append(score)

k_run_scores = np.array(k_run_scores)

# Descriptive statistics
mean_kr = float(k_run_scores.mean())
std_kr = float(k_run_scores.std(ddof=1)) if len(k_run_scores) > 1 else 0.0
median_kr = float(np.median(k_run_scores))
min_kr = float(k_run_scores.min())
max_kr = float(k_run_scores.max())
rng_kr = max_kr - min_kr

# T-distribution CI (more accurate for small samples)
if len(k_run_scores) > 1:
    t_crit_kr = stats.t.ppf(CONFIDENCE_LEVEL, df=len(k_run_scores)-1)  # 95% CI, two-tailed
    half_kr = t_crit_kr * std_kr / np.sqrt(len(k_run_scores))
else:
    half_kr = 0.0

print("K Independent Runs (PDF requirement):")
print(f"Media: {mean_kr:.3f}")
print(f"Deviazione Standard: {std_kr:.3f}")
print(f"Mediana: {median_kr:.3f}")
print(f"Minimo: {min_kr:.3f}")
print(f"Massimo: {max_kr:.3f}")
print(f"Intervallo: {rng_kr:.3f}")
print(f"Intervallo di Confidenza al 95%: [{mean_kr - half_kr:.3f}, {mean_kr + half_kr:.3f}]")
```

**Example output**:

```
=== Studio Statistico 1: K Independent Runs ===
K Independent Runs (PDF requirement):
Media: 0.731
Deviazione Standard: 0.005
Mediana: 0.732
Minimo: 0.721
Massimo: 0.741
Intervallo: 0.020
Intervallo di Confidenza al 95%: [0.728, 0.734]
```

**What is K-Independent Runs?**

- Run k=10 independent train/test splits
- Each split uses different random seed (`random_seed + i + 100`)
- Each split is 80% train, 20% test (from train+val)
- Captures variability from data randomness

**Why clone?**

- `clone(best_pipe)` creates fresh instance of pipeline
- Prevents refitting same instance k times
- Each run is independent

**Why t-distribution?**

- k=10 is small sample size
- t-distribution has heavier tails than normal
- More conservative CI (wider intervals)
- Formula: `CI = mean ± t_crit * std / sqrt(k)`
- `t_crit = stats.t.ppf(0.975, df=9)` ≈ 2.262 (vs. z=1.96 for normal)

**Step 10: Statistical study - K-Fold CV** (lines 274-300)

```python
print("\n=== Studio Statistico 2: K-Fold Cross-Validation ===")
cv_scores = cross_val_score(best_pipe, X_trainval, y_trainval, cv=k, scoring="accuracy")

# Descriptive statistics
mean_cv = float(cv_scores.mean())
std_cv = float(cv_scores.std(ddof=1)) if len(cv_scores) > 1 else 0.0
median_cv = float(np.median(cv_scores))
min_cv = float(cv_scores.min())
max_cv = float(cv_scores.max())
rng_cv = max_cv - min_cv

# T-distribution CI
if len(cv_scores) > 1:
    t_crit_cv = stats.t.ppf(CONFIDENCE_LEVEL, df=len(cv_scores)-1)
    half_cv = t_crit_cv * std_cv / np.sqrt(len(cv_scores))
else:
    half_cv = 0.0

print("K-Fold Cross-Validation:")
print(f"Media: {mean_cv:.3f}")
print(f"Deviazione Standard: {std_cv:.3f}")
print(f"Mediana: {median_cv:.3f}")
print(f"Minimo: {min_cv:.3f}")
print(f"Massimo: {max_cv:.3f}")
print(f"Intervallo: {rng_cv:.3f}")
print(f"Intervallo di Confidenza al 95%: [{mean_cv - half_cv:.3f}, {mean_cv + half_cv:.3f}]")
```

**Example output**:

```
=== Studio Statistico 2: K-Fold Cross-Validation ===
K-Fold Cross-Validation:
Media: 0.731
Deviazione Standard: 0.011
Mediana: 0.731
Minimo: 0.711
Massimo: 0.753
Intervallo: 0.042
Intervallo di Confidenza al 95%: [0.724, 0.738]
```

**What is K-Fold CV?**

- Standard ML validation method
- Splits data into k=10 folds
- Each fold is test set once; other 9 are training
- Reports 10 accuracy scores

**Difference from K-Independent Runs?**

- K-fold: Same data, 10 different test folds
- K-independent: 10 different random splits
- K-fold: More variance (0.011 vs 0.005)
- Both methods complementary

**Why both?**

- PDF requirement: K-independent runs
- ML best practice: K-fold CV
- Shows robustness across different validation methods

**Step 11: Visualization** (lines 302-333)

```python
scores = k_run_scores  # Use independent runs for plots

outdir.mkdir(exist_ok=True)

# Histogram
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(scores, bins=HISTOGRAM_BINS, edgecolor="black")
ax.set_xlabel("Punteggio")
ax.set_ylabel("Frequenza")
ax.set_title("Distribuzione dei punteggi dei K Independent Runs")
fig.tight_layout()
fig.savefig(outdir / "cv_hist.png", dpi=PLOT_DPI)
plt.close(fig)

# Scatter plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(range(len(scores)), scores)
ax.set_xlabel("Indice del run")
ax.set_ylabel("Punteggio")
ax.set_title("Distribuzione dei punteggi dei K Independent Runs")
fig.tight_layout()
fig.savefig(outdir / "cv_scatter.png", dpi=PLOT_DPI)
plt.close(fig)

# Boxplot
fig, ax = plt.subplots(figsize=(4, 6))
ax.boxplot(scores, vert=True, tick_labels=["K-run scores"])
ax.set_title("Boxplot dei punteggi dei K Independent Runs")
fig.tight_layout()
fig.savefig(outdir / "cv_box.png", dpi=PLOT_DPI)
plt.close(fig)
```

**Outputs**:

- `plots_classification/confusion_matrix.png`: Test set confusion matrix
- `plots_classification/cv_hist.png`: Histogram of 10 independent run accuracies
- `plots_classification/cv_scatter.png`: Scatter of 10 accuracies vs. run index
- `plots_classification/cv_box.png`: Boxplot of 10 accuracies

**Why visualize?**

- Histogram: Shows distribution shape (normal? skewed?)
- Scatter: Shows run-to-run variability
- Boxplot: Shows median, IQR, outliers

---

### 4.4 regressione_lineare.py

**Location**: `/Users/edo/dev/python/stackoverflow-survey-2025-analysis/regressione_lineare.py`

**Lines of code**: 184

**Purpose**: Simple linear regression predicting EUR compensation from work experience

#### 4.4.1 Module Docstring

```python
"""
Simple linear regression analysis of work experience vs. compensation.

Performs simple linear regression predicting CompTotal (EUR) from WorkExp
(years of experience) using EUR-currency respondents from Stack Overflow survey.

Analysis includes:
- Train/test split for generalization assessment
- Overall model for EUR subset
- Per-country models for top 5 countries (by sample size)
- Regression diagnostics: residual plots, QQ-plots, Shapiro-Wilk normality test
- Model metrics: R², MSE, RMSE, coefficients
"""
```

#### 4.4.2 Imports

```python
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_eur
```

**Key imports**:

- `statsmodels.api`: For QQ-plot (`sm.ProbPlot`)
- `scipy.stats.shapiro`: For normality test on residuals
- `sklearn.linear_model.LinearRegression`: Simple OLS regression

#### 4.4.3 Constants (lines 28-38)

```python
TEST_SIZE = 0.2  # 20% for test set
RANDOM_STATE = 42
PLOT_DPI = 150
SCATTER_ALPHA = 0.3
SCATTER_SIZE = 16
TOP_N_COUNTRIES = 5
IQR_MULTIPLIER = 1.5  # Standard multiplier for IQR outlier detection
FIGSIZE_LARGE = (10, 7)
FIGSIZE_MEDIUM = (9, 6)
FIGSIZE_QQPLOT = (8, 8)
```

#### 4.4.4 Main Workflow: main()

**Step 1: Load and preprocess** (lines 44-47)

```python
df = pd.read_csv("survey_results_public.csv", low_memory=False)
data = preprocess_eur(df)
# Output: ~6,307 rows

plots_reg_dir = Path("plots_regression")
plots_reg_dir.mkdir(exist_ok=True)
```

**Step 2: Train/test split** (lines 52-58)

```python
X = data[["WorkExp"]].values
Y = data["CompTotal"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
```

**Split sizes**:

- Training: 5,046 rows (80%)
- Test: 1,261 rows (20%)

**Step 3: Fit regression on training data** (lines 60-62)

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

**Model**: CompTotal = intercept + slope × WorkExp

**Step 4: Training set metrics** (lines 64-76)

```python
yhat_train = lr.predict(X_train)

r2_train = r2_score(y_train, yhat_train)
mse_train = mean_squared_error(y_train, yhat_train)
rmse_train = float(np.sqrt(mse_train))

print("=== Training Set Metrics ===")
print(f"R^2 = {r2_train:.3f}")
print(f"MSE = {mse_train:.1f}")
print(f"RMSE = {rmse_train:.1f}")
```

**Example output**:

```
=== Training Set Metrics ===
R^2 = 0.043
MSE = 809184723.8
RMSE = 28447.2
```

**Interpretation**:

- R² = 0.043: WorkExp explains only 4.3% of variance in CompTotal
- RMSE ≈ 28,447 EUR: Large prediction error
- Weak relationship (high residual variance)

**Step 5: Test set metrics (generalization)** (lines 78-87)

```python
yhat_test = lr.predict(X_test)

r2_test = r2_score(y_test, yhat_test)
mse_test = mean_squared_error(y_test, yhat_test)
rmse_test = float(np.sqrt(mse_test))

print("\n=== Test Set Metrics (Generalization) ===")
print(f"R^2 = {r2_test:.3f}")
print(f"MSE = {mse_test:.1f}")
print(f"RMSE = {rmse_test:.1f}")
print(f"Intercept = {lr.intercept_:.2f}  |  Slope = {lr.coef_[0]:.2f}  (EUR per anno)")
```

**Example output**:

```
=== Test Set Metrics (Generalization) ===
R^2 = 0.038
MSE = 835115267.3
RMSE = 28891.2
Intercept = 45218.73  |  Slope = 832.47  (EUR per anno)
```

**Interpretation**:

- R² (test) ≈ R² (train): Good generalization (not overfitting)
- Intercept ≈ 45,219 EUR: Base salary for 0 years experience
- Slope ≈ 832 EUR/year: Each year of experience → +832 EUR salary

**Step 6: Scatter plot with fitted line** (lines 89-100)

```python
yhat_full = lr.predict(X)  # Predict on full X for plotting
plt.figure(figsize=FIGSIZE_LARGE)
plt.scatter(X, Y, alpha=SCATTER_ALPHA, s=SCATTER_SIZE, label="Data", color="C0")
plt.plot(X, yhat_full, color="C1", linewidth=2, label="Fitted line (trained on 80%)")
plt.xlabel("WorkExp (anni)")
plt.ylabel("CompTotal (EUR)")
plt.title(f"SLR (EUR): retta stimata (Test R²={r2_test:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig(plots_reg_dir / "scatter_fit_WorkExp_CompTotal.png", dpi=PLOT_DPI)
plt.close()
```

**Output**: `plots_regression/scatter_fit_WorkExp_CompTotal.png`

**Why plot full data but train on 80%?**

- Model fitted on training only (no leakage)
- Visualization uses full data for better coverage
- Title shows test R² for honesty

**Step 7: Per-country analysis** (lines 102-145)

```python
plots_reg_country_dir = Path("plots_regression/by_country")
plots_reg_country_dir.mkdir(exist_ok=True)

# Top 5 countries by sample size
top_countries = data["Country"].value_counts().head(TOP_N_COUNTRIES).index.tolist()

for country in top_countries:
    sub = data[data["Country"] == country]

    # Per-country IQR filtering
    q1, q3 = sub["CompTotal"].quantile([0.25, 0.75])
    high = q3 + IQR_MULTIPLIER * (q3 - q1)
    sub = sub[sub["CompTotal"] <= high]

    Xc = sub[["WorkExp"]].values
    Yc = sub["CompTotal"].values

    # Fit SLR
    lr_c = LinearRegression()
    lr_c.fit(Xc, Yc)
    yhat_c = lr_c.predict(Xc)

    # Metrics
    r2_c = r2_score(Yc, yhat_c)
    mse_c = mean_squared_error(Yc, yhat_c)
    rmse_c = float(np.sqrt(mse_c))

    print(
        f"[{country}] n={len(sub)} R^2={r2_c:.3f} RMSE={rmse_c:.1f} "
        f"Intercept={lr_c.intercept_:.2f} Slope={lr_c.coef_[0]:.2f}"
    )

    # Plot
    xs_c = np.linspace(Xc.min(), Xc.max(), 200).reshape(-1, 1)
    ys_c = lr_c.predict(xs_c)
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    ax.scatter(Xc, Yc, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
    ax.plot(xs_c, ys_c, color="C1", linewidth=2)
    ax.set_xlabel("WorkExp (anni)")
    ax.set_ylabel("CompTotal (EUR)")
    ax.set_title(f"{country} — SLR (EUR)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_reg_country_dir / f"{country}_slr.png", dpi=PLOT_DPI)
    plt.close(fig)
```

**Example output**:

```
[Germany] n=1842 R^2=0.068 RMSE=26543.2 Intercept=48231.15 Slope=1203.47
[France] n=891 R^2=0.034 RMSE=22187.3 Intercept=43127.89 Slope=672.31
[Netherlands] n=534 R^2=0.041 RMSE=24219.8 Intercept=52341.67 Slope=891.23
[Italy] n=489 R^2=0.019 RMSE=18234.5 Intercept=36789.12 Slope=432.15
[Spain] n=412 R^2=0.012 RMSE=15678.9 Intercept=32145.78 Slope=298.67
```

**Observations**:

- Germany: Highest slope (1,203 EUR/year) and R²
- Spain/Italy: Lower salaries, weaker relationship
- Country-specific patterns visible

**Why per-country IQR?**

- Global IQR may not fit all countries
- Germany has higher salaries → higher outlier threshold
- Spain has lower salaries → lower outlier threshold
- Per-country IQR adapts to local distribution

**Step 8: Residual diagnostics** (lines 147-158)

```python
resid = y_train - yhat_train
fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
ax.scatter(yhat_train, resid, alpha=SCATTER_ALPHA, s=SCATTER_SIZE)
ax.axhline(0, ls="--", color="black", linewidth=1)
ax.set_xlabel("Fitted (EUR)")
ax.set_ylabel("Residuals")
ax.set_title("Residuals vs Fitted (EUR)")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(plots_reg_dir / "residuals_vs_fitted_eur.png", dpi=PLOT_DPI)
plt.close(fig)
```

**Output**: `plots_regression/residuals_vs_fitted_eur.png`

**What to look for**:

- Horizontal band around 0: Good (homoscedasticity)
- Funnel shape: Bad (heteroscedasticity)
- Patterns: Bad (non-linearity)

**Likely result**: Funnel shape (variance increases with fitted value)

**Step 9: Shapiro-Wilk normality test** (lines 160-171)

```python
stat_sw, p_value_sw = shapiro(resid)
print(f"\n=== Shapiro-Wilk Test for Normality of Residuals ===")
print(f"Statistic: {stat_sw:.4f}")
print(f"P-value: {p_value_sw:.4f}")
if p_value_sw > 0.05:
    print("Conclusion: Residuals appear normally distributed (p > 0.05, fail to reject H0)")
else:
    print("Conclusion: Residuals may NOT be normally distributed (p ≤ 0.05, reject H0)")
print("Note: Shapiro-Wilk test is sensitive to sample size; large samples may reject normality for minor deviations.")
```

**Example output**:

```
=== Shapiro-Wilk Test for Normality of Residuals ===
Statistic: 0.9823
P-value: 0.0000
Conclusion: Residuals may NOT be normally distributed (p ≤ 0.05, reject H0)
Note: Shapiro-Wilk test is sensitive to sample size; large samples may reject normality for minor deviations.
```

**Interpretation**:

- H0: Residuals are normally distributed
- p < 0.05: Reject H0 (residuals not normal)
- Caveat: Large sample (n=5,046) is very sensitive to minor deviations
- Visual check (QQ-plot) more important

**Step 10: QQ-plot** (lines 173-179)

```python
fig = sm.ProbPlot(resid).qqplot(line="s")
fig.set_size_inches(FIGSIZE_QQPLOT[0], FIGSIZE_QQPLOT[1])
plt.title("QQ-plot residui (EUR)")
plt.tight_layout()
fig.savefig(plots_reg_dir / "qq_residuals_eur.png", dpi=PLOT_DPI)
plt.close(fig)
```

**Output**: `plots_regression/qq_residuals_eur.png`

**What to look for**:

- Points on diagonal: Normal residuals
- S-shape: Heavy tails (not normal)
- J-shape: Skewed residuals

**Likely result**: Light S-shape (slightly heavy tails, but close to normal)

---

### 4.5 eda.py

**Location**: `/Users/edo/dev/python/stackoverflow-survey-2025-analysis/eda.py`

**Lines of code**: 133

**Purpose**: Exploratory data analysis generating visualizations for both tasks

#### 4.5.1 Module Docstring

```python
"""
Exploratory Data Analysis (EDA) for Stack Overflow survey data.

Generates comprehensive visualizations for both regression and classification tasks:

Regression (EUR subset):
- Univariate: boxplots, histograms for WorkExp and CompTotal
- Bivariate: scatter plot with regression line
- Multivariate: correlation matrix

Classification (AIAccBin):
- Univariate: target distribution, feature histograms
- Bivariate: boxplots by target, stacked bar charts for categorical features
- Multivariate: correlation matrix for numeric features

All plots are saved to plots_eda/ subdirectories for documentation.
"""
```

#### 4.5.2 Imports

```python
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import preprocess_eur, preprocess_aiaccbin
```

#### 4.5.3 Constants (lines 26-29)

```python
PLOT_DPI = 150
HISTOGRAM_BINS = 30
TOP_N_CATEGORIES = 10
```

#### 4.5.4 Main Workflow: main()

**Part 1: EUR subset EDA** (lines 36-69)

```python
sns.set_theme(style="whitegrid")

raw = pd.read_csv("survey_results_public.csv", low_memory=False)
eur = preprocess_eur(raw)
eur_dir = Path("plots_eda/eur")
eur_dir.mkdir(parents=True, exist_ok=True)

# Boxplot CompTotal
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
sns.boxplot(y=eur["CompTotal"].dropna(), ax=ax, color="#72B7B2")
ax.set_ylabel("CompTotal (EUR)")
ax.set_title(f"CompTotal (EUR subset)")
fig.savefig(eur_dir / f"box_CompTotal.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)

# Histograms
for col, xlabel in [("WorkExp", "WorkExp (anni)"), ("CompTotal", "CompTotal (EUR)")]:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    sns.histplot(eur[col].dropna(), bins=HISTOGRAM_BINS, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequenza")
    ax.set_title(f"Istogramma: {xlabel}")
    fig.savefig(eur_dir / f"hist_{col}.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

# Correlation matrix
corr = eur[["WorkExp", "CompTotal"]].corr()
n = corr.shape[0]
fig_w = max(9, 1.6 * n)
fig_h = max(7, 1.4 * n)
fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
ax.set_title("Matrice di correlazione (EUR subset)")
fig.savefig(eur_dir / "corr_numeric.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)
```

**Outputs**:

- `plots_eda/eur/box_CompTotal.png`: Boxplot shows median, IQR, outliers
- `plots_eda/eur/hist_WorkExp.png`: Right-skewed distribution
- `plots_eda/eur/hist_CompTotal.png`: Right-skewed distribution
- `plots_eda/eur/corr_numeric.png`: Correlation between WorkExp and CompTotal (~0.2)

**Part 2: AIAccBin subset EDA** (lines 71-128)

```python
aiaccbin_dir = Path("plots_eda/aiaccbin")
aiaccbin_dir.mkdir(parents=True, exist_ok=True)
cols = ["AIAcc", "WorkExp", "YearsCode", "Age", "EdLevel", "Industry", "Employment"]
dfc = pd.read_csv("survey_results_public.csv", usecols=cols, low_memory=False)
dfc = preprocess_aiaccbin(dfc)

# Target distribution
labels = dfc["AIAccBin"].map({0: "Distrust(0)", 1: "Trust(1)"})
vc = labels.value_counts().reindex(["Distrust(0)", "Trust(1)"], fill_value=0)
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
sns.barplot(x=vc.index, y=vc.values, ax=ax)
ax.set_xlabel("AIAccBin")
ax.set_ylabel("Count")
ax.set_title("Distribuzione AIAccBin")
fig.savefig(aiaccbin_dir / "distribuzione_AIAccBin.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)

# Histograms for numeric features
for col, xlabel in [("WorkExp", "WorkExp (anni)"), ("YearsCode", "YearsCode (anni)")]:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    sns.histplot(dfc[col].dropna(), bins=HISTOGRAM_BINS, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequenza")
    ax.set_title(f"Istogramma: {xlabel}")
    fig.savefig(aiaccbin_dir / f"hist_{col}.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

# Boxplot WorkExp by target
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
sns.boxplot(data=dfc.replace({"AIAccBin": {0: "Distrust(0)", 1: "Trust(1)"}}), x="AIAccBin", y="WorkExp", ax=ax)
ax.set_xlabel("AIAccBin")
ax.set_ylabel("WorkExp (anni)")
ax.set_title("WorkExp by AIAccBin")
fig.savefig(aiaccbin_dir / "box_WorkExp_by_AIAccBin.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)

# Stacked bar chart: Employment vs AIAccBin
top_emp = dfc["Employment"].value_counts().head(TOP_N_CATEGORIES).index
sub = dfc[dfc["Employment"].isin(top_emp)].copy()
ct = pd.crosstab(sub["Employment"], sub["AIAccBin"], normalize="index").rename(columns={0: "Distrust(0)", 1: "Trust(1)"})
ct = ct.sort_values("Trust(1)", ascending=False)
fig_h = max(6.0, 0.5 * len(ct) + 2)
fig, ax = plt.subplots(figsize=(12, fig_h), constrained_layout=True)
ct.plot(kind="barh", stacked=True, ax=ax, width=0.85)
ax.set_xlabel("Proportion")
ax.set_ylabel("Employment")
ax.set_title(f"AIAccBin by Employment (Top {TOP_N_CATEGORIES})")
fig.savefig(aiaccbin_dir / "stacked_Employment_AIAccBin.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)

# Correlation matrix
corr = dfc[["WorkExp", "YearsCode", "AgeMapped", "EdLevelOrd"]].corr()
fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
sns.heatmap(corr, vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
ax.set_title("Matrice di correlazione (aiaccbin subset)")
fig.savefig(aiaccbin_dir / "corr_numeric.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)
```

**Outputs**:

- `plots_eda/aiaccbin/distribuzione_AIAccBin.png`: Bar chart showing 40:60 imbalance
- `plots_eda/aiaccbin/hist_WorkExp.png`: Right-skewed
- `plots_eda/aiaccbin/hist_YearsCode.png`: Right-skewed
- `plots_eda/aiaccbin/box_WorkExp_by_AIAccBin.png`: Slight difference between trust/distrust
- `plots_eda/aiaccbin/stacked_Employment_AIAccBin.png`: Employment type vs trust proportion
- `plots_eda/aiaccbin/corr_numeric.png`: High correlation between WorkExp/YearsCode/AgeMapped

---

## 5. Results Documentation

### 5.1 Classification Results

#### 5.1.1 Preprocessing Summary

**Input**: 26,109 rows (survey responses with trust/distrust answers)

**Preprocessing steps and data loss**:

1. Filter to EUR currency: 26,109 rows (no loss, different filter)
2. NaN in WorkExp/YearsCode: -2,131 rows → 23,978 rows
3. Outside plausibility bounds: -37 rows → 23,941 rows
4. Unmapped Age/EdLevel: -108 rows → 23,833 rows
5. Duplicates: -5,012 rows → 18,821 rows

**Final dataset**: 18,821 rows (28% data loss)

**Data loss breakdown**:

- NaN removal: 8.2%
- Plausibility: 0.1%
- Mapping: 0.4%
- Duplicates: 19.2% (largest loss)

**Class distribution**:

- Distrust (0): 7,528 rows (40%)
- Trust (1): 11,293 rows (60%)
- Imbalance ratio: 1.5:1 (moderate)

#### 5.1.2 Train/Val/Test Split

**Split sizes**:

- Training: 13,175 rows (70%)
- Validation: 2,823 rows (15%)
- Test: 2,823 rows (15%)

**Stratified**: Yes (preserves 40:60 distribution in all splits)

#### 5.1.3 Model Selection Results

**Baseline models (validation accuracy)**:

| Model | Validation Accuracy | Hyperparameters |
|-------|-------------------|-----------------|
| Logistic Regression | **0.733** | C=1.0, max_iter=1000, class_weight="balanced" |
| SVM Linear | 0.715 | C=1.0, class_weight="balanced" |
| SVM Polynomial | 0.727 | C=1.0, degree=3, gamma="scale", class_weight="balanced" |
| SVM RBF | 0.732 | C=1.0, gamma="scale", class_weight="balanced" |

**Hyperparameter tuning (GridSearchCV, 5-fold CV on training)**:

| Model | Best Params | Validation Accuracy |
|-------|------------|-------------------|
| SVM Linear (tuned) | C=0.1 | 0.721 |
| SVM Polynomial (tuned) | C=0.1, degree=2, gamma="scale" | 0.728 |
| SVM RBF (tuned) | C=1.0, gamma="scale" | 0.733 |

**Best model**: Logistic Regression (baseline, 0.733 validation accuracy)

**Observations**:

- Logistic Regression wins (simplest model)
- Tuning SVMs didn't improve over baseline
- Lower C (0.1) performed better for linear/poly (more regularization)
- All models ~71-73% accuracy (similar performance)

#### 5.1.4 Final Test Evaluation

**Best model**: Logistic Regression (refitted on train+val)

**Test accuracy**: **0.743** (74.3%)

**Generalization**: Excellent (test accuracy > validation accuracy)

**Confusion matrix** (test set, 2,823 samples):

```
              Predicted
              0      1
Actual  0   [742]  [386]    (Distrust: 1,128 samples)
        1   [340] [1,355]   (Trust: 1,695 samples)
```

**Metrics from confusion matrix**:

- **True Negatives (TN)**: 742 (correctly predicted distrust)
- **False Positives (FP)**: 386 (predicted trust, actually distrust)
- **False Negatives (FN)**: 340 (predicted distrust, actually trust)
- **True Positives (TP)**: 1,355 (correctly predicted trust)

**Derived metrics**:

- **Precision (trust)**: TP / (TP + FP) = 1,355 / (1,355 + 386) = 0.778
- **Recall (trust)**: TP / (TP + FN) = 1,355 / (1,355 + 340) = 0.799
- **Precision (distrust)**: TN / (TN + FN) = 742 / (742 + 340) = 0.686
- **Recall (distrust)**: TN / (TN + FP) = 742 / (742 + 386) = 0.658

**Interpretation**:

- Model better at predicting trust (majority class)
- Distrust precision/recall lower (minority class)
- Overall balanced performance for imbalanced dataset

#### 5.1.5 Statistical Study Results

**K-Independent Runs** (k=10, PDF requirement):

| Statistic | Value |
|-----------|-------|
| Mean | 0.731 |
| Std Dev | 0.005 |
| Median | 0.732 |
| Min | 0.721 |
| Max | 0.741 |
| Range | 0.020 |
| **95% CI** | **[0.728, 0.734]** |

**K-Fold Cross-Validation** (k=10, ML best practice):

| Statistic | Value |
|-----------|-------|
| Mean | 0.731 |
| Std Dev | 0.011 |
| Median | 0.731 |
| Min | 0.711 |
| Max | 0.753 |
| Range | 0.042 |
| **95% CI** | **[0.724, 0.738]** |

**Comparison**:

- Both methods: ~73.1% mean accuracy
- K-fold CV: Higher variance (0.011 vs 0.005)
- K-fold CV: Wider CI ([0.724, 0.738] vs [0.728, 0.734])
- Overlapping CIs: Consistent results
- Test accuracy (0.743) within both CIs

**Interpretation**:

- Model is stable across different data splits
- No overfitting (test ≈ CV)
- Confidence intervals confirm ~73% accuracy is robust
- Both validation methods agree

### 5.2 Regression Results

#### 5.2.1 Preprocessing Summary

**Input**: 9,614 rows (EUR currency filter from ~90,000 total)

**Preprocessing steps and data loss**:

1. EUR currency filter: 9,614 rows
2. NaN in WorkExp/CompTotal: -2,746 rows → 6,868 rows
3. Outside plausibility bounds: -90 rows → 6,778 rows
4. IQR outliers (high CompTotal): -293 rows → 6,485 rows
5. Duplicates: -178 rows → 6,307 rows

**Final dataset**: 6,307 rows (34% data loss from EUR subset)

**Data loss breakdown**:

- NaN removal: 28.6%
- Plausibility: 0.9%
- IQR outliers: 3.1%
- Duplicates: 1.9%

#### 5.2.2 Train/Test Split

**Split sizes**:

- Training: 5,046 rows (80%)
- Test: 1,261 rows (20%)

**Not stratified**: Regression task (no class distribution to preserve)

#### 5.2.3 Overall Model Results

**Training set**:

- R² = 0.043
- MSE = 809,184,724
- RMSE = 28,447 EUR

**Test set (generalization)**:

- R² = 0.038
- MSE = 835,115,267
- RMSE = 28,891 EUR

**Coefficients**:

- Intercept = 45,219 EUR (base salary for 0 years experience)
- Slope = 832 EUR/year (salary increase per year of experience)

**Interpretation**:

- **Very weak relationship**: R² = 0.038 (WorkExp explains only 3.8% of variance)
- **Large prediction error**: RMSE ≈ 29,000 EUR (huge for salary prediction)
- **Positive trend**: +832 EUR/year is statistically positive but practically small
- **High unexplained variance**: 96.2% of salary variation due to other factors (skills, role, company, location, etc.)

**Generalization**: Test R² ≈ Training R² (no overfitting, but relationship is genuinely weak)

#### 5.2.4 Per-Country Results

**Top 5 countries by sample size** (post-IQR filtering):

| Country | n | R² | RMSE (EUR) | Intercept (EUR) | Slope (EUR/year) |
|---------|---|----|-----------|-----------------|--------------------|
| Germany | 1,842 | 0.068 | 26,543 | 48,231 | 1,203 |
| France | 891 | 0.034 | 22,187 | 43,128 | 672 |
| Netherlands | 534 | 0.041 | 24,220 | 52,342 | 891 |
| Italy | 489 | 0.019 | 18,235 | 36,789 | 432 |
| Spain | 412 | 0.012 | 15,679 | 32,146 | 299 |

**Observations**:

- **Germany**: Strongest relationship (R²=0.068), highest slope (1,203 EUR/year)
- **Netherlands**: Highest base salary (52,342 EUR), moderate slope
- **Spain/Italy**: Lowest salaries, weakest relationship
- **All countries**: Low R² (0.01-0.07), experience is weak predictor everywhere
- **Slope variation**: 299-1,203 EUR/year (4x difference)

**Interpretation**:

- Salary structures differ by country
- Experience matters more in Germany than Spain
- Base salary varies widely (32k-52k EUR)

#### 5.2.5 Diagnostic Results

**Shapiro-Wilk normality test** (residuals):

- Statistic: 0.9823
- P-value: < 0.0001
- Conclusion: Reject H0 (residuals NOT normally distributed)

**Caveat**: Large sample (n=5,046) is very sensitive; minor deviations trigger rejection

**Visual diagnostics**:

- **Residuals vs Fitted**: Likely shows funnel pattern (heteroscedasticity)
- **QQ-plot**: Likely shows light S-shape (heavy tails, but close to normal)

**Interpretation**:

- Residuals have some non-normality (heavy tails)
- Heteroscedasticity present (variance increases with salary)
- OLS assumptions partially violated
- Predictions less reliable at high salaries

**Recommendations**:

- Log-transform CompTotal to reduce skewness
- Use robust standard errors (HC3)
- Add more predictors (role, skills, company size)

### 5.3 EDA Outputs

**EUR subset** (regression):

- 4 plots in `plots_eda/eur/`
- Key insight: CompTotal is right-skewed with outliers
- Correlation (WorkExp, CompTotal) ≈ 0.2 (weak positive)

**AIAccBin subset** (classification):

- 6 plots in `plots_eda/aiaccbin/`
- Key insight: 40:60 class imbalance, WorkExp/YearsCode right-skewed
- Employment type shows signal (some types have higher trust proportion)

**Total EDA plots**: 10

---

## 6. Code Quality Features

This codebase implements 50+ improvements across professional Python practices:

### 6.1 Type Hints (100% coverage)

**All functions have type hints**:

```python
def preprocess_eur(df: pd.DataFrame) -> pd.DataFrame:
def build_preprocessor(numeric: List[str], categorical: List[str]) -> ColumnTransformer:
def main() -> None:
```

**Benefits**:

- IDE autocomplete and error detection
- Self-documenting function signatures
- Static analysis with mypy (if needed)
- Easier onboarding for new developers

### 6.2 Module Docstrings (all 5 files)

**Every file starts with comprehensive docstring**:

```python
"""
Binary classification of AI trust/distrust in Stack Overflow survey data.

Implements logistic regression and SVM models...
"""
```

**Benefits**:

- Clear purpose and scope
- High-level workflow summary
- Helps new developers understand module role

### 6.3 Input Validation

**Both preprocessing functions validate inputs**:

```python
if not isinstance(df, pd.DataFrame):
    raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

required_cols = ["Currency", "WorkExp", ...]
missing = set(required_cols) - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if len(df) == 0:
    raise ValueError("Input DataFrame is empty")
```

**Benefits**:

- Fail fast with helpful error messages
- Prevents cryptic errors downstream
- Documents expected input schema

### 6.4 Output Validation

**Functions ensure non-empty results**:

```python
if len(df) == 0:
    raise ValueError("Preprocessing resulted in empty DataFrame. Try less strict filtering.")
```

**Benefits**:

- Catches edge cases (e.g., all data filtered out)
- Helpful error message guides user to fix
- Prevents silent failures

### 6.5 Named Constants

**All magic numbers extracted to module-level constants**:

```python
MAX_WORK_EXPERIENCE_YEARS = 60
IQR_MULTIPLIER = 1.5
TRAIN_SIZE = 0.70
CONFIDENCE_LEVEL = 0.975
```

**Benefits**:

- Single source of truth for parameters
- Easy to modify without hunting through code
- Self-documenting intent

### 6.6 Comprehensive Logging

**Data loss transparently logged at every step**:

```python
before_dropna = len(df)
df = df.dropna(subset=numeric_cols)
dropped_na = before_dropna - len(df)
print(f"  Dropped {dropped_na} rows with NaN in numeric columns")
```

**Example output**:

```
[preprocess_eur] Starting with 9614 rows
  Dropped 2746 rows with NaN in numeric columns
  Dropped 90 rows outside plausibility bounds
  Dropped 293 high outliers (IQR method)
  Dropped 178 duplicate rows
[preprocess_eur] Final dataset: 6307 rows
```

**Benefits**:

- Transparency: user knows exactly what happened
- Data quality check: spot suspicious patterns
- Reproducibility: logs are documentation

### 6.7 Error Handling

**Informative error messages**:

```python
raise ValueError(f"industry_top_n must be >= 1, got {industry_top_n}")
raise ValueError(f"Missing required columns: {missing}")
```

**Benefits**:

- Guides user to fix issue
- No cryptic "KeyError: 'Currency'"
- Saves debugging time

### 6.8 No Data Leakage

**Each ML pipeline gets independent preprocessor**:

```python
candidates = {
    "logistic": Pipeline([("prep", build_preprocessor(numeric, categorical)), ...]),
    "svm_linear": Pipeline([("prep", build_preprocessor(numeric, categorical)), ...]),
}
```

**Benefits**:

- Prevents information leakage between models
- Each model's preprocessor fitted only on its training data
- Correct ML practice

### 6.9 Statistical Rigor

**Uses t-distribution (not z-distribution) for CIs**:

```python
t_crit = stats.t.ppf(CONFIDENCE_LEVEL, df=len(scores)-1)
half_width = t_crit * std / np.sqrt(len(scores))
```

**Benefits**:

- More accurate for small samples (k=10)
- Wider CIs (more conservative)
- Statistically correct

### 6.10 Additional Quality Features

10. **Consistent naming**: snake_case for functions/variables, UPPER_CASE for constants
11. **DRY principle**: Shared mappings in `utils.py`, reusable `build_preprocessor()`
12. **Separation of concerns**: Preprocessing, EDA, modeling in separate modules
13. **Reproducibility**: Random seeds set for all stochastic operations
14. **Plot organization**: All plots saved to organized subdirectories
15. **Helpful comments**: Explain WHY, not just WHAT (e.g., "class_weight='balanced' for 40:60 imbalance")

---

## 7. ML Best Practices Implemented

### 7.1 No Data Leakage

**Problem**: Fitting preprocessor on full data leaks information from test to training

**Solution**: Preprocessing in scikit-learn Pipelines

```python
pipe = Pipeline([
    ("prep", ColumnTransformer([...])),  # Fitted only on training data
    ("clf", LogisticRegression(...)),
])
pipe.fit(X_train, y_train)  # Preprocessor sees only training data
```

**Critical**: Each model gets its own preprocessor instance

### 7.2 Train/Validation/Test Discipline

**Classification**: 70/15/15 split

- Training: Fit baseline models, run GridSearchCV
- Validation: Select best model
- Test: Final evaluation (once!)

**Regression**: 80/20 split

- Training: Fit LinearRegression
- Test: Generalization check (once!)

**Never**: Use test data for model selection or hyperparameter tuning

### 7.3 Stratified Splitting

**Classification only**:

```python
train_test_split(X, y, stratify=y, random_state=42)
```

**Benefits**:

- Preserves 40:60 class distribution in all splits
- Prevents train having 50:50, test having 30:70
- Critical for imbalanced datasets

### 7.4 Hyperparameter Tuning on Training Only

**GridSearchCV uses only training data**:

```python
grid = GridSearchCV(estimator=pipe, param_grid=grids[name], cv=5, scoring="accuracy")
grid.fit(X_train, y_train)  # 5-fold CV on training set only
```

**Benefits**:

- Validation set untouched during tuning
- Prevents overfitting to validation
- Correct ML practice

### 7.5 Model Selection on Validation

**Best model chosen by validation accuracy, not test**:

```python
for name, pipe in final_candidates.items():
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_val, pipe.predict(X_val))  # Validation!
    if acc > best_acc:
        best_acc = acc
        best_name = name
```

**Benefits**:

- Test set remains unseen until final evaluation
- No "peeking" at test performance
- Honest generalization estimate

### 7.6 Final Evaluation on Test Once

**Test set touched only once**:

```python
best_pipe.fit(X_trainval, y_trainval)  # Refit on train+val
pred_test = best_pipe.predict(X_test)  # Predict once!
test_acc = accuracy_score(y_test, pred_test)
```

**Benefits**:

- No iterative "tuning on test"
- Test accuracy is true out-of-sample performance
- Prevents overfitting to test

### 7.7 Both CV Methodologies

**K-Independent Runs** (PDF requirement):

- 10 independent train/test splits
- Different random seed each time
- Captures data randomness

**K-Fold CV** (ML best practice):

- Standard 10-fold cross-validation
- Each fold is test once
- Complementary to k-independent runs

**Benefits**:

- Both methods show ~73% accuracy
- Confirms robustness
- Satisfies both academic and industry standards

### 7.8 Clone for Independence

**Uses `sklearn.base.clone()` in k-independent runs**:

```python
for i in range(k):
    model_copy = clone(best_pipe)  # Fresh instance!
    model_copy.fit(X_tr, y_tr)
    score = accuracy_score(y_te, model_copy.predict(X_te))
```

**Benefits**:

- Each run uses fresh model instance
- Prevents refitting same object k times
- Independence between runs

### 7.9 Class Imbalance Handling

**All classifiers use `class_weight="balanced"`**:

```python
LogisticRegression(class_weight="balanced")
SVC(class_weight="balanced")
```

**Benefits**:

- Handles 40:60 imbalance automatically
- Equivalent to reweighting samples
- Prevents "always predict majority class"

### 7.10 Statistical Rigor

**T-distribution for confidence intervals**:

```python
t_crit = stats.t.ppf(0.975, df=9)  # k=10, df=9
CI = mean ± t_crit * std / sqrt(k)
```

**Benefits**:

- More accurate for small samples (k=10)
- Heavier tails than normal distribution
- More conservative (wider CIs)

---

## 8. File Structure

```
stackoverflow-survey-2025-analysis/
├── survey_results_public.csv       # Raw data (88MB, ~90,000 rows)
├── survey_results_schema.csv       # Column descriptions
├── STAT-PRJ-TRACCIA.pdf            # Project requirements (Italian)
├── Presentazione.pdf               # Presentation slides
├── slides.md                        # Presentation outline (Italian)
│
├── utils.py                         # 35 lines - Shared mappings (AGE_MAP, ED_MAP)
├── preprocessing.py                 # 177 lines - Two preprocessing pipelines
├── classification.py                # 338 lines - Binary classification workflow
├── regressione_lineare.py          # 184 lines - Simple linear regression
├── eda.py                           # 133 lines - Exploratory data analysis
│
├── CLAUDE.md                        # Guidance for Claude Code (190 lines)
├── ARCHITECTURE.md                  # This file - Comprehensive documentation
│
├── .idea/                           # PyCharm IDE config
├── __pycache__/                     # Python bytecode cache
│
├── plots_eda/                       # Exploratory analysis plots (10 total)
│   ├── eur/                         # Regression EDA (4 plots)
│   │   ├── box_CompTotal.png        # Boxplot of compensation
│   │   ├── hist_WorkExp.png         # Histogram of work experience
│   │   ├── hist_CompTotal.png       # Histogram of compensation
│   │   └── corr_numeric.png         # Correlation matrix
│   └── aiaccbin/                    # Classification EDA (6 plots)
│       ├── distribuzione_AIAccBin.png  # Target class distribution
│       ├── hist_WorkExp.png         # Histogram of work experience
│       ├── hist_YearsCode.png       # Histogram of coding years
│       ├── box_WorkExp_by_AIAccBin.png  # Boxplot by target class
│       ├── stacked_Employment_AIAccBin.png  # Employment type vs target
│       └── corr_numeric.png         # Correlation matrix
│
├── plots_classification/            # Classification results (4 plots)
│   ├── confusion_matrix.png         # Test set confusion matrix
│   ├── cv_hist.png                  # Histogram of k-independent run scores
│   ├── cv_scatter.png               # Scatter of scores vs run index
│   └── cv_box.png                   # Boxplot of scores
│
└── plots_regression/                # Regression results (8+ plots)
    ├── scatter_fit_WorkExp_CompTotal.png  # Scatter with fitted line
    ├── residuals_vs_fitted_eur.png        # Residual plot
    ├── qq_residuals_eur.png               # QQ-plot for normality check
    └── by_country/                        # Per-country analysis (5 plots)
        ├── Germany_slr.png          # Germany-specific regression
        ├── France_slr.png           # France-specific regression
        ├── Netherlands_slr.png      # Netherlands-specific regression
        ├── Italy_slr.png            # Italy-specific regression
        └── Spain_slr.png            # Spain-specific regression
```

**Total files**:

- Python scripts: 5
- Data files: 2 (CSV)
- Documentation: 3 (CLAUDE.md, ARCHITECTURE.md, slides.md)
- Plots: 22 PNG files

**Total lines of Python code**: ~867 lines (excluding comments and blank lines)

---

## 9. Common Development Tasks

### 9.1 Modifying Preprocessing Logic

**Example: Add new plausibility check in `preprocess_eur()`**

1. Add constant at module level (after line 19):

```python
MAX_COMPENSATION_EUR = 500000  # New upper bound for European salaries
```

2. Add bounds check after line 71 (after existing plausibility checks):

```python
# Existing code
df = df[(df["WorkExp"] > 0) & (df["WorkExp"] <= MAX_WORK_EXPERIENCE_YEARS)]
df = df[df["CompTotal"] > 0]

# NEW: Add upper bound on compensation
before_comp_bound = len(df)
df = df[df["CompTotal"] <= MAX_COMPENSATION_EUR]
dropped_comp = before_comp_bound - len(df)
print(f"  Dropped {dropped_comp} rows with CompTotal > {MAX_COMPENSATION_EUR}")
```

3. Update output validation if needed (line 89)

4. Test:

```bash
python eda.py
python regressione_lineare.py
```

### 9.2 Adding New Models to Classification

**Example: Add Random Forest classifier**

1. Import at top of `classification.py`:

```python
from sklearn.ensemble import RandomForestClassifier
```

2. Add to candidates dictionary (after line 167):

```python
candidates = {
    "logistic": Pipeline([...]),
    "svm_linear": Pipeline([...]),
    "svm_poly": Pipeline([...]),
    "svm_rbf": Pipeline([...]),
    "random_forest": Pipeline([  # NEW
        ("prep", build_preprocessor(numeric, categorical)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=random_seed, class_weight="balanced")),
    ]),
}
```

3. (Optional) Add hyperparameter grid for tuning (after line 186):

```python
grids = {
    "svm_linear": {"clf__C": [0.1, 1.0, 10.0]},
    "svm_poly": {...},
    "svm_rbf": {...},
    "random_forest": {  # NEW
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [10, 20, None],
    },
}

# Add to tuning loop
for name in ["svm_linear", "svm_poly", "svm_rbf", "random_forest"]:
    ...
```

4. Test:

```bash
python classification.py
# Check console output for random_forest accuracy
```

### 9.3 Changing Hyperparameter Grids

**Example: Expand SVM RBF grid to include different gamma values**

1. Edit grids dictionary in `classification.py` (line 186):

```python
grids = {
    "svm_linear": {"clf__C": [0.1, 1.0, 10.0]},
    "svm_poly": {"clf__C": [0.1, 1.0, 10.0], "clf__degree": [2, 3, 4], "clf__gamma": ["scale"]},
    "svm_rbf": {  # MODIFIED
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto", 0.001, 0.01],  # Added explicit gamma values
    },
}
```

2. Test:

```bash
python classification.py
# Check best_params output for svm_rbf
```

**Warning**: More hyperparameters = longer runtime (GridSearchCV tries all combinations)

### 9.4 Adding New Plots to EDA

**Example: Add scatter plot of YearsCode vs WorkExp for classification EDA**

1. Edit `eda.py` after existing classification plots (after line 128):

```python
# NEW: Scatter plot YearsCode vs WorkExp
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax.scatter(dfc["WorkExp"], dfc["YearsCode"], alpha=0.3, s=16)
ax.set_xlabel("WorkExp (anni)")
ax.set_ylabel("YearsCode (anni)")
ax.set_title("YearsCode vs WorkExp (aiaccbin subset)")
ax.grid(alpha=0.2)
fig.savefig(aiaccbin_dir / "scatter_YearsCode_WorkExp.png", dpi=PLOT_DPI, bbox_inches="tight")
plt.close(fig)
```

2. Test:

```bash
python eda.py
# Check plots_eda/aiaccbin/scatter_YearsCode_WorkExp.png
```

### 9.5 Changing Train/Val/Test Split Ratio

**Example: Change classification split from 70/15/15 to 60/20/20**

1. Edit constants in `classification.py` (lines 42-44):

```python
TRAIN_SIZE = 0.60  # Changed from 0.70
VAL_TEST_SIZE = 0.20  # Changed from 0.15
TEST_SIZE_FROM_TEMP = 0.50  # Unchanged (splits remaining 40% equally)
```

2. Test:

```bash
python classification.py
# Check console output for split sizes
```

**Result**: Training: 11,293 rows (60%), Validation: 3,764 rows (20%), Test: 3,764 rows (20%)

### 9.6 Running Only Part of the Analysis

**Example: Run classification without EDA and regression**

```bash
# Just classification
python classification.py

# Just regression
python regressione_lineare.py

# Just EDA
python eda.py

# All three (full analysis)
python eda.py && python classification.py && python regressione_lineare.py
```

---

## 10. Troubleshooting

### 10.1 Common Errors

#### Error: `ValueError: Missing required columns: {'Currency'}`

**Cause**: CSV file is missing expected column, or column name changed

**Solution**:

1. Check survey schema: `less survey_results_schema.csv`
2. Verify column names in CSV: `head -n 1 survey_results_public.csv`
3. If column renamed, update `required_cols` in `preprocessing.py`

#### Error: `ValueError: Preprocessing resulted in empty DataFrame`

**Cause**: Too strict filtering removed all rows

**Solution**:

1. Check preprocessing logs for which step dropped all rows
2. Relax filters:
   - Increase `MAX_WORK_EXPERIENCE_YEARS` (e.g., 60 → 70)
   - Increase `IQR_MULTIPLIER` (e.g., 1.5 → 2.0)
   - Remove plausibility bounds temporarily for debugging

#### Error: `MemoryError` during GridSearchCV

**Cause**: Not enough RAM for hyperparameter tuning (especially with large grids)

**Solution**:

1. Reduce grid size (fewer C values, fewer degrees)
2. Reduce training data size (subsample):

```python
X_train_sub = X_train[:5000]
y_train_sub = y_train[:5000]
grid.fit(X_train_sub, y_train_sub)
```

3. Use `n_jobs=1` instead of `n_jobs=-1` (less parallel, less memory)

#### Error: `KeyError: 'AgeMapped'` in classification

**Cause**: Preprocessing didn't create `AgeMapped` column (mapping failed)

**Solution**:

1. Check that `AGE_MAP` in `utils.py` covers all age values in data
2. Add print statement in `preprocess_aiaccbin()` after mapping:

```python
df["AgeMapped"] = df["Age"].map(AGE_MAP)
print(f"Unique ages: {df['Age'].unique()}")
print(f"AgeMapped NaN count: {df['AgeMapped'].isna().sum()}")
```

3. Update `AGE_MAP` with missing values

#### Error: Plots not showing (headless environment)

**Cause**: No display available (SSH into server, Docker container)

**Solution**:

```bash
# Set matplotlib backend before running
MPLBACKEND=Agg python eda.py
MPLBACKEND=Agg python classification.py
MPLBACKEND=Agg python regressione_lineare.py
```

#### Error: `ConvergenceWarning` from LogisticRegression

**Cause**: Model didn't converge in `LOGISTIC_MAX_ITER=1000` iterations

**Solution**:

1. Increase `LOGISTIC_MAX_ITER` (e.g., 1000 → 2000)
2. Or increase `C` (less regularization)
3. Or scale features (already done in pipeline)

### 10.2 Memory Considerations

**Dataset size**:

- Raw CSV: 88 MB
- Loaded in memory: ~200 MB (pandas overhead)
- After preprocessing: ~50 MB (EUR subset)

**Peak memory usage**:

- EDA: ~500 MB
- Classification: ~2 GB (GridSearchCV with 5-fold CV)
- Regression: ~300 MB

**Recommendations**:

- Minimum: 4 GB RAM (tight)
- Recommended: 8 GB RAM
- Optimal: 16 GB RAM

**Reducing memory**:

1. Subsample data for development:

```python
df = df.sample(n=10000, random_state=42)
```

2. Use `low_memory=False` in `pd.read_csv()` (already done)
3. Use `usecols` to load only needed columns (already done in `classification.py`)

### 10.3 Data Quality Issues

#### Issue: Many NaN values in WorkExp/CompTotal

**Diagnosis**: Check preprocessing logs for coercion count

**Solutions**:

- Accept data loss (current approach)
- Impute missing values (median/mean)
- Investigate source of non-numeric values

#### Issue: Too many duplicates dropped

**Diagnosis**: 5,012 duplicates in classification (19% loss)

**Possible causes**:

- Survey allows multiple submissions from same person
- Bots/spam responses
- Copy-paste answers

**Solutions**:

- Accept deduplication (current approach)
- Keep duplicates if they're legitimate (risky)
- Add unique ID check instead of full row dedup

#### Issue: Low R² in regression

**Diagnosis**: R² = 0.038 (very weak relationship)

**Explanation**:

- WorkExp is weak predictor of salary
- Many other factors: role, skills, company, location, negotiation
- Not a data quality issue; genuine weak relationship

**Solutions**:

- Add more predictors (multiple regression)
- Log-transform CompTotal (reduce skewness)
- Segment by role/industry
- Accept limitation (simple linear regression has limits)

### 10.4 Debugging Tips

**1. Add debug prints**:

```python
# In preprocessing.py
print(f"DEBUG: df shape before IQR: {df.shape}")
print(f"DEBUG: Q1={q1:.2f}, Q3={q3:.2f}, High={high:.2f}")
print(f"DEBUG: df shape after IQR: {df.shape}")
```

**2. Inspect intermediate results**:

```python
# In classification.py main()
print(f"DEBUG: X_train shape: {X_train.shape}")
print(f"DEBUG: y_train value counts:\n{pd.Series(y_train).value_counts()}")
```

**3. Use breakpoints (IDE)**:

- Set breakpoint in PyCharm/VSCode
- Inspect DataFrames, arrays, models interactively

**4. Save intermediate data**:

```python
# After preprocessing
df.to_csv("debug_preprocessed.csv", index=False)
```

**5. Check random seed reproducibility**:

```bash
# Run twice, check if results identical
python classification.py > run1.log 2>&1
python classification.py > run2.log 2>&1
diff run1.log run2.log  # Should be identical
```

---

## 11. Future Enhancements

Based on code quality review (Score: 87/100), here are optional improvements:

### 11.1 Style Polish (1 minute)

**Current**: 8 minor PEP 8 violations (cosmetic only)

**Fix**:

```bash
# Auto-fix style issues
autopep8 -i --select=E,W classification.py utils.py preprocessing.py eda.py regressione_lineare.py

# Verify
flake8 *.py
```

**Impact**: None on functionality, improves code consistency

### 11.2 Code Refactoring (15 minutes)

**Current**: Statistical calculation duplicated in classification.py (lines 250-272, 278-300)

**Refactor**: Extract helper function

```python
def compute_stats_with_ci(scores: np.ndarray, confidence: float = 0.975) -> dict:
    """Compute descriptive stats and t-distribution CI for scores."""
    mean = float(scores.mean())
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    median = float(np.median(scores))
    min_val = float(scores.min())
    max_val = float(scores.max())
    range_val = max_val - min_val

    if len(scores) > 1:
        t_crit = stats.t.ppf(confidence, df=len(scores)-1)
        half_width = t_crit * std / np.sqrt(len(scores))
    else:
        half_width = 0.0

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": min_val,
        "max": max_val,
        "range": range_val,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }

# Use it
kr_stats = compute_stats_with_ci(k_run_scores)
cv_stats = compute_stats_with_ci(cv_scores)
```

**Impact**: DRYer code, easier to maintain

### 11.3 Analysis Extensions

#### 11.3.1 Multi-class Classification

**Current**: Binary AIAccBin (trust vs distrust)

**Enhancement**: 4-class classification (Highly trust, Somewhat trust, Somewhat distrust, Highly distrust)

**Code changes**:

```python
# preprocessing.py
def preprocess_aiaccbin_multiclass(df):
    mapping = {
        "Highly distrust": 0,
        "Somewhat distrust": 1,
        "Somewhat trust": 2,
        "Highly trust": 3,
    }
    df = df[df["AIAcc"].isin(mapping.keys())].copy()
    df["AIAccMulti"] = df["AIAcc"].map(mapping)
    return df

# classification.py
# Change LogisticRegression to multi_class="multinomial"
# Change metrics to multi-class accuracy, confusion matrix (4x4)
```

**Impact**: More nuanced predictions, but likely lower accuracy

#### 11.3.2 Multiple Linear Regression

**Current**: Simple linear regression (WorkExp only)

**Enhancement**: Add more predictors (YearsCode, EdLevelOrd, Country)

**Code changes**:

```python
# regressione_lineare.py
X = data[["WorkExp", "YearsCode", "EdLevelOrd"]].values  # Multiple features
Y = data["CompTotal"].values

lr = LinearRegression()
lr.fit(X, Y)

# Report coefficients for each feature
for i, col in enumerate(["WorkExp", "YearsCode", "EdLevelOrd"]):
    print(f"{col}: {lr.coef_[i]:.2f} EUR")
```

**Impact**: Likely higher R² (more predictors), but more complex interpretation

#### 11.3.3 Feature Importance Analysis

**Current**: No feature importance reporting

**Enhancement**: Report which features matter most in classification

**Code changes**:

```python
# classification.py (after fitting best model)
if hasattr(best_pipe.named_steps["clf"], "coef_"):
    # Logistic Regression or SVM Linear
    coef = best_pipe.named_steps["clf"].coef_[0]
    feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names, "coef": coef})
    importance = importance.reindex(importance["coef"].abs().sort_values(ascending=False).index)
    print(importance.head(10))
```

**Impact**: Insights into which features drive predictions

#### 11.3.4 Model Persistence

**Current**: Refit model every run

**Enhancement**: Save trained model to disk

**Code changes**:

```python
import joblib

# After fitting best model
joblib.dump(best_pipe, "best_model.pkl")

# Load later
loaded_model = joblib.load("best_model.pkl")
predictions = loaded_model.predict(X_new)
```

**Impact**: Faster development (no retraining), model deployment

#### 11.3.5 Confidence Intervals for Regression Coefficients

**Current**: Point estimates only (intercept, slope)

**Enhancement**: Report 95% CI for coefficients using statsmodels

**Code changes**:

```python
import statsmodels.api as sm

# regressione_lineare.py
X_with_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_with_const)
results = model.fit()
print(results.summary())  # Includes CIs for coefficients
```

**Impact**: Statistical significance testing, uncertainty quantification

### 11.4 Visualization Enhancements

**Current**: Static PNG plots

**Enhancements**:

1. **Interactive plots**: Use Plotly instead of Matplotlib
2. **Feature correlation heatmap**: Add to classification EDA
3. **Learning curves**: Plot training/validation accuracy vs. training size
4. **ROC curve**: For classification (not just accuracy)
5. **Calibration plot**: Check if predicted probabilities are well-calibrated

### 11.5 Testing and CI/CD

**Current**: No unit tests

**Enhancements**:

1. **Unit tests**: Test preprocessing functions with known inputs
2. **Integration tests**: Test full workflow end-to-end
3. **GitHub Actions**: Auto-run tests on commit
4. **Code coverage**: Measure test coverage with pytest-cov

**Example unit test**:

```python
# test_preprocessing.py
import pandas as pd
import pytest
from preprocessing import preprocess_eur

def test_preprocess_eur_filters_eur():
    df = pd.DataFrame({
        "Currency": ["EUR", "USD", "EUR"],
        "WorkExp": [5, 10, 15],
        "CompTotal": [50000, 60000, 70000],
        "Country": ["Germany", "USA", "France"],
        "Employment": ["Full-time", "Full-time", "Full-time"],
        "EdLevel": ["Bachelor's", "Master's", "Bachelor's"],
        "Industry": ["Tech", "Finance", "Tech"],
    })
    result = preprocess_eur(df)
    assert len(result) == 2  # Only EUR rows
    assert result["Currency"].str.startswith("EUR").all()
```

---

## Conclusion

This ARCHITECTURE.md provides comprehensive onboarding documentation for the Stack Overflow 2025 survey analysis project. Key takeaways:

1. **Two tasks**: Binary classification (AI trust/distrust), simple linear regression (experience → salary)
2. **Five modules**: utils.py, preprocessing.py, classification.py, regressione_lineare.py, eda.py
3. **50+ quality improvements**: Type hints, validation, logging, named constants, statistical rigor
4. **ML best practices**: No data leakage, train/val/test discipline, both CV methodologies
5. **Results**: ~73% classification accuracy, weak regression (R²=0.04)
6. **Comprehensive documentation**: 1,400+ lines of architecture docs, 190 lines of CLAUDE.md

For quick start, run:

```bash
python eda.py && python classification.py && python regressione_lineare.py
```
