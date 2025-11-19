# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic statistical analysis project analyzing the Stack Overflow 2025 Developer Survey dataset. The project performs:
1. **Binary classification** of AI trust/distrust (`AIAccBin`) using Logistic Regression and SVM variants
2. **Simple linear regression** predicting compensation (`CompTotal` in EUR) from work experience (`WorkExp`)

The codebase follows a modular structure with separate preprocessing, EDA, and modeling scripts. The project has undergone extensive refactoring with 50+ improvements including type hints, comprehensive logging, input/output validation, and statistical rigor.

## Running the Analysis

Execute scripts in this order to reproduce the complete analysis:

```bash
# 1. Exploratory Data Analysis - generates plots in plots_eda/
python eda.py

# 2. Classification task - generates plots in plots_classification/
python classification.py

# 3. Linear regression task - generates plots in plots_regression/
python regressione_lineare.py
```

For headless environments (no display):
```bash
MPLBACKEND=Agg python eda.py
MPLBACKEND=Agg python classification.py
MPLBACKEND=Agg python regressione_lineare.py
```

## Dependencies

Required Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- scipy

## Code Quality Features

This codebase implements professional Python practices:

1. **Type Hints**: 100% coverage on all function signatures for IDE support and static analysis
2. **Module Docstrings**: Comprehensive documentation in all 5 Python files explaining purpose and workflow
3. **Input Validation**: Both preprocessing functions validate input types, required columns, and empty DataFrames
4. **Output Validation**: Functions ensure they don't return empty results with helpful error messages
5. **Named Constants**: All magic numbers extracted to module-level constants (MAX_WORK_EXPERIENCE_YEARS=60, IQR_MULTIPLIER=1.5, etc.)
6. **Comprehensive Logging**: Data loss transparently logged at every preprocessing step (NaN removal, outlier filtering, deduplication)
7. **Error Handling**: Informative error messages guide users when validation fails
8. **No Data Leakage**: Each ML pipeline creates independent preprocessor instances; preprocessing fitted only on training data
9. **Statistical Rigor**: Uses t-distribution (not z-distribution) for confidence intervals with small samples (k=10)

## Architecture & Data Flow

### 1. Preprocessing (`preprocessing.py`)

Two main preprocessing pipelines with comprehensive validation and logging:

**`preprocess_eur(df: pd.DataFrame) -> pd.DataFrame`** - For regression task:
- **Input validation**: Type check, required columns verification, empty DataFrame check
- Filters to EUR currency subset (Currency starts with "EUR")
- Extracts: `WorkExp`, `CompTotal`, `Country`, `Employment`, `EdLevel`, `Industry`
- Numeric conversion with coercion logging (reports non-numeric values converted to NaN)
- Applies plausibility bounds: `0 < WorkExp ≤ MAX_WORK_EXPERIENCE_YEARS (60)`, `CompTotal > 0`
- Removes high outliers using `IQR_MULTIPLIER (1.5)` × IQR method on `CompTotal`
- Deduplicates with logging
- **Output validation**: Ensures non-empty result
- **Comprehensive logging**: Reports row count after each step with number of rows dropped

**`preprocess_aiaccbin(df: pd.DataFrame, industry_top_n: int, devtype_top_n: int) -> pd.DataFrame`** - For classification task:
- **Input validation**: Type check, parameter validation, required columns check, empty DataFrame check
- Filters `AIAcc` to 4 trust/distrust categories, creates binary `AIAccBin` (1=trust, 0=distrust)
- Numeric conversion with coercion logging
- Applies bounds: `0 < WorkExp ≤ MAX_WORK_EXPERIENCE_YEARS (60)`, `0 ≤ YearsCode ≤ MAX_YEARS_CODE (60)`
- Maps categorical Age → `AgeMapped` (numeric midpoint) and EdLevel → `EdLevelOrd` (ordinal 1-7) using shared `utils.py` mappings
- Reduces cardinality: keeps top-N for `Industry`/`DevType`, groups rest as "Other"
- Deduplicates with logging
- **Output validation**: Ensures non-empty result
- **Comprehensive logging**: Reports row count after each step with reason for drops

### 2. Exploratory Data Analysis (`eda.py`)

Generates two sets of plots:
- **EUR subset** (`plots_eda/eur/`): boxplots, histograms, correlation matrix for regression features
- **AIAccBin subset** (`plots_eda/aiaccbin/`): class distribution, histograms, boxplots by target, stacked bar charts, correlation matrix

### 3. Classification (`classification.py`)

**Pipeline architecture - NO DATA LEAKAGE:**
- Uses `ColumnTransformer` with separate pipelines for numeric and categorical features
- **Critical**: Each model gets its own independent `build_preprocessor()` instance to prevent data leakage
- Numeric features (WorkExp, YearsCode, AgeMapped, EdLevelOrd): `StandardScaler`
- Categorical features (Employment, IndustryTop, RemoteWork, OrgSize, ICorPM, DevTypeTop, AISelect, AIAgents, AIModelsChoice): `OneHotEncoder` with `handle_unknown="ignore"`

**Train/Val/Test split:**
- 70% Train, 15% Validation, 15% Test (stratified by target to preserve 40:60 class imbalance)
- Constants: `TRAIN_SIZE=0.70`, `VAL_TEST_SIZE=0.15`, `TEST_SIZE_FROM_TEMP=0.50`

**Model selection workflow:**
1. Baseline evaluation on validation set: Logistic Regression, SVM (linear/poly/rbf) with `DEFAULT_SVM_C=1.0`, `class_weight="balanced"`
2. Hyperparameter tuning via `GridSearchCV` (5-fold CV on training set only):
   - SVM linear: `C ∈ {0.1, 1.0, 10.0}`
   - SVM poly: `C ∈ {0.1, 1.0, 10.0}`, `degree ∈ {2, 3, 4}`, `gamma='scale'`
   - SVM rbf: `C ∈ {0.1, 1.0, 10.0}`, `gamma='scale'`
3. Select best model based on validation accuracy
4. Refit on Train+Val combined, evaluate once on Test set
5. **Statistical study - BOTH methodologies**:
   - **K-Independent Runs** (k=10, PDF requirement): Different random train/test splits, captures data randomness
   - **K-Fold Cross-Validation** (k=10, ML best practice): Standard CV for comparison
   - Both use **t-distribution** (not z-distribution) for 95% confidence intervals (more conservative for k=10 samples)
   - Confidence level: `CONFIDENCE_LEVEL=0.975` (two-tailed)

**Outputs:**
- Confusion matrix (`plots_classification/confusion_matrix.png`)
- CV diagnostic plots: histogram, scatter, boxplot of k-independent run scores (`plots_classification/cv_*.png`)
- Console output: Detailed statistics for both CV methodologies (mean, std, median, min, max, range, 95% CI)

**Key implementation notes:**
- `class_weight="balanced"` handles moderate class imbalance (~40:60 in AIAccBin)
- `LOGISTIC_MAX_ITER=1000` ensures convergence
- Each pipeline uses `clone()` to avoid refitting same instance
- Random seeds: `DEFAULT_RANDOM_SEED=42` for reproducibility

### 4. Linear Regression (`regressione_lineare.py`)

**Simple linear regression:** `WorkExp → CompTotal` (EUR subset)

**Workflow with train/test split:**
1. Load and preprocess EUR subset via `preprocess_eur()`
2. **Train/test split**: 80% training, 20% test (`TEST_SIZE=0.2`, `RANDOM_STATE=42`)
3. Fit `LinearRegression()` on training data only
4. Evaluate on both training and test sets for generalization assessment
5. Report metrics (training and test): R², MSE, RMSE, intercept, slope
6. Generate diagnostic plots: scatter+fit line, residuals vs fitted (on training), QQ-plot
7. Statistical test: **Shapiro-Wilk normality test** on residuals (H0: residuals are normally distributed)
8. Per-country analysis: fit separate SLR for `TOP_N_COUNTRIES=5` countries (by sample size), with additional `IQR_MULTIPLIER=1.5` filtering per country

**Constants:**
- `TEST_SIZE=0.2`, `RANDOM_STATE=42`, `PLOT_DPI=150`
- `SCATTER_ALPHA=0.3`, `SCATTER_SIZE=16`
- `FIGSIZE_LARGE=(10,7)`, `FIGSIZE_MEDIUM=(9,6)`, `FIGSIZE_QQPLOT=(8,8)`

**Outputs:**
- Global regression plots (`plots_regression/scatter_fit_WorkExp_CompTotal.png`, `residuals_vs_fitted_eur.png`, `qq_residuals_eur.png`)
- Per-country plots (`plots_regression/by_country/{Country}_slr.png`)
- Console output: Training/test metrics, per-country statistics, Shapiro-Wilk test results

## Shared Utilities (`utils.py`)

Contains two mapping dictionaries with type hints used by preprocessing:
- `AGE_MAP: Dict[str, int]`: Maps age range strings to numeric midpoints (e.g., "18-24 years old" → 21, "25-34 years old" → 30, up to "65 years or older" → 70)
- `ED_MAP: Dict[str, int]`: Maps education level strings to ordinal values (1=Primary through 7=Professional degree, with "Other" → 3)

## Data Files

- `survey_results_public.csv`: Main survey data (loaded by all scripts)
- `survey_results_schema.csv`: Schema/column descriptions (not used in code)
- `STAT-PRJ-TRACCIA.pdf`: Project requirements (Italian)
- `slides.md`: Presentation outline with analysis methodology

## Important Conventions

1. **No data leakage:** Preprocessing (scaling, encoding) is always done within scikit-learn Pipelines; each model gets its own preprocessor instance; fitted only on training data
2. **Test set discipline:** Test set is evaluated exactly once at the end; validation set is used for model selection; never use test data for hyperparameter tuning
3. **Reproducibility:** Random seeds are set (`RANDOM_STATE=42` or `DEFAULT_RANDOM_SEED=42` for train/test split; incremented seeds for k-independent runs)
4. **Output organization:** All plots are saved to subdirectories (`plots_eda/eur/`, `plots_eda/aiaccbin/`, `plots_classification/`, `plots_regression/`, `plots_regression/by_country/`)
5. **Language mix:** Code comments and variable names are primarily in English; slides.md is in Italian
6. **Named constants:** All magic numbers are extracted to module-level constants at the top of each file for easy modification
7. **Stratified splitting:** All train/test/validation splits use `stratify=y` to preserve class distribution in classification
8. **Statistical rigor:** Confidence intervals use t-distribution (scipy.stats.t.ppf) for small sample sizes, not z-distribution

## ML Best Practices Implemented

1. **Independent preprocessor instances**: Each model in classification.py gets its own `build_preprocessor()` call to prevent data leakage between models
2. **Train/validation/test discipline**: Classification uses 70/15/15 split; regression uses 80/20 split
3. **Hyperparameter tuning on training only**: GridSearchCV uses only training data with 5-fold CV
4. **Model selection on validation**: Best model chosen based on validation accuracy, never test accuracy
5. **Final evaluation on test once**: Test set touched only once at the very end
6. **Both CV methodologies**: K-independent runs (PDF requirement) and k-fold CV (ML best practice) for comprehensive statistical validation
7. **Clone for independence**: Uses `sklearn.base.clone()` to create fresh model instances in k-independent runs
8. **Stratified splits**: Preserves class distribution in classification task (~40:60 imbalance)
