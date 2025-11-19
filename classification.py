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

# Costruzione del preprocessore per le feature numeriche e categoriche
# OneHotEncoding per le categoriche, StandardScaler per le numeriche
# Pipeline e ColumnTransformer per gestire il flusso in modo modulare e pulito (no data leakage)
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

# Caricamento e pulizia del DataFrame
# Selezione delle colonne rilevanti e applicazione del preprocess specifico per AIAcc
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

PLOT_DPI = 150  # DPI for saved plots
HISTOGRAM_BINS = 10  # Number of bins for histograms


def plot_confusion(cm: np.ndarray, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="viridis", aspect="equal")

    # etichette assi
    ax.set_xticks([0, 1], labels=["False", "True"])
    ax.set_yticks([0, 1], labels=["False", "True"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # numeri al centro cella
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="red")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=PLOT_DPI)
    plt.close(fig)


# Intero flusso di lavoro per la classificazione binaria, con SVM e Logistic Regression
# Hyperparameter tuning e valutazione finale
def main(
    path: str = "survey_results_public.csv",
    random_seed: int = DEFAULT_RANDOM_SEED,
    k: int = DEFAULT_K_FOLDS
) -> None:
    df = load_clean_df(path)

    # Split 70/15/15 Train/Val/Test
    y = df["AIAccBin"].astype(int).values
    X = df.drop(columns=["AIAcc", "AIAccBin"]).copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=1-TRAIN_SIZE, stratify=y, random_state=random_seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE_FROM_TEMP, stratify=y_temp, random_state=random_seed + 1
    )

    # Definizione delle feature numeriche e categoriche
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

    # Definizione dei modelli candidati per la classificazione
    candidates = {
        "logistic": Pipeline([
            ("prep", build_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(max_iter=LOGISTIC_MAX_ITER, C=DEFAULT_SVM_C, class_weight="balanced")), # balanced per gestire sbilanciamento 40:60 in AIAccBin
        ]),
        "svm_linear": Pipeline([
            ("prep", build_preprocessor(numeric, categorical)),
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

    # Valutazione baseline su validation (accuracy)
    print("Accuratezza baseline su validation:")
    val_scores = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_val, pipe.predict(X_val))
        val_scores[name] = float(acc)
        print(f"  {name}: {acc:.3f}")

    # Hyperparameter tuning (GridSearchCV) sui SVM
    tuned = {}

    # Definizione delle griglie di ricerca per ciascun SVM
    grids = {
        "svm_linear": {"clf__C": [0.1, 1.0, 10.0]},
        "svm_poly": {"clf__C": [0.1, 1.0, 10.0], "clf__degree": [2, 3, 4], "clf__gamma": ["scale"]},
        "svm_rbf": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale"]},
    }

    # Esecuzione del GridSearchCV per ciascun SVM
    # Obiettivo: massimizzare l'accuracy su validation
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

    # Selezione finale su validation tra tutti i candidati (logistic + tuned SVM)
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

    # Valutazione finale su Test (dopo refit su Train+Val)
    X_trainval = pd.concat([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val])
    best_pipe.fit(X_trainval, y_trainval)
    pred_test = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, pred_test)
    print(f"Accuratezza su test: {test_acc:.3f}")

    # Confusion matrix dell’ultimo test
    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    outdir = Path("plots_classification")
    plot_confusion(cm, outdir)

    # Studio statistico 1: k independent runs (as per PDF requirement)
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

    # Statistiche descrittive per k-runs
    mean_kr = float(k_run_scores.mean())
    std_kr = float(k_run_scores.std(ddof=1)) if len(k_run_scores) > 1 else 0.0
    median_kr = float(np.median(k_run_scores))
    min_kr = float(k_run_scores.min())
    max_kr = float(k_run_scores.max())
    rng_kr = max_kr - min_kr
    # Use t-distribution for CI (more accurate for small samples)
    # t-critical value for 95% CI (two-tailed) with n-1 degrees of freedom
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

    # Studio statistico 2: k-fold Cross-Validation (best practice)
    print("\n=== Studio Statistico 2: K-Fold Cross-Validation ===")
    cv_scores = cross_val_score(best_pipe, X_trainval, y_trainval, cv=k, scoring="accuracy")

    # Statistiche descrittive per k-fold CV
    mean_cv = float(cv_scores.mean())
    std_cv = float(cv_scores.std(ddof=1)) if len(cv_scores) > 1 else 0.0
    median_cv = float(np.median(cv_scores))
    min_cv = float(cv_scores.min())
    max_cv = float(cv_scores.max())
    rng_cv = max_cv - min_cv
    # Use t-distribution for CI (more accurate for small samples)
    # t-critical value for 95% CI (two-tailed) with n-1 degrees of freedom
    if len(cv_scores) > 1:
        t_crit_cv = stats.t.ppf(CONFIDENCE_LEVEL, df=len(cv_scores)-1)  # 95% CI, two-tailed
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

    # Use k_run_scores for plots (PDF requirement), but keep cv_scores available
    scores = k_run_scores  # Use independent runs for plots

    outdir.mkdir(exist_ok=True)

    # Diagramma a barre per i punteggi di ciascun run indipendente
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(scores, bins=HISTOGRAM_BINS, edgecolor="black")
    ax.set_xlabel("Punteggio")
    ax.set_ylabel("Frequenza")
    ax.set_title("Distribuzione dei punteggi dei K Independent Runs")
    fig.tight_layout()
    fig.savefig(outdir / "cv_hist.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Scatter dei punteggi di ciascun run indipendente
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(range(len(scores)), scores)
    ax.set_xlabel("Indice del run")
    ax.set_ylabel("Punteggio")
    ax.set_title("Distribuzione dei punteggi dei K Independent Runs")
    fig.tight_layout()
    fig.savefig(outdir / "cv_scatter.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Boxplot dei punteggi di ciascun run indipendente
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.boxplot(scores, vert=True, tick_labels=["K-run scores"])
    ax.set_title("Boxplot dei punteggi dei K Independent Runs")
    fig.tight_layout()
    fig.savefig(outdir / "cv_box.png", dpi=PLOT_DPI)
    plt.close(fig)


if __name__ == "__main__":
    main(k=DEFAULT_K_FOLDS)
